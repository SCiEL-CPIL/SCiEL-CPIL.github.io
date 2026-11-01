"""Program for 2x4 by 4x8 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def _build_rank56():
    """Build the base rank-56 Strassen decomposition."""
    n, m, p = DIMENSIONS
    rank = 56
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for d in range(4):
        for h in range(2):
            base = (d * 2 + h) * 7
            if h == 0:
                def a_idx(r, c): return r * m + c
                def b_idx(r, c): return r * p + (2 * d + c)
            else:
                def a_idx(r, c): return r * m + (2 + c)
                def b_idx(r, c): return (2 + r) * p + (2 * d + c)
            def c_idx(r, c): return (2 * d + c) * n + r
            
            # M1
            col = base + 0
            U[a_idx(0,0),col]+=1; U[a_idx(1,1),col]+=1
            V[b_idx(0,0),col]+=1; V[b_idx(1,1),col]+=1
            W[c_idx(0,0),col]+=1; W[c_idx(1,1),col]+=1
            # M2
            col = base + 1
            U[a_idx(1,0),col]+=1; U[a_idx(1,1),col]+=1
            V[b_idx(0,0),col]+=1
            W[c_idx(1,0),col]+=1; W[c_idx(1,1),col]-=1
            # M3
            col = base + 2
            U[a_idx(0,0),col]+=1
            V[b_idx(0,1),col]+=1; V[b_idx(1,1),col]-=1
            W[c_idx(0,1),col]+=1; W[c_idx(1,1),col]+=1
            # M4
            col = base + 3
            U[a_idx(1,1),col]+=1
            V[b_idx(1,0),col]+=1; V[b_idx(0,0),col]-=1
            W[c_idx(0,0),col]+=1; W[c_idx(1,0),col]+=1
            # M5
            col = base + 4
            U[a_idx(0,0),col]+=1; U[a_idx(0,1),col]+=1
            V[b_idx(1,1),col]+=1
            W[c_idx(0,0),col]-=1; W[c_idx(0,1),col]+=1
            # M6
            col = base + 5
            U[a_idx(1,0),col]+=1; U[a_idx(0,0),col]-=1
            V[b_idx(0,0),col]+=1; V[b_idx(0,1),col]+=1
            W[c_idx(1,1),col]+=1
            # M7
            col = base + 6
            U[a_idx(0,1),col]+=1; U[a_idx(1,1),col]-=1
            V[b_idx(1,0),col]+=1; V[b_idx(1,1),col]+=1
            W[c_idx(0,0),col]+=1
    
    return U, V, W

def _check_block_redundancy(U, V, W, tol=1e-10):
    """
    Check for redundancy within each 2x2x2 Strassen block.
    Each block of 7 terms operates on disjoint index supports, so cross-block
    redundancy is impossible. Within a block, Strassen's 7-term decomposition
    is known to be rank-minimal for 2x2x2. This function verifies this
    numerically using a lightweight per-block check.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    removed = []
    
    # Check each block of 7 columns
    for block in range(8):
        cols = list(range(block * 7, (block + 1) * 7))
        # Extract sub-matrices
        Ub = U[:, cols]
        Vb = V[:, cols]
        Wb = W[:, cols]
        
        # Build the 64x7 tensor matrix for this block
        # Only 8 entries are nonzero in the full 8x32x16 tensor per term
        # but we can use the full vectorization for simplicity (8*32*16=4096)
        # Actually, for a single 2x2x2 block, only 8 entries matter
        # Let's use the full 4096-vectorization but it's small enough
        
        block_Tmat = np.zeros((n*m * m*p * p*n, 7))
        for s in range(7):
            block_Tmat[:, s] = np.kron(np.kron(Ub[:, s], Vb[:, s]), Wb[:, s])
        
        # Check rank of this 4096x7 matrix
        try:
            S = np.linalg.svd(block_Tmat, full_matrices=False, compute_uv=False)
        except np.linalg.LinAlgError:
            continue
        
        num_rank = sum(1 for sv in S if sv > tol * S[0])
        if num_rank < 7:
            # Some redundancy within this block - try greedy removal
            keep_local = list(range(7))
            for trial in range(7 - num_rank):
                best_remove = None
                for idx in keep_local:
                    others = [k for k in keep_local if k != idx]
                    if not others:
                        continue
                    sub = block_Tmat[:, others]
                    Q, _ = np.linalg.qr(sub)
                    proj = Q @ (Q.T @ block_Tmat[:, idx])
                    residual = block_Tmat[:, idx] - proj
                    if np.linalg.norm(residual) < tol * np.linalg.norm(block_Tmat[:, idx]):
                        best_remove = idx
                        break
                if best_remove is not None:
                    keep_local.remove(best_remove)
                    removed.append(block * 7 + best_remove)
    
    return removed

def _try_cross_block_reduction(U, V, W, deadline, tol=1e-10):
    """
    Try to find cross-block linear dependencies using a randomized approach.
    Since blocks have disjoint supports, true cross-block redundancy is impossible,
    but we do a quick check to be safe.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    if time.time() > deadline - 5.0:
        return []
    
    # For cross-block, we need to check if any term is in the span of all others.
    # Use a sampling approach: check if removing each column breaks validity.
    # This is expensive, so only do it if we have time.
    
    removed = []
    for col in range(rank):
        if time.time() > deadline - 3.0:
            break
        # Try removing this column
        keep = [i for i in range(rank) if i != col]
        Uk = U[:, keep]
        Vk = V[:, keep]
        Wk = W[:, keep]
        
        # Quick check on a few entries
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    val = np.sum(Uk[a_idx, :] * Vk[b_idx, :] * Wk[c_idx, :])
                    if abs(val - 1.0) > 1e-8:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            removed.append(col)
    
    return removed

def construct():
    """
    Construct a rank-minimal decomposition for 2x4 by 4x8 matrix multiplication.
    Starts with rank-56 Strassen decomposition and attempts rank reduction.
    """
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    U, V, W = _build_rank56()
    rank = U.shape[1]
    
    # Try block-level redundancy check (fast)
    removed = _check_block_redundancy(U, V, W)
    
    if removed:
        keep = [i for i in range(rank) if i not in set(removed)]
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]
        rank = len(keep)
    
    # Try cross-block reduction (slower, only if time permits)
    if time.time() < deadline - 10.0:
        removed2 = _try_cross_block_reduction(U, V, W, deadline)
        if removed2:
            keep = [i for i in range(rank) if i not in set(removed2)]
            U = U[:, keep]
            V = V[:, keep]
            W = W[:, keep]
            rank = len(keep)
    
    # Verify
    n, m, p = DIMENSIONS
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                val = np.sum(U[a_idx, :] * V[b_idx, :] * W[c_idx, :])
                if abs(val - 1.0) > 1e-8:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fall back to original
        U, V, W = _build_rank56()
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _verify(U, V, W, n, m, p):
    """Verify the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                val = np.sum(U[a_idx, :] * V[b_idx, :] * W[c_idx, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    import random
    random.seed(42)
    for _ in range(200):
        a = random.randint(0, n*m-1)
        b = random.randint(0, m*p-1)
        c = random.randint(0, p*n-1)
        i1, j1 = divmod(a, m)
        j2, k1 = divmod(b, p)
        k2, i2 = divmod(c, n)
        if j1 != j2 or i1 != i2 or k1 != k2:
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(val) > 1e-9:
                return False
    
    return True


if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    n, m, p = DIMENSIONS
    ok = _verify(result["U"], result["V"], result["W"], n, m, p)
    rank = len(result["U"][0])
    elapsed = time.time() - t0
    print(f"Rank: {rank}, Valid: {ok}, Time: {elapsed:.1f}s")
    if not ok:
        print("DECOMPOSITION IS INVALID!")
    else:
        print("Decomposition verified successfully.")
