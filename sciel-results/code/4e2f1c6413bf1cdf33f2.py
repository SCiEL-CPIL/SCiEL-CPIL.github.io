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

def _find_redundant(U, V, W, tol=1e-10):
    """
    Try to find linearly dependent rank-1 terms.
    For each column s, check if the tensor T_s = U[:,s] ⊗ V[:,s] ⊗ W[:,s]
    is a linear combination of other columns.
    We flatten each rank-1 term into a vector in R^{n*m*p} and check rank.
    """
    n, m, p = DIMENSIONS
    dim = n * m * p  # 64
    rank = U.shape[1]
    
    # Build tensor matrix: each column s gives a vector of length n*m*p
    # T_s[i,j,k] = U[i,s]*V[j,s]*W[k,s] where i in [0,n*m), j in [0,m*p), k in [0,p*n)
    # But we only care about valid entries where the indices are consistent.
    # Actually, the full tensor lives in R^{(n*m) x (m*p) x (p*n)} but most entries are 0.
    # Let's just use the full vectorization.
    
    T = np.zeros((dim * dim * dim, rank))  # This is too big: 64*32*16 = 32768 rows
    # Actually let's be smarter. The tensor has shape (n*m, m*p, p*n) = (8, 32, 16)
    # Total entries = 4096. Each rank-1 term contributes U[:,s] ⊗ V[:,s] ⊗ W[:,s]
    
    # Build the 4096 x rank matrix
    Tmat = np.zeros((n*m * m*p * p*n, rank))
    for s in range(rank):
        Tmat[:, s] = np.kron(np.kron(U[:, s], V[:, s]), W[:, s])
    
    # Now find which columns are linearly dependent
    # Use SVD to find the numerical rank
    try:
        U_svd, S, Vt = np.linalg.svd(Tmat, full_matrices=False)
    except np.linalg.LinAlgError:
        return list(range(rank)), Tmat
    
    # Determine numerical rank
    num_rank = 0
    for i, sv in enumerate(S):
        if sv > tol * S[0]:
            num_rank = i + 1
        else:
            break
    
    # Identify which original columns are "redundant"
    # A column s is redundant if its projection onto the null space of the other columns is ~0
    # Simpler: use the Vt matrix. Columns of Tmat that are in the span of others
    # will have small norm in the "redundant" directions.
    
    # Alternative approach: greedy elimination
    # Start with all columns, try to remove one at a time
    keep = list(range(rank))
    removed = []
    
    for trial in range(rank):
        if not keep:
            break
        best_remove = None
        best_residual = 0
        for idx in keep:
            # Check if column idx is in span of keep\{idx}
            others = [k for k in keep if k != idx]
            if not others:
                continue
            # Project Tmat[:,idx] onto span of Tmat[:,others]
            sub = Tmat[:, others]
            # Compute residual
            Q, _ = np.linalg.qr(sub)
            proj = Q @ (Q.T @ Tmat[:, idx])
            residual = Tmat[:, idx] - proj
            res_norm = np.linalg.norm(residual)
            if res_norm < tol * np.linalg.norm(Tmat[:, idx]):
                best_remove = idx
                best_residual = res_norm
                break
        
        if best_remove is not None:
            keep.remove(best_remove)
            removed.append(best_remove)
    
    return removed, Tmat

def construct():
    """
    Construct a rank-minimal decomposition for 2x4 by 4x8 matrix multiplication.
    Starts with rank-56 Strassen decomposition and attempts rank reduction.
    """
    deadline = time.time() + 120.0
    U, V, W = _build_rank56()
    rank = U.shape[1]
    
    # Try to find redundant columns
    removed, Tmat = _find_redundant(U, V, W)
    
    if removed:
        keep = [i for i in range(rank) if i not in set(removed)]
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
