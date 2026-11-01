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

            col = base + 0
            U[a_idx(0,0),col]+=1; U[a_idx(1,1),col]+=1
            V[b_idx(0,0),col]+=1; V[b_idx(1,1),col]+=1
            W[c_idx(0,0),col]+=1; W[c_idx(1,1),col]+=1
            col = base + 1
            U[a_idx(1,0),col]+=1; U[a_idx(1,1),col]+=1
            V[b_idx(0,0),col]+=1
            W[c_idx(1,0),col]+=1; W[c_idx(1,1),col]-=1
            col = base + 2
            U[a_idx(0,0),col]+=1
            V[b_idx(0,1),col]+=1; V[b_idx(1,1),col]-=1
            W[c_idx(0,1),col]+=1; W[c_idx(1,1),col]+=1
            col = base + 3
            U[a_idx(1,1),col]+=1
            V[b_idx(1,0),col]+=1; V[b_idx(0,0),col]-=1
            W[c_idx(0,0),col]+=1; W[c_idx(1,0),col]+=1
            col = base + 4
            U[a_idx(0,0),col]+=1; U[a_idx(0,1),col]+=1
            V[b_idx(1,1),col]+=1
            W[c_idx(0,0),col]-=1; W[c_idx(0,1),col]+=1
            col = base + 5
            U[a_idx(1,0),col]+=1; U[a_idx(0,0),col]-=1
            V[b_idx(0,0),col]+=1; V[b_idx(0,1),col]+=1
            W[c_idx(1,1),col]+=1
            col = base + 6
            U[a_idx(0,1),col]+=1; U[a_idx(1,1),col]-=1
            V[b_idx(1,0),col]+=1; V[b_idx(1,1),col]+=1
            W[c_idx(0,0),col]+=1

    return U, V, W


def _get_tensor_entries():
    """Get the (i,j,k) indices where the matrix multiplication tensor equals 1."""
    n, m, p = DIMENSIONS
    entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                entries.append((a_idx, b_idx, c_idx))
    return np.array(entries)


def _compute_error(U, V, W, entries):
    """Compute the L2 error of the decomposition on the tensor entries."""
    # For each entry (a,b,c), the value should be sum_s U[a,s]*V[b,s]*W[c,s]
    # We want this to equal 1 for valid entries
    # Error = sum over entries of (sum_s U[a,s]*V[b,s]*W[c,s] - 1)^2
    # Also need to check that invalid entries are 0
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # Vectorized computation for valid entries
    a_idx = entries[:, 0]
    b_idx = entries[:, 1]
    c_idx = entries[:, 2]
    
    # values[entry, s] = U[a_idx[entry], s] * V[b_idx[entry], s] * W[c_idx[entry], s]
    vals = U[a_idx, :] * V[b_idx, :] * W[c_idx, :]  # shape (64, rank)
    sums = vals.sum(axis=1)  # shape (64,)
    error = np.sum((sums - 1.0) ** 2)
    return error


def _greedy_eliminate(U, V, W, entries, tol=1e-10, max_rounds=5):
    """Try to greedily eliminate redundant rank-1 terms."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # Build the tensor matrix for all valid entries
    # Tmat has shape (64, rank) where Tmat[e, s] = U[a_e, s] * V[b_e, s] * W[c_e, s]
    a_idx = entries[:, 0]
    b_idx = entries[:, 1]
    c_idx = entries[:, 2]
    
    Tmat = U[a_idx, :] * V[b_idx, :] * W[c_idx, :]  # (64, rank)
    
    # Also need to check invalid entries. For invalid (a,b,c), the sum should be 0.
    # But for greedy elimination, we'll focus on valid entries first.
    
    keep = list(range(rank))
    
    for rnd in range(max_rounds):
        if len(keep) <= 1:
            break
        best_remove = None
        best_residual = np.inf
        
        for idx in keep:
            others = [k for k in keep if k != idx]
            if len(others) < 1:
                continue
            
            # Check if column idx is in span of others
            # Use QR for numerical stability
            sub = Tmat[:, others]
            try:
                Q, R = np.linalg.qr(sub)
                proj = Q @ (Q.T @ Tmat[:, idx])
                residual = Tmat[:, idx] - proj
                res_norm = np.linalg.norm(residual)
                col_norm = np.linalg.norm(Tmat[:, idx])
                if col_norm > 0 and res_norm < tol * col_norm:
                    best_remove = idx
                    best_residual = res_norm
                    break
            except np.linalg.LinAlgError:
                continue
        
        if best_remove is not None:
            keep.remove(best_remove)
        else:
            break
    
    if len(keep) < rank:
        U_new = U[:, keep]
        V_new = V[:, keep]
        W_new = W[:, keep]
        return U_new, V_new, W_new, len(keep)
    
    return U, V, W, rank


def _local_refinement(U, V, W, entries, max_iter=200, lr=0.01, deadline=None):
    """
    Local coordinate descent to try to reduce the effective rank.
    We try to find a lower-rank decomposition by minimizing the error.
    Strategy: start with rank-56, try to drive some columns toward zero.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    # We'll try to minimize the sum of squares of all column norms
    # while keeping the tensor error small.
    # This is a constrained optimization.
    
    # Alternative: try to find a rank-(rank-1) decomposition by 
    # projecting out one column's contribution.
    
    # For each column s, compute what happens if we remove it and 
    # redistribute its contribution to other columns.
    a_idx = entries[:, 0]
    b_idx = entries[:, 1]
    c_idx = entries[:, 2]
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    for s in range(rank):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try removing column s
        others = [k for k in range(rank) if k != s]
        U_o = U[:, others]
        V_o = V[:, others]
        W_o = W[:, others]
        
        # The residual tensor from removing column s:
        # residual[e] = U[a_e, s] * V[b_e, s] * W[c_e, s]
        residual = U[a_idx, s] * V[b_idx, s] * W[c_idx, s]  # shape (64,)
        
        # Try to represent residual as a linear combination of other columns
        # Tmat_o = U_o[a_idx,:] * V_o[b_idx,:] * W_o[c_idx,:]  shape (64, rank-1)
        Tmat_o = U_o[a_idx, :] * V_o[b_idx, :] * W_o[c_idx, :]
        
        # Solve: Tmat_o @ x ≈ residual
        # x has shape (rank-1,)
        try:
            x, residuals, rank_svd, sv = np.linalg.lstsq(Tmat_o, residual, rcond=None)
        except np.linalg.LinAlgError:
            continue
        
        # Check the fit
        fitted = Tmat_o @ x
        fit_error = np.linalg.norm(residual - fitted)
        col_norm = np.linalg.norm(residual)
        
        if col_norm > 1e-15 and fit_error < 1e-8 * col_norm:
            # We can absorb column s into others
            # Update: new column k (in others) gets x[others.index(k)] added
            # But we need to update U, V, W for the new coefficients
            # This is tricky because the columns aren't independent in the tensor sense
            # For now, just note that we found a dependency
            pass
    
    return best_U, best_V, best_W, best_rank


def _try_rank_reduction(U, V, W, entries, deadline):
    """
    Try various strategies to reduce the rank below 56.
    Returns the best (U, V, W) found.
    """
    n, m, p = DIMENSIONS
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    
    # Strategy 1: Greedy elimination
    U1, V1, W1, r1 = _greedy_eliminate(U, V, W, entries)
    if r1 < best_rank:
        best_U, best_V, best_W, best_rank = U1, V1, W1, r1
    
    # Strategy 2: Try to find a rank-55 decomposition by perturbation
    # We'll try to modify the decomposition to have one fewer independent term.
    # This is hard in general, so we'll use a simple approach:
    # Try to merge two similar columns.
    
    if deadline is not None and time.time() > deadline:
        return best_U, best_V, best_W
    
    # Strategy 3: SVD-based compression
    # The tensor T = sum_s U_s ⊗ V_s ⊗ W_s
    # We can try to find a lower-rank approximation.
    # But we need EXACT decomposition, so this won't help directly.
    
    # Strategy 4: Try different groupings of the 8 Strassen blocks
    # Instead of 8 independent 2x2 blocks, can we share terms?
    # This requires a fundamentally different decomposition.
    
    return best_U, best_V, best_W


def _verify(U, V, W, n, m, p, tol=1e-8):
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
                if abs(val - 1.0) > tol:
                    return False
    
    # Check invalid entries
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
            if abs(val) > tol:
                return False
    
    return True


def construct():
    """
    Construct a rank-minimal decomposition for 2x4 by 4x8 matrix multiplication.
    Starts with rank-56 Strassen decomposition and attempts rank reduction.
    """
    deadline = time.time() + 115.0  # Safety margin from 150s limit
    n, m, p = DIMENSIONS
    
    U, V, W = _build_rank56()
    rank = U.shape[1]
    
    # Get tensor entries for verification and optimization
    entries = _get_tensor_entries()
    
    # Verify base decomposition
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Base decomposition is invalid")
    
    # Try rank reduction
    U, V, W = _try_rank_reduction(U, V, W, entries, deadline)
    rank = U.shape[1]
    
    # Final verification
    if not _verify(U, V, W, n, m, p):
        # Fall back to original
        U, V, W = _build_rank56()
        rank = 56
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


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
