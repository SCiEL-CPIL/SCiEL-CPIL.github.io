"""Tensor rank minimization for 2x3x5 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def _validate(U, V, W, tol=1e-6):
    """Check that U @ diag @ V^T reconstructs W appropriately."""
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # The tensor: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # T should equal 1 when (a,b,c) form a valid (i,j),(j,k),(k,i) triple, else 0.
    # Build target tensor
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                target[a, b, c] = 1.0
    # Reconstruct
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    err = np.max(np.abs(recon - target))
    return err < tol, err

def _try_rank(rank, seed, time_budget):
    """Try to find a rank-`rank` decomposition via random init + ALS."""
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    U = rng.randn(n*m, rank)
    V = rng.randn(m*p, rank)
    W = rng.randn(p*n, rank)
    
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                target[a, b, c] = 1.0
    
    start = time.time()
    for it in range(2000):
        if time.time() - start > time_budget:
            break
        # ALS: update each factor
        # U_new[a,r] = sum_{b,c} target[a,b,c] V[b,r] W[c,r] / (sum V[b,r]^2 W[c,r]^2)
        # Simpler: use standard ALS
        # U = target @ (V^T ⊙ W^T) ... use einsum
        VW = np.einsum('br,cr->bcr', V, W)  # (m*p, p*n, rank)
        # U[a,r] = sum_{b,c} target[a,b,c] * V[b,r]*W[c,r]
        U_new = np.einsum('abc,br,cr->ar', target, V, W)
        denom = np.einsum('br,cr->r', V**2, W**2)
        denom = np.where(denom < 1e-12, 1.0, denom)
        U_new /= denom[None, :]
        
        UW = np.einsum('ar,cr->acr', U, W)
        V_new = np.einsum('abc,ar,cr->br', target, U, W)
        denom = np.einsum('ar,cr->r', U**2, W**2)
        denom = np.where(denom < 1e-12, 1.0, denom)
        V_new /= denom[None, :]
        
        UV = np.einsum('ar,br->abr', U, V)
        W_new = np.einsum('abc,ar,br->cr', target, U, V)
        denom = np.einsum('ar,br->r', U**2, V**2)
        denom = np.where(denom < 1e-12, 1.0, denom)
        W_new /= denom[None, :]
        
        U, V, W = U_new, V_new, W_new
    
    err, _ = _validate(U, V, W)
    return err, (U, V, W)

def _round_to_half(X):
    """Round to nearest half-integer."""
    return np.round(2*X)/2.0

def _try_round_and_verify(U, V, W):
    """Try rounding each factor to half-integers and check."""
    best_err = float('inf')
    best = None
    for _ in range(50):
        Ur = _round_to_half(U + np.random.randn(*U.shape)*0.1)
        Vr = _round_to_half(V + np.random.randn(*V.shape)*0.1)
        Wr = _round_to_half(W + np.random.randn(*W.shape)*0.1)
        err, _ = _validate(Ur, Vr, Wr)
        if err < best_err:
            best_err = err
            best = (Ur, Vr, Wr)
    return best_err, best

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    # Try increasing ranks
    for rank in range(1, n*m*p+1):
        if time.time() > deadline:
            break
        # Multiple random restarts per rank
        max_restarts = 20
        for seed in range(max_restarts):
            if time.time() > deadline:
                break
            budget = min(2.0, max(0.5, (deadline - time.time())/10))
            err, (U, V, W) = _try_rank(rank, seed, budget)
            if err < 1e-6:
                # Found! Try to round to half-integers
                _, rounded = _try_round_and_verify(U, V, W)
                if rounded is not None:
                    Ur, Vr, Wr = rounded
                    # Convert to lists
                    U_list = Ur.tolist()
                    V_list = Vr.tolist()
                    W_list = Wr.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
                # If rounding fails, return as-is (may not be half-integer but valid)
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
