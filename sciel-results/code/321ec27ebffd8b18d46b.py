"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the target tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    deadline = time.time() + 115.0
    
    # Try different ranks, starting from the theoretical lower bound
    for rank in [25, 26, 27, 28, 29, 30, 32, 35, 40, 50, 60, 80, 100]:
        if time.time() > deadline - 20:
            break
        result = _try_rank_als(T, nm, mp, pn, rank, deadline)
        if result is not None:
            return result
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_als(T, nm, mp, pn, rank, deadline):
    """
    Attempt to find a rank-rank exact decomposition using:
    1. Random restarts + ALS with sparsity thresholding
    2. Structured ansatz based on diagonal/block decomposition
    """
    best_result = None
    np.random.seed(42)
    
    # Strategy 1: Random ALS with multiple restarts
    max_restarts = 30
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        result = _als_optimize(T, U, V, W, nm, mp, pn, rank, deadline, max_iters=200)
        if result is not None:
            return result
        if best_result is None:
            best_result = result
    
    # Strategy 2: Structured diagonal ansatz
    # For matrix multiplication, there are known rank decompositions using
    # diagonal matrices. Try structured forms where each rank-1 term
    # corresponds to a specific (i,j,k) interaction pattern.
    result = _try_structured(T, nm, mp, pn, rank, deadline)
    if result is not None:
        return result
    
    # Strategy 3: Integer/half-integer grid search for small ranks
    if rank <= 30:
        result = _try_half_integer_grid(T, nm, mp, pn, rank, deadline)
        if result is not None:
            return result
    
    return best_result

def _als_optimize(T, U, V, W, nm, mp, pn, rank, deadline, max_iters=200):
    """Alternating least squares with sparsity and half-integer rounding checks."""
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        # Precompute Gram matrices (recomputed each outer iteration for stability)
        VT_V = V.T @ V
        WT_W = W.T @ W
        G = VT_V * WT_W  # Hadamard product (rank, rank)
        reg = 1e-8 * np.eye(rank)
        
        # Update U
        for a in range(nm):
            TW = T[a, :, :] @ W  # (mp, rank)
            h = V.T @ TW  # (rank,)
            try:
                U[a, :] = np.linalg.solve(G + reg, h)
            except np.linalg.LinAlgError:
                U[a, :] = np.linalg.lstsq(G + reg, h, rcond=None)[0]
        
        # Update V
        VT_V = V.T @ V
        WT_W = W.T @ W
        G = VT_V * WT_W
        for b in range(mp):
            UT_T = U.T @ T[:, b, :]  # (rank, pn)
            h = UT_T @ W  # (rank,)
            try:
                V[b, :] = np.linalg.solve(G + reg, h)
            except np.linalg.LinAlgError:
                V[b, :] = np.linalg.lstsq(G + reg, h, rcond=None)[0]
        
        # Update W
        VT_V = V.T @ V
        WT_W = W.T @ W
        G = VT_V * WT_W
        for d in range(pn):
            UT_T = U.T @ T[:, :, d]  # (rank, mp)
            h = UT_T @ V  # (rank,)
            try:
                W[d, :] = np.linalg.solve(G + reg, h)
            except np.linalg.LinAlgError:
                W[d, :] = np.linalg.lstsq(G + reg, h, rcond=None)[0]
        
        # Sparsity thresholding
        thresh = 0.05
        U[np.abs(U) < thresh] = 0.0
        V[np.abs(V) < thresh] = 0.0
        W[np.abs(W) < thresh] = 0.0
        
        # Check error
        T_hat = np.einsum('ac,bc,dc->abd', U, V, W)
        err = np.max(np.abs(T - T_hat))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-9:
            return _check_and_format(U, V, W, T)
        
        # Check half-integer rounding
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        T_hat_r = np.einsum('ac,bc,dc->abd', U_r, V_r, W_r)
        err_r = np.max(np.abs(T - T_hat_r))
        if err_r < 1e-9:
            return _format(U_r, V_r, W_r)
        
        if best_err < 1e-6:
            # Close enough, try to snap
            U_r = np.round(U * 2) / 2.0
            V_r = np.round(V * 2) / 2.0
            W_r = np.round(W * 2) / 2.0
            T_hat_r = np.einsum('ac,bc,dc->abd', U_r, V_r, W_r)
            if np.max(np.abs(T - T_hat_r)) < 1e-9:
                return _format(U_r, V_r, W_r)
    
    if best_err < 1e-9:
        return _check_and_format(best_U, best_V, best_W, T)
    if best_err < 1e-6:
        U_r = np.round(best_U * 2) / 2.0
        V_r = np.round(best_V * 2) / 2.0
        W_r = np.round(best_W * 2) / 2.0
        T_hat_r = np.einsum('ac,bc,dc->abd', U_r, V_r, W_r)
        if np.max(np.abs(T - T_hat_r)) < 1e-9:
            return _format(U_r, V_r, W_r)
    
    return None

def _check_and_format(U, V, W, T):
    """Check exactness and format result."""
    T_hat = np.einsum('ac,bc,dc->abd', U, V, W)
    if np.max(np.abs(T - T_hat)) < 1e-9:
        return _format(U, V, W)
    # Try half-integer
    U_r = np.round(U * 2) / 2.0
    V_r = np.round(V * 2) / 2.0
    W_r = np.round(W * 2) / 2.0
    T_hat_r = np.einsum('ac,bc,dc->abd', U_r, V_r, W_r)
    if np.max(np.abs(T - T_hat_r)) < 1e-9:
        return _format(U_r, V_r, W_r)
    return None

def _format(U, V, W):
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _try_structured(T, nm, mp, pn, rank, deadline):
    """
    Try structured decompositions. For matrix multiplication C = AB where
    A is n×m and B is m×p, the tensor has a specific structure.
    
    One approach: use rank-1 terms corresponding to each (i,j,k) triple,
    but combine them cleverly.
    
    For 4×5×5, try the "diagonal" decomposition where we use rank-1 terms
    based on diagonal matrices of various forms.
    """
    n, m, p = DIMENSIONS
    
    # Known: the trivial rank is n*m*p = 100
    # Try to find a lower rank using structured patterns
    
    # Approach: use the fact that for each pair (i,k), the j-index
    # creates a rank-1 contribution. Group terms by (i,k) pairs.
    # There are n*p = 20 such pairs. For each pair, the j-range is 1..m=5.
    # So we can have at least 20 rank-1 terms if we can share j-indices.
    
    # Try: for each (i, k), create one rank-1 term that covers all j
    # This won't work directly because different j's need different weights.
    
    # Alternative: use the "Kronecker" style decomposition
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # = 1 if a % m == b // p and b % p == c // n and c % n == a // m
    
    # Let's try a decomposition where we enumerate over possible "paths"
    # For each rank-1 term, we need vectors u (nm,), v (mp,), w (pn,)
    # such that u[a]*v[b]*w[c] matches the tensor pattern.
    
    # For a rank-1 term to be non-zero, it must satisfy:
    # u[a]*v[b]*w[c] = 1 when (a,b,c) is a valid triple, 0 otherwise.
    # This means u, v, w must be "consistent" with the index constraints.
    
    # Let's try: for each permutation-like structure, define:
    # u[a] = 1 if a corresponds to (i,j) for some subset, else 0
    # v[b] = 1 if b corresponds to (j,k) for the same j
    # w[c] = 1 if c corresponds to (k,i) for the same i
    
    # This is essentially the schoolbook decomposition with 100 terms.
    # To get lower rank, we need to combine terms.
    
    # Try a randomized structured approach:
    # Initialize with random but structured patterns
    np.random.seed(123)
    
    for attempt in range(5):
        if time.time() > deadline - 10:
            break
        
        # Create structured initial guess
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # Assign each (i,j,k) triple to a rank component
        triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        
        if len(triples) <= rank:
            # Direct assignment
            for r, (i, j, k) in enumerate(triples):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, r] = 1.0
                V[b, r] = 1.0
                W[c, r] = 1.0
            # Check if this is exact (it should be if rank >= n*m*p)
            T_hat = np.einsum('ac,bc,dc->abd', U, V, W)
            if np.max(np.abs(T - T_hat)) < 1e-9:
                return _format(U, V, W)
        else:
            # Need to combine: assign multiple triples to same rank component
            # Use random assignment
            assignment = np.random.randint(0, rank, size=len(triples))
            for r_idx, (i, j, k) in enumerate(triples):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                r = assignment[r_idx]
                # Accumulate: if multiple triples map to same r,
                # they must be compatible (same a or b or c patterns)
                # This is tricky; just set to 1 and see
                U[a, r] += 1.0
                V[b, r] += 1.0
                W[c, r] += 1.0
            
            # This won't generally be exact, but try ALS to fix it
            result = _als_optimize(T, U, V, W, nm, mp, pn, rank, deadline, max_iters=100)
            if result is not None:
                return result
    
    return None

def _try_half_integer_grid(T, nm, mp, pn, rank, deadline):
    """
    For small ranks, try a restricted search over half-integer values.
    This is only feasible for very small ranks due to combinatorial explosion.
    We use a heuristic: try to find sparse solutions first.
    """
    # This is a placeholder - full grid search is infeasible.
    # Instead, do a few random half-integer perturbations of zero matrix.
    np.random.seed(999)
    
    for attempt in range(50):
        if time.time() > deadline - 5:
            break
        
        # Random sparse half-integer initialization
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # Place random non-zeros
        for r in range(rank):
            # Choose a few non-zero entries
            nz_u = np.random.choice(nm, size=min(3, nm), replace=False)
            nz_v = np.random.choice(mp, size=min(3, mp), replace=False)
            nz_w = np.random.choice(pn, size=min(3, pn), replace=False)
            
            for idx in nz_u:
                U[idx, r] = np.random.choice([-1, -0.5, 0.5, 1])
            for idx in nz_v:
                V[idx, r] = np.random.choice([-1, -0.5, 0.5, 1])
            for idx in nz_w:
                W[idx, r] = np.random.choice([-1, -0.5, 0.5, 1])
        
        # Check if exact
        T_hat = np.einsum('ac,bc,dc->abd', U, V, W)
        if np.max(np.abs(T - T_hat)) < 1e-9:
            return _format(U, V, W)
        
        # Try to refine with a few ALS steps
        result = _als_optimize(T, U, V, W, nm, mp, pn, rank, deadline, max_iters=50)
        if result is not None:
            return result
    
    return None
# EVOLVE-BLOCK-END
