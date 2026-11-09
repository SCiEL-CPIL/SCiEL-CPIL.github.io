<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import least_squares

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    U_fb, V_fb, W_fb = schoolbook()
    best_rank = n * m * p
    best_U, best_V, best_W = U_fb, V_fb, W_fb
    
    # Build constraint matrix: for each (i,j,k), the equation is
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # This is a 3-linear equation.
    
    # Try to find lower rank decompositions using optimization
    # We'll parameterize U, V, W and minimize the residual
    
    def compute_residuals(params, r):
        """params is a flat vector of length r*(n*m + m*p + p*n)"""
        U = params[:r*n*m].reshape(n*m, r)
        V = params[r*n*m:r*(n*m+m*p)].reshape(m*p, r)
        W = params[r*(n*m+m*p):].reshape(p*n, r)
        residuals = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    residuals.append(val - 1.0)
        return residuals
    
    def try_rank(r, max_iter=2000, n_restarts=3):
        nonlocal best_rank, best_U, best_V, best_W
        if r >= best_rank:
            return False
        total_params = r * (n*m + m*p + p*n)
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            # Initialize with random values
            if restart == 0:
                # Start from scaled schoolbook-like init
                params = np.zeros(total_params)
                for idx in range(min(r, n*m*p)):
                    i = idx // (m*p)
                    j = (idx // p) % m
                    k = idx % p
                    if i < n and j < m and k < p and idx < r:
                        params[idx*n*m + i*m+j] = 1.0
                        params[idx*n*m + n*m + idx*m*p + j*p+k] = 1.0
                        params[idx*n*m + n*m + m*p + idx*p*n + k*n+i] = 1.0
            else:
                params = np.random.randn(total_params) * 0.5
            
            try:
                result = least_squares(
                    compute_residuals, params, args=(r,),
                    method='trf', max_nfev=max_iter, ftol=1e-12, xtol=1e-12, gtol=1e-12
                )
                if result.success or result.cost < 1e-10:
                    # Check if solution is valid
                    res = compute_residuals(result.x, r)
                    if np.max(np.abs(res)) < 1e-8:
                        U = result.x[:r*n*m].reshape(n*m, r)
                        V = result.x[r*n*m:r*(n*m+m*p)].reshape(m*p, r)
                        W = result.x[r*(n*m+m*p):].reshape(p*n, r)
                        # Round to nearest half-integer if close
                        U_r = np.round(U * 2) / 2
                        V_r = np.round(V * 2) / 2
                        W_r = np.round(W * 2) / 2
                        # Check if rounded version still works
                        res_r = compute_residuals(
                            np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()]), r
                        )
                        if np.max(np.abs(res_r)) < 1e-8:
                            U, V, W = U_r, V_r, W_r
                        best_rank = r
                        best_U, best_V, best_W = U, V, W
                        return True
            except Exception:
                continue
        return False
    
    # Try ranks from low to high
    # Lower bound: at least max(n*m, m*p, p*n) = 35
    # Try a range of ranks
    ranks_to_try = list(range(35, best_rank))
    random.shuffle(ranks_to_try)
    # Sort by rank to try lower first, but also try some random order
    ranks_to_try = sorted(set(ranks_to_try + list(range(50, 80))))
    
    found = False
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        if try_rank(r):
            found = True
            # If we found one, try to go lower
            if r > 35:
                # Try a few lower ranks
                for r2 in range(max(35, r-10), r):
                    if time.time() > deadline:
                        break
                    if try_rank(r2):
                        pass
    
    # Convert to list of lists for return
    def to_list(mat):
        return mat.tolist()
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

Wait, I realize the residual computation is O(n*m*p) per evaluation which is 105 operations, and with many iterations this could be slow. Let me optimize by vectorizing the residual computation.

Also, I should be more careful about the approach. Let me reconsider.

The key insight is that this is a 3-linear tensor decomposition problem. The residual for each (i,j,k) triple is:
Σ_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1

This can be vectorized. Let me rewrite more efficiently.

