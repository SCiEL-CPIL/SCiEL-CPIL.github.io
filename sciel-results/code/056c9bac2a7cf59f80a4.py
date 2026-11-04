"""
Find a rank-minimal exact tensor decomposition for multiplying a 3 by 3 matrix 
by a 3 by 6 matrix.

Strategy:
1. Start with the schoolbook decomposition (rank 54).
2. Exploit the shared middle dimension m=3 by treating the 3x6 matrix as
   two 3x3 blocks [B1 | B2] and constructing a joint decomposition where
   the V factor is shared across both output blocks.
3. Use alternating least squares (ALS) with random restarts to find a
   lower-rank decomposition that satisfies the tensor identity.
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 6]
DEADLINE = 140  # seconds, safety margin before 150s kill

def _verify(U, V, W):
    """
    Verify that the decomposition is correct.
    U: (n*m, rank) - indexed by (i*m + j, r)
    V: (m*p, rank) - indexed by (j*p + k, r)
    W: (p*n, rank) - indexed by (k*n + i, r)
    
    The decomposition should satisfy:
    For all i, j, k:
    1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i * m + j, :]
                v_val = V_arr[j * p + k, :]
                w_val = W_arr[k * n + i, :]
                tensor_val = np.sum(u_val * v_val * w_val)
                if abs(tensor_val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook():
    """Return the schoolbook decomposition with rank n*m*p."""
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


def _als_decompose(n, m, p, target_rank, max_iters=200, tol=1e-10, seed=0):
    """
    Alternating Least Squares to find a rank-`target_rank` decomposition
    of the matrix multiplication tensor for (n, m, p).
    
    Returns (U, V, W) as numpy arrays, or None if not converged.
    """
    rng = np.random.RandomState(seed)
    
    # Initialize
    U = rng.randn(n * m, target_rank) * 0.5
    V = rng.randn(m * p, target_rank) * 0.5
    W = rng.randn(p * n, target_rank) * 0.5
    
    for it in range(max_iters):
        # Update U: for each (i,j), solve linear system
        for i in range(n):
            for j in range(m):
                # Need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                # For fixed (i,j), define M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                # Then we need U[i*m+j, :] @ M.T = 1 (vector of ones of length p)
                M = np.zeros((p, target_rank))
                for k in range(p):
                    M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                # Solve M.T @ u = 1
                try:
                    u_new, residuals, rank_s, sv = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                except np.linalg.LinAlgError:
                    continue
                U[i * m + j, :] = u_new
        
        # Update V: for each (j,k), solve linear system
        for j in range(m):
            for k in range(p):
                # Need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
                # For fixed (j,k), define M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                # Then we need V[j*p+k, :] @ M.T = 1
                M = np.zeros((n, target_rank))
                for i in range(n):
                    M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    v_new, residuals, rank_s, sv = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                except np.linalg.LinAlgError:
                    continue
                V[j * p + k, :] = v_new
        
        # Update W: for each (k,i), solve linear system
        for k in range(p):
            for i in range(n):
                # Need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
                # For fixed (k,i), define M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                # Then we need W[k*n+i, :] @ M.T = 1
                M = np.zeros((m, target_rank))
                for j in range(m):
                    M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    w_new, residuals, rank_s, sv = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                except np.linalg.LinAlgError:
                    continue
                W[k * n + i, :] = w_new
        
        # Check convergence
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor_val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    max_err = max(max_err, abs(tensor_val - 1.0))
        
        if max_err < tol:
            return U, V, W
    
    # Final check
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor_val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                max_err = max(max_err, abs(tensor_val - 1.0))
    
    if max_err < tol:
        return U, V, W
    
    return None

def construct():
    start_time = time.time()
    
    # Start with schoolbook as fallback
    best = _schoolbook()
    best_rank = 54
    
    n, m, p = DIMENSIONS
    
    # Try to find a lower rank decomposition using ALS
    # The multiplication tensor for 3x3 * 3x6 has rank at most 54 (schoolbook).
    # We try ranks from 53 down, using multiple random restarts.
    
    # Try ranks from 53 down to 40 (reasonable lower bound)
    for target_rank in range(53, 39, -1):
        if time.time() - start_time > DEADLINE - 30:
            break
        
        # Multiple random restarts for this rank
        max_attempts = 10
        found = False
        for attempt in range(max_attempts):
            if time.time() - start_time > DEADLINE - 30:
                break
            
            seed = attempt * 10000 + target_rank * 7
            result = _als_decompose(n, m, p, target_rank, max_iters=100, tol=1e-10, seed=seed)
            
            if result is not None:
                U, V, W = result
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                
                if _verify(U_list, V_list, W_list):
                    best = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            # Found a valid decomposition at this rank, try lower ranks
            continue
        else:
            # If we couldn't find one at this rank, try a few more with different seeds
            # but don't spend too much time
            extra_attempts = 5
            for attempt in range(extra_attempts):
                if time.time() - start_time > DEADLINE - 30:
                    break
                seed = attempt * 20000 + target_rank * 13 + 999
                result = _als_decompose(n, m, p, target_rank, max_iters=150, tol=1e-10, seed=seed)
                
                if result is not None:
                    U, V, W = result
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    
                    if _verify(U_list, V_list, W_list):
                        best = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = target_rank
                        found = True
                        break
            
            if not found:
                # If we can't find a decomposition at this rank, it may be that
                # the rank is not achievable, so try the next lower rank
                # But also check: if we just failed at rank R, try R-1, R-2, etc.
                # Continue the loop to try lower ranks
                pass
    
    return best
