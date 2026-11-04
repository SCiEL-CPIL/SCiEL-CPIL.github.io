"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _snap_to_half_int(x):
    """Snap x to the nearest half-integer."""
    return round(x * 2) / 2.0

def _try_optimization(n, m, p, rank, time_limit=10.0):
    """
    Attempt to find a decomposition of given rank using continuous optimization
    followed by snapping to half-integers.
    """
    from scipy.optimize import minimize
    
    # Flatten dimensions
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Total number of variables
    num_vars = rank * (dim_u + dim_v + dim_w)
    
    # Objective function
    def objective(x):
        # Reshape x into U, V, W
        # x is a flat vector
        U_flat = x[:rank * dim_u]
        V_flat = x[rank * dim_u : rank * dim_u + rank * dim_v]
        W_flat = x[rank * dim_u + rank * dim_v :]
        
        # Reshape to matrices (rank, dim) for easier vectorized ops?
        # Or keep as (dim, rank) to match access pattern?
        # Access: U[i*m+j][c] -> U_flat[c*dim_u + i*m+j] if column-major?
        # Let's use row-major for simplicity: U_flat[(i*m+j)*rank + c]
        
        # We want to compute sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # This is a tensor contraction.
        # U is (dim_u, rank), V is (dim_v, rank), W is (dim_w, rank)
        
        U = U_flat.reshape(dim_u, rank)
        V = V_flat.reshape(dim_v, rank)
        W = W_flat.reshape(dim_w, rank)
        
        # We need to check all i, j, k
        # Error = sum_{i,j,k} (sum_c U[i,j,c] V[j,k,c] W[k,i,c] - 1)^2
        
        # Vectorized computation:
        # This is hard to fully vectorize without einsum or similar.
        # Let's use einsum for speed if available (numpy has it)
        
        # T_ijk = sum_c U_ijc V_jkc W_kic
        # np.einsum('ic,jkc,kic->ijk', U, V, W)
        
        # Check if numpy version supports einsum with these indices
        try:
            T = np.einsum('ic,jkc,kic->ijk', U, V, W, optimize=True)
            error = np.sum((T - 1.0) ** 2)
            return error
        except Exception:
            # Fallback to loops if einsum fails (unlikely)
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        err += (s - 1.0) ** 2
            return err

    # Initialize with small random values
    np.random.seed(42)
    x0 = np.random.randn(num_vars) * 0.1
    
    # Run optimization
    start_time = time.time()
    
    # We use L-BFGS-B as it handles large variable counts well
    # maxiter is a safeguard, but we also check time
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
    
    # Check time
    if time.time() - start_time > time_limit:
        return None
        
    # Extract solution
    x_sol = res.x
    U_flat = x_sol[:rank * dim_u]
    V_flat = x_sol[rank * dim_u : rank * dim_u + rank * dim_v]
    W_flat = x_sol[rank * dim_u + rank * dim_v :]
    
    U = U_flat.reshape(dim_u, rank)
    V = V_flat.reshape(dim_v, rank)
    W = W_flat.reshape(dim_w, rank)
    
    # Snap to half-integers
    U_snapped = np.round(U * 2) / 2.0
    V_snapped = np.round(V * 2) / 2.0
    W_snapped = np.round(W * 2) / 2.0
    
    # Convert to list of lists for verification and return
    U_list = U_snapped.tolist()
    V_list = V_snapped.tolist()
    W_list = W_snapped.tolist()
    
    # Verify
    if _verify(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # If snapping failed, try local search around the snapped values?
    # For now, just return None to try next rank or fallback
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin for 120s limit
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 90
    
    # Try ranks from low to high
    # Lower bound estimate: maybe 40?
    # We'll try a range of ranks.
    # Optimization is expensive, so we can't try too many.
    ranks_to_try = [50, 60, 70, 80, 85, 89]
    
    for rank in ranks_to_try:
        if rank >= best_rank:
            break
        if time.time() > deadline - 20:
            break
            
        result = _try_optimization(n, m, p, rank, time_limit=15.0)
        
        if result is not None:
            best = result
            best_rank = rank
            # We found a better rank, but maybe we can find an even lower one?
            # The loop goes upwards, so if we found 60, we might have tried 50 and failed.
            # We should continue to see if we missed a lower one? No, we go up.
            # So if we found 60, we stop? No, we want the MINIMAL rank.
            # So we should iterate from low to high and stop at the first success?
            # Yes, because we want the minimal rank.
            break
            
    # If we didn't find anything better, return schoolbook
    return best
# EVOLVE-BLOCK-END
