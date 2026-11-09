"""Search for a low-rank exact tensor decomposition of 4x5x5 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [4, 5, 5]
_START_TIME = time.time()
_TIMEOUT = 120.0
SAFETY_MARGIN = 10.0  # seconds

def _time_remaining():
    return _TIMEOUT - SAFETY_MARGIN - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # W index: k*n + i
    # U index: i*m + j
    # V index: j*p + k
    
    # Build target tensor T[i, j, k] = 1 if we are multiplying (i,j) by (j,k) -> contributes to (i,k)
    # The tensor T is of shape (n, m, p) where T[i,j,k] = 1 for all i,j,k
    # Wait, the tensor for matrix multiplication is T_{i,j,k} such that C_{ik} = sum_{j} A_{ij} B_{jk}.
    # In tensor decomposition form: T = sum_t U_t \otimes V_t \otimes W_t
    # where (U_t)_{ij} U_{ij}, (V_t)_{jk} V_{jk}, (W_t)_{ki} W_{ki}.
    # The target tensor T_{ijk} is 1 if i, j, k are valid indices.
    # Specifically, T is a 3D array of ones of shape (n, m, p).
    # Let's verify the index convention.
    # U is (n*m, rank). U[i*m+j, t].
    # V is (m*p, rank). V[j*p+k, t].
    # W is (p*n, rank). W[k*n+i, t].
    # The product sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] should equal 1 for all i,j,k.
    
    target = np.ones((n, m, p))
    
    # Try a few ranks. Schoolbook is 100.
    # Known lower bounds:
    # For 4x5x5, rank >= ?
    # Strassen for 2x2x2 is 7.
    # For 4x4x4, rank 49 (Stothers).
    # For 5x5x5, rank 49 (Stothers) or 35? Actually 5x5x5 is unknown but > 35?
    # Let's try ranks: 40, 45, 50.
    
    # We will attempt to find a decomposition for a specific rank.
    # To make it tractable, we use a heuristic optimization.
    # Variables: U, V, W flattened.
    # Objective: || sum_t u_t v_t^T w_t - T ||_F^2
    
    best_solution = None
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize schoolbook solution as fallback
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    
    def objective(params):
        if _time_remaining() < 0:
            return np.inf
        # Reshape params into U, V, W
        # Sizes: n*m*rank, m*p*rank, p*n*rank
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:size_U+size_V+size_W]
        
        U = U_flat.reshape((n*m, rank))
        V = V_flat.reshape((m*p, rank))
        W = W_flat.reshape((p*n, rank))
        
        # Compute tensor sum
        # T[ i, j, k ] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        # This is a contraction.
        # We can do this efficiently:
        # For each t, outer product of U[:,t], V[:,t], W[:,t]
        # Summing over t is like a batched matrix multiplication or einsum.
        
        # Let's use einsum.
        # U shape: (n, m, rank) -> reshape from (n*m, rank)
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        # Contract over rank
        # Result shape: (n, m, p)
        # T[i,j,k] = sum_t U_reshaped[i,j,t] * V_reshaped[j,k,t] * W_reshaped[k,i,t]
        
        # Note: W index is k*n+i, so W_reshaped[k,i,t] corresponds to W_flat[k*n+i, t].
        # This matches the definition.
        
        try:
            T_calc = np.einsum('ijt,jkt,kit->ijk', U_reshaped, V_reshaped, W_reshaped)
        except Exception:
            return np.inf
            
        diff = T_calc - target
        return np.sum(diff * diff)
    
    # We try ranks in increasing order of difficulty? Or just a few promising ones.
    # Since 4x5x5 is small, rank 40-50 is plausible.
    # Let's try rank 40, 45, 50.
    
    ranks_to_try = [40, 45, 50]
    
    for rank in ranks_to_try:
        if _time_remaining() < 5:
            break
            
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        total_size = size_U + size_V + size_W
        
        # Random initialization
        # Scale down to avoid large gradients initially
        np.random.seed(42)
        x0 = np.random.randn(total_size) * 0.1
        
        # Use L-BFGS-B
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 1000})
            
            if res.fun < 1e-8:
                # Found a low error solution, try to snap
                U_flat = res.x[:size_U]
                V_flat = res.x[size_U:size_U+size_V]
                W_flat = res.x[size_U+size_V:size_U+size_V+size_W]
                
                U = U_flat.reshape((n*m, rank))
                V = V_flat.reshape((m*p, rank))
                W = W_flat.reshape((p*n, rank))
                
                # Snap to nearest integer or half-integer
                # Multiply by 2, round, divide by 2
                U_snapped = np.round(U * 2) / 2
                V_snapped = np.round(V * 2) / 2
                W_snapped = np.round(W * 2) / 2
                
                # Verify exactness
                U_reshaped = U_snapped.reshape(n, m, rank)
                V_reshaped = V_snapped.reshape(m, p, rank)
                W_reshaped = W_snapped.reshape(p, n, rank)
                
                T_calc = np.einsum('ijt,jkt,kit->ijk', U_reshaped, V_reshaped, W_reshaped)
                
                if np.allclose(T_calc, target, atol=1e-10):
                    best_solution = res.x
                    best_rank = rank
                    best_U = U_snapped
                    best_V = V_snapped
                    best_W = W_snapped
                    break
        except Exception as e:
            continue
    
    if best_solution is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        # Fallback to schoolbook
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
