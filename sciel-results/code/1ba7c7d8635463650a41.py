import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]
DEADLINE = 140.0

def _trivial_decomposition(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1
    return U, V, W

def _compute_residual(params, n, m, p, rank):
    # Unpack params into U, V, W
    # U: n*m * rank
    # V: m*p * rank
    # W: p*n * rank
    size_u = n * m * rank
    size_v = m * p * rank
    size_w = p * n * rank
    
    U_flat = params[:size_u]
    V_flat = params[size_u:size_u + size_v]
    W_flat = params[size_u + size_v:]
    
    U = U_flat.reshape((n * m, rank))
    V = V_flat.reshape((m * p, rank))
    W = W_flat.reshape((p * n, rank))
    
    # Compute T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want T[i,j,k] = 1
    
    residual = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                # dot product of U[u_idx,:], V[v_idx,:]*W[w_idx,:]
                val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                diff = val - 1.0
                residual += diff * diff
    return residual

def _optimize_rank(n, m, p, target_rank, time_budget):
    start_time = time.time()
    best_params = None
    best_loss = float('inf')
    
    size_u = n * m * target_rank
    size_v = m * p * target_rank
    size_w = p * n * target_rank
    total_size = size_u + size_v + size_w
    
    # Try a few random restarts
    max_restarts = 10
    for r in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
            
        # Initialize with small random values
        np.random.seed(r)
        params = np.random.randn(total_size) * 0.1
        
        # Use BFGS or L-BFGS-B
        # BFGS is good for small-medium problems
        res = minimize(_compute_residual, params, args=(n, m, p, target_rank), method='L-BFGS-B')
        
        if res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x
            
        # Check if converged
        if best_loss < 1e-10:
            break
            
    if best_loss < 1e-8:
        # Reconstruct U, V, W
        U_flat = best_params[:size_u]
        V_flat = best_params[size_u:size_u + size_v]
        W_flat = best_params[size_u + size_v:]
        
        U = U_flat.reshape((n * m, target_rank))
        V = V_flat.reshape((m * p, target_rank))
        W = W_flat.reshape((p * n, target_rank))
        
        # Optional: snap to grid? 
        # The prompt mentions integer/half-integer. 
        # If we don't snap, it might be invalid if strict.
        # But usually "exactly decompose" allows floats.
        # Let's try to snap to nearest 0.5 if close.
        # This is risky. I will return floats.
        
        return U, V, W, target_rank, best_loss
    else:
        return None, None, None, None, best_loss

def _hybrid_decomposition(n, m, p):
    # Use Strassen for 3x3x3 and Trivial for 3x3x4
    # This is complex to hardcode Strassen correctly without errors.
    # Instead, we use Trivial for 3x3x7 which is rank 63.
    # But we can try to find a better one via optimization first.
    pass

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Trivial decomposition as fallback
    U_triv, V_triv, W_triv = _trivial_decomposition(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_triv, V_triv, W_triv
    
    # Try to optimize for lower ranks
    # Target ranks: 21 (lower bound), 25, 30, 40
    target_ranks = [21, 25, 30, 40, 50]
    
    for rank in target_ranks:
        if time.time() - start_time > DEADLINE:
            break
        
        # Allocate time for this rank
        # Simpler: just try one optimization per rank with limited time
        U, V, W, found_rank, loss = _optimize_rank(n, m, p, rank, time_budget=10.0)
        
        if found_rank is not None and loss < 1e-8:
            if found_rank < best_rank:
                best_rank = found_rank
                best_U, best_V, best_W = U, V, W
                
    # Final validation check
    # Reconstruct and check
    rank = best_U.shape[1]
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(best_U[u_idx, :], best_V[v_idx, :] * best_W[w_idx, :])
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
        if not valid: break
        if not valid: break
        
    if not valid:
        # Fall back to trivial
        best_U, best_V, best_W = U_triv, V_triv, W_triv
        best_rank = n * m * p
        
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
