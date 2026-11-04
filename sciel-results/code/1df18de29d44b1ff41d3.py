import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor T is defined by T[(i,j), (j,k), (i,k)] = 1
    # We flatten the indices.
    # U: (n*m, R)
    # V: (m*p, R)
    # W: (p*n, R) -> Note: prompt says W[k*n+i, r]. So index is k*n+i.
    
    # We try to find a decomposition of rank R=20.
    R = 20
    
    # Precompute target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T_target[a, b, c] = 1
    
    # Initialize U, V, W with better random initialization
    np.random.seed(42)
    
    # Use structured initialization that partially satisfies constraints
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Assign some columns to satisfy the main constraints approximately
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col < R:
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
    
    # Add some random perturbation to break symmetry
    U += np.random.randn(n*m, R) * 0.1
    V += np.random.randn(m*p, R) * 0.1
    W += np.random.randn(p*n, R) * 0.1
    
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s limit
    
    def residual_vectorized(params):
        """Compute residual using efficient tensor operations"""
        size_U = n*m * R
        size_V = m*p * R
        size_W = p*n * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:size_U+size_V+size_W]
        
        U_mat = U_flat.reshape(n*m, R)
        V_mat = V_flat.reshape(m*p, R)
        W_mat = W_flat.reshape(p*n, R)
        
        # T_decomp[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # Efficient computation using matrix multiplication
        # T_decomp = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        
        # Alternative: compute as (U * W) @ V^T or similar
        # Let's use the direct einsum which is clear and reasonably fast for small sizes
        T_decomp = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat, optimize=True)
        
        return (T_decomp - T_target).flatten()
    
    # Initial guess
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    best_result = None
    best_cost = np.inf
    
    try:
        # Try multiple optimization attempts with different initializations
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            # Different random seeds for diversity
            np.random.seed(42 + attempt * 100)
            
            U_init = np.zeros((n*m, R))
            V_init = np.zeros((m*p, R))
            W_init = np.zeros((p*n, R))
            
            # Partially structured initialization
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < R:
                            U_init[i*m + j, col] = 1.0
                            V_init[j*p + k, col] = 1.0
                            W_init[k*n + i, col] = 1.0
                            col += 1
            
            # Add perturbation
            U_init += np.random.randn(n*m, R) * (0.05 + 0.05 * attempt)
            V_init += np.random.randn(m*p, R) * (0.05 + 0.05 * attempt)
            W_init += np.random.randn(p*n, R) * (0.05 + 0.05 * attempt)
            
            x0_attempt = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            result = least_squares(
                residual_vectorized, 
                x0_attempt, 
                method='lm', 
                max_nfev=200,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10
            )
            
            if result.cost < best_cost:
                best_cost = result.cost
                best_result = result
            
            if best_cost < 1e-8:
                break
            
            if time.time() > deadline:
                break
        
        if best_result is not None and best_cost < 1e-6:
            # Extract solution
            size_U = n*m * R
            size_V = m*p * R
            size_W = p*n * R
            
            U_sol = best_result.x[:size_U].reshape(n*m, R)
            V_sol = best_result.x[size_U:size_U+size_V].reshape(m*p, R)
            W_sol = best_result.x[size_U+size_V:size_U+size_V+size_W].reshape(p*n, R)
            
            # Round to nearest 0.5
            U_sol = np.round(U_sol * 2) / 2
            V_sol = np.round(V_sol * 2) / 2
            W_sol = np.round(W_sol * 2) / 2
            
            # Verify
            T_decomp = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol, optimize=True)
            
            if np.allclose(T_decomp, T_target, atol=1e-6):
                # Try to reduce rank by removing zero columns
                # Check which columns are all zero
                col_norms_U = np.linalg.norm(U_sol, axis=0)
                col_norms_V = np.linalg.norm(V_sol, axis=0)
                col_norms_W = np.linalg.norm(W_sol, axis=0)
                
                # Keep only non-zero columns
                mask = (col_norms_U > 1e-6) & (col_norms_V > 1e-6) & (col_norms_W > 1e-6)
                
                if mask.sum() < R:
                    U_sol = U_sol[:, mask]
                    V_sol = V_sol[:, mask]
                    W_sol = W_sol[:, mask]
                    
                    # Re-verify
                    T_decomp = np.einsum('ar,br,cr->abc', U_sol, V_sol, W_sol, optimize=True)
                    if np.allclose(T_decomp, T_target, atol=1e-6):
                        return {
                            "U": U_sol.tolist(),
                            "V": V_sol.tolist(),
                            "W": W_sol.tolist()
                        }
                
                return {
                    "U": U_sol.tolist(),
                    "V": V_sol.tolist(),
                    "W": W_sol.tolist()
                }
                
    except Exception:
        pass
        
    return _schoolbook()

def _schoolbook():
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
