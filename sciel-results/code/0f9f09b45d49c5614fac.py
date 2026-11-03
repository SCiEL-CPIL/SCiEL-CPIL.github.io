import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    n, m, p = DIMENSIONS
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Target Tensor Indices
    # T[i, j, k] = 1
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_indices.append((i * m + j, j * p + k, k * n + i))
                
    target_indices = np.array(target_indices)
    u_idx = target_indices[:, 0]
    v_idx = target_indices[:, 1]
    w_idx = target_indices[:, 2]
    
    def objective(params, R):
        # Unpack params into U, V, W
        # U: n*m x R
        # V: m*p x R
        # W: p*n x R
        size_u = n * m * R
        size_v = m * p * R
        size_w = p * n * R
        
        U = params[:size_u].reshape(n * m, R)
        V = params[size_u:size_u + size_v].reshape(m * p, R)
        W = params[size_u + size_v:].reshape(p * n, R)
        
        # Compute residual
        # We only care about the target indices
        # Residual = T - Sum(U_i V_j W_k)
        # T is 1 at target indices.
        
        # Efficient computation:
        # For each target index (u, v, w), compute sum_c U[u,c] V[v,c] W[w,c]
        # This is a batch of dot products.
        
        # U[u_idx, :] shape (40, R)
        # V[v_idx, :] shape (40, R)
        # W[w_idx, :] shape (40, R)
        
        # Elementwise multiply and sum over R
        # (40, R) * (40, R) * (40, R) -> (40, R) -> sum(axis=1) -> (40,)
        val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :], axis=1)
        residual = 1.0 - val
        return np.sum(residual**2)
        
    def gradient(params, R):
        # Analytical gradient might be faster, but numerical is safer to implement correctly.
        # L-BFGS-B uses numerical gradient by default if jac is not provided.
        # Given small size, numerical gradient is fine.
        pass

    best_solution = None
    best_score = float('inf')
    
    # Try Rank 36
    target_rank = 36
    
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        # Random initialization
        np.random.seed(42 + r)
        U_init = np.random.randn(n * m, target_rank)
        V_init = np.random.randn(m * p, target_rank)
        W_init = np.random.randn(p * n, target_rank)
        
        # Scale initialization to help convergence
        # Normalize columns?
        for c in range(target_rank):
            norm_u = np.linalg.norm(U_init[:, c])
            norm_v = np.linalg.norm(V_init[:, c])
            norm_w = np.linalg.norm(W_init[:, c])
            # Avoid division by zero
            if norm_u > 1e-9: U_init[:, c] /= norm_u
            if norm_v > 1e-9: V_init[:, c] /= norm_v
            if norm_w > 1e-9: W_init[:, c] /= norm_w
            
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            res = minimize(objective, params_init, args=(target_rank,), method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
            if res.success and res.fun < 1e-6:
                # Extract solution
                size_u = n * m * target_rank
                size_v = m * p * target_rank
                U_sol = res.x[:size_u].reshape(n * m, target_rank)
                V_sol = res.x[size_u:size_u + size_v].reshape(m * p, target_rank)
                W_sol = res.x[size_u + size_v:].reshape(p * n, target_rank)
                
                # Check for integer/half-integer
                # Multiply by 2 and check if close to integer
                U_check = np.round(U_sol * 2) / 2
                V_check = np.round(V_sol * 2) / 2
                W_check = np.round(W_sol * 2) / 2
                
                # Verify exactness with rounded values
                val = np.sum(U_check[u_idx, :] * V_check[v_idx, :] * W_check[w_idx, :], axis=1)
                if np.allclose(val, 1.0, atol=1e-9):
                    # Convert to list of lists as required
                    U_list = U_check.tolist()
                    V_list = V_check.tolist()
                    W_list = W_check.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
                    
        except Exception:
            pass

    # Fallback to schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

# EVOLVE-BLOCK-START
# The function construct is defined above.
# We need to ensure the structure matches the request.
# The request asks to rewrite the program.
# I will place the code in the EVOLVE-BLOCK.
# EVOLVE-BLOCK-END
