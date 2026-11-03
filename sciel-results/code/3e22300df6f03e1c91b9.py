import numpy as np
import scipy.optimize as opt

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Target rank to try
    # Schoolbook is n*m*p = 120.
    # We try to find a decomposition of rank 119.
    # If that fails, we fall back to 120.
    
    target_rank = n * m * p - 1
    
    # Create the target tensor T_target of shape (n, m, p)
    # T_target[i, j, k] = 1 for all i, j, k
    T_target = np.ones((n, m, p), dtype=float)
    
    def compute_tensor(U, V, W):
        """
        U: (n*m, rank)
        V: (m*p, rank)
        W: (p*n, rank)
        
        Returns tensor of shape (n, m, p)
        """
        rank = U.shape[1]
        U_3d = U.reshape(n, m, rank)
        V_3d = V.reshape(m, p, rank)
        W_3d = W.reshape(p, n, rank)
        
        # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
        # Note: W is indexed by [k, i, r]
        # We can do this in two steps for efficiency
        # Temp[i, j, k, r] is not needed if we use einsum directly
        # But einsum with 3 operands is fine for small sizes.
        
        # Let's use einsum
        # 'ijr,jkr,kr->ijk' would be if W was (k, r)
        # But W is (p, n, rank) -> W_3d[k, i, r]
        # So we need 'ijr,jkr,ki r->ijk'
        
        T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        return T

    def objective(params):
        """
        params: flattened array of U, V, W
        """
        U_flat_size = n * m * target_rank
        V_flat_size = m * p * target_rank
        W_flat_size = p * n * target_rank
        
        U_flat = params[:U_flat_size]
        V_flat = params[U_flat_size:U_flat_size + V_flat_size]
        W_flat = params[U_flat_size + V_flat_size:]
        
        U = U_flat.reshape(n, m, target_rank)
        V = V_flat.reshape(m, p, target_rank)
        W = W_flat.reshape(p, n, target_rank)
        
        T = compute_tensor(U, V, W)
        error = T - T_target
        return error.flatten()

    # Try to find a solution with rank 119
    # We use multiple random restarts
    found_solution = False
    U_sol = None
    V_sol = None
    W_sol = None
    
    for restart in range(20):
        np.random.seed(42 + restart)
        # Initialize with small random integers
        U_init = np.random.randint(-1, 2, (n, m, target_rank)).astype(float)
        V_init = np.random.randint(-1, 2, (m, p, target_rank)).astype(float)
        W_init = np.random.randint(-1, 2, (p, n, target_rank)).astype(float)
        
        params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        try:
            result = opt.least_squares(objective, params_init, method='lm', max_nfev=1000)
            
            if result.success or np.linalg.norm(result.fun) < 1e-6:
                # Extract solution
                U_flat_size = n * m * target_rank
                V_flat_size = m * p * target_rank
                
                U_sol = result.x[:U_flat_size].reshape(n, m, target_rank)
                V_sol = result.x[U_flat_size:U_flat_size + V_flat_size].reshape(m, p, target_rank)
                W_sol = result.x[U_flat_size + V_flat_size:].reshape(p, n, target_rank)
                
                # Round to nearest integer
                U_int = np.round(U_sol).astype(int)
                V_int = np.round(V_sol).astype(int)
                W_int = np.round(W_sol).astype(int)
                
                # Verify the integer solution
                T_check = compute_tensor(U_int, V_int, W_int)
                if np.allclose(T_check, T_target, atol=1e-9):
                    found_solution = True
                    break
        except Exception:
            continue

    if found_solution:
        # Return in the required format
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        U_out = U_int.reshape(n * m, target_rank)
        V_out = V_int.reshape(m * p, target_rank)
        W_out = W_int.reshape(p * n, target_rank)
        
        # Convert to lists of lists as per interface? 
        # The interface says "factor matrices of shapes...". 
        # The seed program returned lists of lists. 
        # Let's check if numpy arrays are accepted. 
        # "integer/half-integer entries". 
        # Usually, returning numpy arrays is fine if the evaluator handles it, 
        # but the seed used lists. To be safe, I'll convert to lists.
        
        U_list = U_out.tolist()
        V_list = V_out.tolist()
        W_list = W_out.tolist()
        
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
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
