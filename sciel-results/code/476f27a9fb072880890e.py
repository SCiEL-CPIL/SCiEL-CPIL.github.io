import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    target_rank = n * m * p
    best_solution = None
    best_rank = target_rank
    
    # Precompute indices for the tensor
    # T[i, j, k] = 1
    # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Helper to create the schoolbook decomposition
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
        return U, V, W, rank

    # Initial best is schoolbook
    U_init, V_init, W_init, best_rank = schoolbook()
    
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Search for lower ranks
    # Theoretical lower bound is n+m+p-2 = 8
    # We try ranks from 9 up to best_rank - 1
    for rank in range(9, best_rank):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        found_solution = False
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            # Initialize with small random integers/half-integers
            # Scaling matters. Let's use small values.
            U = np.random.randint(-2, 3, size=(n * m, rank)) * 0.5
            V = np.random.randint(-2, 3, size=(m * p, rank)) * 0.5
            W = np.random.randint(-2, 3, size=(p * n, rank)) * 0.5
            
            # Flatten
            def objective(params):
                U_flat = params[0:n*m*rank]
                V_flat = params[n*m*rank:n*m*rank + m*p*rank]
                W_flat = params[n*m*rank + m*p*rank:]
                
                U_m = U_flat.reshape(n * m, rank)
                V_m = V_flat.reshape(m * p, rank)
                W_m = W_flat.reshape(p * n, rank)
                
                error = 0.0
                # Calculate tensor contraction
                # For each i, j, k, sum over r
                # Vectorized: 
                # Create index arrays
                # This loop is small (30 iterations), so it's fast enough
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_val = U_m[i * m + j, :]
                            v_val = V_m[j * p + k, :]
                            w_val = W_m[k * n + i, :]
                            # sum(u * v * w)
                            prod_sum = np.sum(u_val * v_val * w_val)
                            error += (prod_sum - 1.0) ** 2
                return error

            # We need to provide a good initial guess.
            # Random restarts might get stuck.
            # Let's try to use the structure.
            # For now, just run minimize.
            initial_params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use Nelder-Mead as it doesn't require gradients
            res = minimize(objective, initial_params, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.fun < 1e-6:
                # Check for half-integer solution
                U_opt = res.x[0:n*m*rank].reshape(n * m, rank)
                V_opt = res.x[n*m*rank:n*m*rank + m*p*rank].reshape(m * p, rank)
                W_opt = res.x[n*m*rank + m*p*rank:].reshape(p * n, rank)
                
                # Round to nearest half-integer
                U_int = np.round(U_opt * 2) / 2
                V_int = np.round(V_opt * 2) / 2
                W_int = np.round(W_opt * 2) / 2
                
                # Verify exactness
                is_valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_val = U_int[i * m + j, :]
                            v_val = V_int[j * p + k, :]
                            w_val = W_int[k * n + i, :]
                            prod_sum = np.sum(u_val * v_val * w_val)
                            if not np.isclose(prod_sum, 1.0, atol=1e-9):
                                is_valid = False
                                break
                        if not is_valid:
                            break
                    if not is_valid:
                        break
                
                if is_valid:
                    best_solution = (U_int, V_int, W_int, rank)
                    found_solution = True
                    break # Break attempt loop, try next rank (lower)
        
        if found_solution:
            # We found a solution for this rank. 
            # Since we are iterating upwards from 9, the first one found is the minimal rank?
            # Not necessarily, if we miss a solution for rank 9 but find one for 10.
            # But we break the rank loop if we find one? 
            # No, we should continue to see if a lower rank exists? 
            # We are iterating rank from 9 upwards. So the first found is the lowest.
            break

    if best_solution:
        U, V, W, rank = best_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        U, V, W, rank = U_init, V_init, W_init, best_rank
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
