import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Target tensor T[alpha, beta, gamma]
    # alpha = i*m + j
    # beta = j*p + k
    # gamma = k*n + i
    Target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                Target[alpha, beta, gamma] = 1.0

    def verify(U, V, W, tol=1e-9):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        T_calc = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        if not np.allclose(T_calc, Target, atol=tol):
            return False
        return True

    def snap_to_half_int(x):
        return np.round(2 * x) / 2.0

    def is_half_integer(x, tol=1e-5):
        return np.allclose(x, snap_to_half_int(x), atol=tol)

    def compute_residual(U, V, W):
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((T_calc - Target) ** 2)

    def solve_for_W(U, V):
        """
        Given U and V, solve for W using least squares for each gamma.
        Returns W and the residual.
        """
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        
        R = U_arr.shape[1]
        
        # M[alpha, beta, r] = U[alpha, r] * V[beta, r]
        M = np.einsum('ar,br->abr', U_arr, V_arr)
        
        W = np.zeros((dim_W, R))
        residual = 0.0
        
        # A is (dim_U * dim_V) x R
        A = M.reshape(dim_U * dim_V, R)
        
        for gamma in range(dim_W):
            b = Target[:, :, gamma].reshape(dim_U * dim_V)
            # Solve A x = b
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            W[gamma, :] = x
            
            # Calculate residual
            r = b - A @ x
            residual += np.sum(r**2)
            
        return W, residual

    start_time = time.time()
    time_limit = 110.0 # Leave margin

    # Schoolbook solution (Rank 75)
    best_rank = n * m * p
    U_sb = np.zeros((dim_U, best_rank))
    V_sb = np.zeros((dim_V, best_rank))
    W_sb = np.zeros((dim_W, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
                
    if verify(U_sb, V_sb, W_sb):
        best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    else:
        best_result = None # Should not happen

    # Discrete Search Strategy
    # We search for ranks from 25 (theoretical lower bound mp=25) up to 74.
    # For each rank, we do random initialization and local search (hill climbing).
    
    values = [-1.0, -0.5, 0.0, 0.5, 1.0]
    
    found = False
    
    for R in range(25, best_rank):
        if time.time() - start_time > time_limit:
            break
            
        # Number of restarts per rank
        # Allocate time budget dynamically? Or fixed?
        # Let's try a fixed number of restarts, but check time frequently.
        num_restarts = 100
        
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U and V with random half-integers
            U = np.array([random.choice(values) for _ in range(dim_U * R)]).reshape(dim_U, R)
            V = np.array([random.choice(values) for _ in range(dim_V * R)]).reshape(dim_V, R)
            
            # Solve for W
            W, res = solve_for_W(U, V)
            
            # Local Search / Hill Climbing on W? No, W is determined by U,V.
            # We should optimize U and V.
            # Let's try a simple local search on U and V entries.
            
            current_res = res
            improved = True
            max_steps = 50
            
            while improved and max_steps > 0 and (time.time() - start_time < time_limit):
                improved = False
                max_steps -= 1
                
                # Try changing a random entry in U or V
                # Pick random matrix
                if random.random() < 0.5:
                    mat = 'U'
                    arr = U
                else:
                    mat = 'V'
                    arr = V
                    
                r_idx, c_idx = random.randint(0, arr.shape[0]-1), random.randint(0, arr.shape[1]-1)
                
                old_val = arr[r_idx, c_idx]
                new_val = random.choice([v for v in values if v != old_val])
                
                # Apply change
                if mat == 'U':
                    U[r_idx, c_idx] = new_val
                else:
                    V[r_idx, c_idx] = new_val
                    
                # Recalculate W and residual
                W_new, res_new = solve_for_W(U, V)
                
                if res_new < current_res:
                    current_res = res_new
                    improved = True
                    # If residual is very small, check for exact solution
                    if current_res < 1e-9:
                        if is_half_integer(W_new):
                            W_snap = snap_to_half_int(W_new)
                            if verify(U, V, W_snap):
                                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W_snap.tolist()}
                                best_rank = R
                                found = True
                                break
                else:
                    # Revert change
                    if mat == 'U':
                        U[r_idx, c_idx] = old_val
                    else:
                        V[r_idx, c_idx] = old_val
                        
            if found:
                break
                
        if found:
            break

    if best_result is None:
        # Fallback to schoolbook if no better found
        best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return best_result
# EVOLVE-BLOCK-END
