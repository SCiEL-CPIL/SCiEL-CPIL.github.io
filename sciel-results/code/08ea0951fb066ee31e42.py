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
        
        # Solve all gammas at once using lstsq with multiple RHS
        b = Target.reshape(dim_U * dim_V, dim_W)
        # We need to solve A^T A W = A^T b for each column of b
        # Actually, let's solve A x = b for each column of b
        # But A is (dim_U*dim_V) x R, b is (dim_U*dim_V) x dim_W
        # So we want W such that A @ W = b
        # This is an underdetermined system if R < dim_U*dim_V
        # Use lstsq
        W_sol, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        # W_sol is (R, dim_W), we need (dim_W, R)
        W = W_sol.T
        
        T_calc = np.einsum('ar,br,cr->abc', U_arr, V_arr, W)
        residual = np.sum((T_calc - Target) ** 2)
        
        return W, residual

    def compute_residual_uv(U, V):
        """Compute residual given only U and V (W is solved implicitly)."""
        W, res = solve_for_W(U, V)
        return res

    def gradient_descent_step(U, V, lr=0.01, max_iter=50, tol=1e-8):
        """
        Perform gradient descent on U and V to minimize the residual.
        Uses analytical gradients.
        """
        U = U.copy()
        V = V.copy()
        
        for iteration in range(max_iter):
            W, current_res = solve_for_W(U, V)
            
            if current_res < tol:
                break
            
            # Compute gradients analytically
            # residual = sum_{a,b,g} (sum_r U[a,r] V[b,r] W[g,r] - T[a,b,g])^2
            # d(residual)/dU[a,r] = 2 * sum_{b,g} (sum_r' U[a,r'] V[b,r'] W[g,r'] - T[a,b,g]) * V[b,r] * W[g,r]
            
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            error = T_calc - Target  # (dim_U, dim_V, dim_W)
            
            # dU: 2 * sum_{b,g} error[a,b,g] * V[b,r] * W[g,r]
            # dU[a,r] = 2 * sum_{b,g} error[a,b,g] * V[b,r] * W[g,r]
            U_grad = 2 * np.einsum('abg,br,gr->ar', error, V, W)
            
            # dV[b,r] = 2 * sum_{a,g} error[a,b,g] * U[a,r] * W[g,r]
            V_grad = 2 * np.einsum('abg,ar,gr->br', error, U, W)
            
            # Line search
            alpha = lr
            for _ in range(10):
                U_new = U - alpha * U_grad
                V_new = V - alpha * V_grad
                W_new, new_res = solve_for_W(U_new, V_new)
                
                if new_res < current_res:
                    U = U_new
                    V = V_new
                    break
                alpha *= 0.5
            
            if alpha < 1e-10:
                break
        
        W, final_res = solve_for_W(U, V)
        return U, V, W, final_res

    def local_search_uv(U, V, max_steps=100, time_budget=5.0):
        """
        Local search on U and V using discrete moves and gradient steps.
        """
        start = time.time()
        U = U.copy()
        V = V.copy()
        W, current_res = solve_for_W(U, V)
        
        values = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
        improved = True
        step = 0
        
        while improved and step < max_steps and (time.time() - start < time_budget):
            improved = False
            step += 1
            
            # Try discrete moves
            for _ in range(20):
                if time.time() - start > time_budget:
                    break
                
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
                    W = W_new
                    improved = True
                else:
                    # Revert change
                    if mat == 'U':
                        U[r_idx, c_idx] = old_val
                    else:
                        V[r_idx, c_idx] = old_val
            
            # Try gradient step if residual is small enough
            if current_res < 1.0 and step % 3 == 0:
                U_grad, V_grad, W_grad, res_grad_check = gradient_descent_step(U, V, lr=0.01, max_iter=20)
                if res_grad_check < current_res:
                    U = U_grad
                    V = V_grad
                    W = W_grad
                    current_res = res_grad_check
                    improved = True
        
        return U, V, W, current_res

    start_time = time.time()
    time_limit = 130.0 # Leave margin for 150s kill

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
            
        # Number of restarts per rank - adjust based on time
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 10:
            break
        num_restarts = min(30, max(5, int(remaining_time / 3)))
        
        for restart in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize U and V with random half-integers
            U = np.array([random.choice(values) for _ in range(dim_U * R)]).reshape(dim_U, R)
            V = np.array([random.choice(values) for _ in range(dim_V * R)]).reshape(dim_V, R)
            
            # Solve for W
            W, res = solve_for_W(U, V)
            
            # Local Search / Hill Climbing on U and V
            time_per_restart = max(1.0, min(3.0, (time_limit - (time.time() - start_time)) / max(1, num_restarts - restart)))
            U_opt, V_opt, W_opt, res_opt = local_search_uv(U, V, max_steps=60, time_budget=time_per_restart)
            
            # If residual is very small, check for exact solution
            if res_opt < 1e-9:
                if is_half_integer(W_opt):
                    W_snap = snap_to_half_int(W_opt)
                    if verify(U_opt, V_opt, W_snap):
                        best_result = {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_snap.tolist()}
                        best_rank = R
                        found = True
                        break
            elif res_opt < 1e-6:
                # Try to polish with more iterations
                U_pol, V_pol, W_pol, res_pol = gradient_descent_step(U_opt, V_opt, lr=0.005, max_iter=100, tol=1e-12)
                if res_pol < 1e-9 and is_half_integer(W_pol):
                    W_snap = snap_to_half_int(W_pol)
                    if verify(U_pol, V_pol, W_snap):
                        best_result = {"U": U_pol.tolist(), "V": V_pol.tolist(), "W": W_snap.tolist()}
                        best_rank = R
                        found = True
                        break
        
        if found:
            break

    if best_result is None:
        # Fallback to schoolbook if no better found
        best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return best_result
# EVOLVE-BLOCK-END
