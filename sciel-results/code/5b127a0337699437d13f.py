import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m  # 15
    dim_V = m * p  # 25
    dim_W = p * n  # 15
    
    start_time = time.time()
    time_limit = 115.0  # Leave safety margin
    
    # Target tensor T[alpha, beta, gamma]
    # alpha = i*m + j (i=0..n-1, j=0..m-1)
    # beta = j*p + k (j=0..m-1, k=0..p-1)  
    # gamma = k*n + i (k=0..p-1, i=0..n-1)
    Target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                Target[alpha, beta, gamma] = 1.0

    def verify_fast(U, V, W, tol=1e-9):
        """Fast verification using einsum"""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        T_calc = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        return np.allclose(T_calc, Target, atol=tol)

    def snap_to_half_int(x):
        return np.round(2 * x) / 2.0

    def is_half_integer(x, tol=1e-5):
        return np.allclose(x, snap_to_half_int(x), atol=tol)

    # Schoolbook decomposition (rank 75)
    def schoolbook():
        U = np.zeros((dim_U, n*m*p))
        V = np.zeros((dim_V, n*m*p))
        W = np.zeros((dim_W, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    # Try to find a lower rank decomposition using structured approach
    # Key insight: We can exploit the tensor structure
    
    # Method 1: Try rank 50 (reduction from 75)
    # The tensor has structure: for each (i,j,k), we have a 1 in T[i*m+j, j*p+k, k*n+i]
    # We can try to merge some of these
    
    best_result = None
    best_rank = n * m * p  # 75
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = schoolbook()
    if verify_fast(U_sb, V_sb, W_sb):
        best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
        best_rank = n * m * p
    
    # Try to find lower rank solutions
    # Strategy: Use the fact that the tensor is sparse and structured
    
    # For 3x5x5, we can think of it as 3 separate 1x5x5 problems
    # Each 1x5x5 problem has rank at most 5 (schoolbook)
    # So total rank at most 3*5 = 15? No, that's not right.
    # Actually, the structure couples them.
    
    # Let's try a different approach: use random search with better initialization
    # and more efficient solving
    
    def solve_for_W_fast(U, V):
        """
        Given U and V, solve for W using least squares.
        Returns W and the residual.
        """
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        
        R = U_arr.shape[1]
        
        # M[alpha, beta, r] = U[alpha, r] * V[beta, r]
        M = np.einsum('ar,br->abr', U_arr, V_arr)
        
        W = np.zeros((dim_W, R))
        residual = 0.0
        
        # For each gamma, solve the linear system
        for gamma in range(dim_W):
            A = M.reshape(dim_U * dim_V, R)
            b = Target[:, :, gamma].reshape(dim_U * dim_V)
            
            # Use lstsq
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            W[gamma, :] = x
            
            r = b - A @ x
            residual += np.sum(r**2)
        
        return W, residual

    # Try structured random search
    # Use half-integer values for better chance of exact solutions
    
    for R in range(45, best_rank + 1):
        if time.time() - start_time > time_limit:
            break
        
        found = False
        max_restarts = 30
        
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize with structured patterns
            # Try different initialization strategies
            
            if restart % 3 == 0:
                # Strategy 1: Sparse random
                U_init = np.zeros((dim_U, R))
                V_init = np.zeros((dim_V, R))
                for r in range(R):
                    # Pick random non-zero positions
                    u_idx = np.random.choice(dim_U, size=min(3, dim_U), replace=False)
                    v_idx = np.random.choice(dim_V, size=min(3, dim_V), replace=False)
                    U_init[u_idx, r] = np.random.choice([-1, -0.5, 0.5, 1])
                    V_init[v_idx, r] = np.random.choice([-1, -0.5, 0.5, 1])
            elif restart % 3 == 1:
                # Strategy 2: Block structured
                U_init = np.zeros((dim_U, R))
                V_init = np.zeros((dim_V, R))
                for r in range(R):
                    # Assign blocks
                    i_block = r % n
                    j_block = (r // n) % m
                    k_block = (r // (n*m)) % p
                    U_init[i_block*m + j_block, r] = 1
                    V_init[j_block*p + k_block, r] = 1
            else:
                # Strategy 3: Dense random
                U_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(dim_U, R))
                V_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(dim_V, R))
            
            # Check if this initialization works
            W_check, res_check = solve_for_W_fast(U_init, V_init)
            if res_check < 1e-9:
                if is_half_integer(W_check):
                    W_snap = snap_to_half_int(W_check)
                    if verify_fast(U_init, V_init, W_snap):
                        best_result = {"U": U_init.tolist(), "V": V_init.tolist(), "W": W_snap.tolist()}
                        best_rank = R
                        found = True
                        break
            
            # If not found, try to optimize
            if time.time() - start_time > time_limit:
                break
                
            # Use a simpler optimization: coordinate descent on U and V
            # Or use scipy with tight constraints
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten()])
            
            def objective(params):
                U_size = dim_U * R
                U_flat = params[:U_size]
                V_flat = params[U_size:]
                U = U_flat.reshape(dim_U, R)
                V = V_flat.reshape(dim_V, R)
                _, res = solve_for_W_fast(U, V)
                return res
            
            try:
                # Use L-BFGS-B with limited iterations
                res = minimize(objective, params_init, method='L-BFGS-B', 
                              options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-6})
            except Exception:
                continue
                
            if res.fun < 1e-8:
                params_opt = res.x
                U_opt = params_opt[:dim_U*R].reshape(dim_U, R)
                V_opt = params_opt[dim_U*R:].reshape(dim_V, R)
                
                W_opt, res_final = solve_for_W_fast(U_opt, V_opt)
                
                if res_final < 1e-8:
                    # Try to snap to half-integers
                    if is_half_integer(U_opt) and is_half_integer(V_opt) and is_half_integer(W_opt):
                        U_snap = snap_to_half_int(U_opt)
                        V_snap = snap_to_half_int(V_opt)
                        W_snap = snap_to_half_int(W_opt)
                        if verify_fast(U_snap, V_snap, W_snap):
                            best_result = {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                            best_rank = R
                            found = True
                            break
                    elif is_half_integer(W_opt):
                        W_snap = snap_to_half_int(W_opt)
                        if verify_fast(U_opt, V_opt, W_snap):
                            best_result = {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_snap.tolist()}
                            best_rank = R
                            found = True
                            break

        if found:
            break

    if best_result is None:
        best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return best_result
# EVOLVE-BLOCK-END
