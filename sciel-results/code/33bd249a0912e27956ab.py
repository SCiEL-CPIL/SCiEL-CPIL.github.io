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
    
    # Build target tensor
    Target = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                Target[alpha, beta, gamma] = 1.0
    
    # Precompute the A matrix for each rank R: A is (dim_U*dim_V) x R
    # A[alpha*dim_V + beta, r] = U[alpha, r] * V[beta, r]
    # This depends on U and V, so we compute it on the fly but cache structure
    
    def verify(U, V, W, tol=1e-9):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        T_calc = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        return np.allclose(T_calc, Target, atol=tol)
    
    def snap_to_half_int(x):
        return np.round(2 * x) / 2.0
    
    def is_half_integer(x, tol=1e-5):
        return np.allclose(x, snap_to_half_int(x), atol=tol)
    
    def compute_residual(U, V):
        """Compute the minimum residual over W for given U, V."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        R = U_arr.shape[1]
        M = np.einsum('ar,br->abr', U_arr, V_arr)
        A = M.reshape(dim_U * dim_V, R)
        
        residual = 0.0
        for gamma in range(dim_W):
            b = Target[:, :, gamma].reshape(dim_U * dim_V)
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            r = b - A @ x
            residual += np.sum(r**2)
        return residual
    
    def compute_residual_and_W(U, V):
        """Compute residual and W for given U, V."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        R = U_arr.shape[1]
        M = np.einsum('ar,br->abr', U_arr, V_arr)
        A = M.reshape(dim_U * dim_V, R)
        
        W = np.zeros((dim_W, R))
        residual = 0.0
        for gamma in range(dim_W):
            b = Target[:, :, gamma].reshape(dim_U * dim_V)
            x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            W[gamma, :] = x
            r = b - A @ x
            residual += np.sum(r**2)
        return residual, W
    
    def objective_and_grad(params, R):
        """
        Objective: sum of squared residuals from least-squares W for each gamma.
        params: flattened [U_flat, V_flat] of length dim_U*R + dim_V*R
        Returns: (objective_value, gradient)
        """
        U_flat = params[:dim_U * R].reshape(dim_U, R)
        V_flat = params[dim_U * R:].reshape(dim_V, R)
        
        # Compute A = U ⊗ V (element-wise outer product per rank)
        M = np.einsum('ar,br->abr', U_flat, V_flat)
        A = M.reshape(dim_U * dim_V, R)
        
        # Compute residual and gradient
        # For each gamma: minimize ||A x - b_gamma||^2
        # The optimal x_gamma = (A^T A)^{-1} A^T b_gamma (if A has full column rank)
        # Residual_gamma = ||b_gamma||^2 - b_gamma^T A (A^T A)^{-1} A^T b_gamma
        
        # More efficiently: compute the projection
        try:
            ATA = A.T @ A
            # Add small regularization for numerical stability
            ATA_reg = ATA + 1e-12 * np.eye(R)
            ATA_inv = np.linalg.inv(ATA_reg)
        except np.linalg.LinAlgError:
            ATA_inv = np.linalg.pinv(ATA)
        
        obj = 0.0
        grad_U = np.zeros_like(U_flat)
        grad_V = np.zeros_like(V_flat)
        
        for gamma in range(dim_W):
            b = Target[:, :, gamma].reshape(dim_U * dim_V)
            # x = ATA_inv @ A^T @ b
            x = ATA_inv @ (A.T @ b)
            # residual vector
            r = b - A @ x
            # residual contribution
            obj += np.sum(r**2)
            
            # Gradient of ||A x - b||^2 with respect to A
            # d/dA ||A x - b||^2 = 2 (A x - b) x^T = 2 r x^T
            # But x depends on A, so we need the full derivative
            # 
            # Actually, for the least squares problem, the gradient of the minimum
            # residual with respect to A is:
            # d/dA min_x ||Ax-b||^2 = 2 (Ax-b) x^T where x is the minimizer
            # This is a known result from matrix calculus.
            
            grad_A = 2 * np.outer(r, x)
            
            # Now backpropagate through A = M.reshape(...)
            # M[alpha, beta, r] = U[alpha, r] * V[beta, r]
            # dM[alpha,beta,r] = dA[alpha*dim_V+beta, r]
            grad_M = grad_A.reshape(dim_U, dim_V, R)
            
            # d/dU[alpha, r] = sum_beta grad_M[alpha, beta, r] * V[beta, r]
            # d/dV[beta, r] = sum_alpha grad_M[alpha, beta, r] * U[alpha, r]
            for r_idx in range(R):
                grad_U[:, r_idx] += np.sum(grad_M[:, :, r_idx] * V_flat[:, r_idx], axis=1)
                grad_V[:, r_idx] += np.sum(grad_M[:, :, r_idx] * U_flat[:, r_idx], axis=0)
        
        grad_flat = np.concatenate([grad_U.flatten(), grad_V.flatten()])
        return obj, grad_flat
    
    def objective(params, R):
        obj, _ = objective_and_grad(params, R)
        return obj
    
    def snap_and_verify(U, V, R, tol=1e-9):
        """Snap U, V, W to half-integers and verify."""
        residual, W = compute_residual_and_W(U, V)
        
        if residual > 1e-6:
            return None
        
        U_snap = snap_to_half_int(U)
        V_snap = snap_to_half_int(V)
        W_snap = snap_to_half_int(W)
        
        if verify(U_snap, V_snap, W_snap, tol=tol):
            return U_snap, V_snap, W_snap
        
        # Try slightly different snapping
        for tol2 in [1e-8, 1e-7, 1e-6]:
            if verify(U_snap, V_snap, W_snap, tol=tol2):
                return U_snap, V_snap, W_snap
        
        return None
    
    start_time = time.time()
    time_limit = 115.0  # Leave 35s margin
    
    # Schoolbook solution
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
                
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    found = False
    
    # Try ranks from lower bound up
    # Lower bound is max(dim_U, dim_V, dim_W) = max(15, 25, 15) = 25
    # But realistic lower bound for matrix mult is higher
    # Try from 25 to 74
    
    # Time budget per rank
    ranks_to_try = list(range(25, best_rank))
    
    # Sort by likelihood: try ranks that are more likely to work first
    # For 3x5x5, known results suggest rank around 40-50 might be achievable
    # Let's try a mix: some low, some medium
    
    # Use a priority queue approach: try ranks in order of increasing difficulty
    # But first, let's do a quick feasibility check for each rank
    
    for R in ranks_to_try:
        if found:
            break
        if time.time() - start_time > time_limit:
            break
        
        # Time budget for this rank
        time_spent = time.time() - start_time
        time_remaining = time_limit - time_spent
        # Allocate proportional time based on rank
        # Lower ranks get more time since they're harder
        time_for_rank = min(20.0, max(2.0, time_remaining * (R - 24) / (best_rank - 25)))
        
        if time_for_rank < 1.0:
            continue
        
        # Multiple restarts
        num_restarts = max(3, int(time_for_rank / 3.0))
        
        for restart in range(num_restarts):
            if found:
                break
            if time.time() - start_time > time_limit:
                break
            
            # Initialize U and V
            # Strategy 1: Random small values
            # Strategy 2: Structured initialization
            # Strategy 3: Near-schoolbook perturbation
            
            if restart == 0:
                # Structured: try to use the schoolbook pattern but compressed
                U = np.zeros((dim_U, R))
                V = np.zeros((dim_V, R))
                # Assign each (i,j,k) triple to a random rank slot
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            r_slot = random.randint(0, R - 1)
                            U[i*m+j, r_slot] += 1.0
                            V[j*p+k, r_slot] += 1.0
                            # W will be solved
            elif restart == 1:
                # Random half-integers
                values = [-1.0, -0.5, 0.0, 0.5, 1.0]
                U = np.array([random.choice(values) for _ in range(dim_U * R)]).reshape(dim_U, R)
                V = np.array([random.choice(values) for _ in range(dim_V * R)]).reshape(dim_V, R)
            else:
                # Gaussian initialization
                U = np.random.randn(dim_U, R) * 0.5
                V = np.random.randn(dim_V, R) * 0.5
            
            # Compute initial residual
            initial_res, _ = compute_residual_and_W(U, V)
            
            if initial_res < 1e-9:
                result = snap_and_verify(U, V, R)
                if result is not None:
                    U_s, V_s, W_s = result
                    best_result = {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                    best_rank = R
                    found = True
                    break
            
            # Gradient-based optimization
            params0 = np.concatenate([U.flatten(), V.flatten()])
            
            # Use L-BFGS-B with the gradient
            try:
                result_opt = minimize(
                    objective,
                    params0,
                    args=(R,),
                    method='L-BFGS-B',
                    jac=lambda p, R: objective_and_grad(p, R)[1],
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 100}
                )
                
                if result_opt.success or result_opt.fun < initial_res:
                    params_opt = result_opt.x
                    U_opt = params_opt[:dim_U * R].reshape(dim_U, R)
                    V_opt = params_opt[dim_U * R:].reshape(dim_V, R)
                    
                    opt_res, W_opt = compute_residual_and_W(U_opt, V_opt)
                    
                    if opt_res < 1e-6:
                        result = snap_and_verify(U_opt, V_opt, R)
                        if result is not None:
                            U_s, V_s, W_s = result
                            best_result = {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                            best_rank = R
                            found = True
                            break
                    
                    # If close but not exact, try to polish
                    if opt_res < 1e-3:
                        # Local search on half-integers
                        U_local = snap_to_half_int(U_opt)
                        V_local = snap_to_half_int(V_opt)
                        
                        # Try small perturbations
                        for _ in range(20):
                            if time.time() - start_time > time_limit:
                                break
                            # Pick random entry to flip
                            if random.random() < 0.5:
                                idx = random.randint(0, dim_U * R - 1)
                                i_idx, r_idx = divmod(idx, R)
                                old_val = U_local[i_idx, r_idx]
                                U_local[i_idx, r_idx] = old_val + 0.5
                                res_test, _ = compute_residual_and_W(U_local, V_local)
                                if res_test > opt_res:
                                    U_local[i_idx, r_idx] = old_val
                            else:
                                idx = random.randint(0, dim_V * R - 1)
                                i_idx, r_idx = divmod(idx, R)
                                old_val = V_local[i_idx, r_idx]
                                V_local[i_idx, r_idx] = old_val + 0.5
                                res_test, _ = compute_residual_and_W(U_local, V_local)
                                if res_test > opt_res:
                                    V_local[i_idx, r_idx] = old_val
                        
                        result = snap_and_verify(U_local, V_local, R)
                        if result is not None:
                            U_s, V_s, W_s = result
                            best_result = {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                            best_rank = R
                            found = True
                            break
                        
            except Exception:
                pass
        
        if found:
            break
    
    if best_result is None:
        best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return best_result
# EVOLVE-BLOCK-END
