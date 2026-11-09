"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as so

def find_root_system(F, x0):
    """
    Robust root finder for 2D systems using Global Optimization formulation.
    
    Strategy:
    1. Formulate as minimizing g(x) = ||F(x)||^2.
    2. Use a derivative-free approach with a local quadratic model updated via 3 function evaluations per iteration.
    3. Adaptive step-halving line search along the steepest descent direction of the model.
    4. Fallback to scipy.optimize.fsolve if the optimization approach fails.
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s hard limit

    def time_ok():
        return time.time() - start_time < DEADLINE

    x = np.array(x0, dtype=float).copy()
    n = 2

    def compute_F(val):
        return np.asarray(F(val), dtype=float)

    best_x = x.copy()
    best_norm = np.inf

    # Method 1: Global optimization with local quadratic model
    # We approximate g(x) = ||F(x)||^2 with a local quadratic model.
    # The model is updated using 3 function evaluations per iteration:
    # - One at the current point
    # - Two at points offset along the coordinate axes
    # This gives us estimates of the gradient and Hessian of g.
    
    x_opt = x.copy()
    max_iters = 100
    
    for it in range(max_iters):
        if not time_ok():
            break
        
        # Evaluate F at current point
        Fx = compute_F(x_opt)
        norm_Fx = np.linalg.norm(Fx)
        
        if norm_Fx < best_norm:
            best_norm = norm_Fx
            best_x = x_opt.copy()
        
        if norm_Fx <= 1e-10:
            break
        
        # Estimate gradient and Hessian of g(x) = ||F(x)||^2
        # g(x) = Fx^T @ Fx
        # grad_g = 2 * J^T @ Fx
        # Hess_g = 2 * (J^T @ J + sum(Fx_i * Hessian_F_i))
        # For simplicity, we approximate Hessian as J^T @ J (ignoring second-order terms)
        # This is the Gauss-Newton approximation, which is positive semi-definite
        
        # Use central differences to estimate Jacobian J (5 calls total, but we reuse Fx)
        h = 1e-7 * max(1.0, np.linalg.norm(x_opt))
        J = np.zeros((2, 2))
        
        # We already have Fx, so we need 4 more calls for central differences
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_plus = compute_F(x_opt + e)
            Fx_minus = compute_F(x_opt - e)
            J[:, j] = (Fx_plus - Fx_minus) / (2 * h)
        
        # Compute gradient of g: grad_g = 2 * J^T @ Fx
        grad_g = 2.0 * (J.T @ Fx)
        
        # Compute Gauss-Newton Hessian approximation: H = 2 * J^T @ J
        H = 2.0 * (J.T @ J)
        
        # Regularize Hessian to ensure positive definiteness
        # Add a small multiple of identity if H is singular or ill-conditioned
        try:
            eigvals = np.linalg.eigvalsh(H)
            min_eig = np.min(eigvals)
            if min_eig < 1e-10:
                H = H + (1e-10 - min_eig + 1e-12) * np.eye(2)
        except:
            H = H + 1e-10 * np.eye(2)
        
        # Solve H @ dx = -grad_g for the Newton step
        try:
            dx = np.linalg.solve(H, -grad_g)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.lstsq(H, -grad_g, rcond=None)[0]
            except:
                break
        
        # Line search with step halving
        alpha = 1.0
        accepted = False
        for ls_iter in range(30):
            x_trial = x_opt + alpha * dx
            Fx_trial = compute_F(x_trial)
            norm_Fx_trial = np.linalg.norm(Fx_trial)
            
            if norm_Fx_trial < norm_Fx:
                x_opt = x_trial
                accepted = True
                break
            else:
                alpha *= 0.5
        
        if not accepted:
            break

    # Check if optimization converged
    if best_norm <= 1e-8:
        return best_x

    # Method 2: SciPy fsolve fallback
    if time_ok():
        try:
            start_point = best_x
            sol, info, ier, msg = so.fsolve(
                lambda x: np.asarray(F(x), dtype=float), 
                start_point, 
                full_output=True, 
                maxfev=400, 
                xtol=1e-10
            )
            if ier == 1:
                norm_sol = np.linalg.norm(compute_F(sol))
                if norm_sol < best_norm:
                    best_norm = norm_sol
                    best_x = sol.copy()
                if best_norm <= 1e-8:
                    return best_x
        except Exception:
            pass

    # Method 3: Random restarts with simple Newton
    if best_norm > 1e-8 and time_ok():
        for attempt in range(10):
            if not time_ok():
                break
            x_try = best_x + np.random.randn(2) * 0.5
            x_cur = x_try.copy()
            
            for it in range(30):
                if not time_ok():
                    break
                Fx = compute_F(x_cur)
                norm_Fx = np.linalg.norm(Fx)
                if norm_Fx < best_norm:
                    best_norm = norm_Fx
                    best_x = x_cur.copy()
                if norm_Fx <= 1e-10:
                    break
                
                h = 1e-7 * max(1.0, np.linalg.norm(x_cur))
                Fx0 = Fx
                e1 = np.array([h, 0.0])
                e2 = np.array([0.0, h])
                J = np.zeros((2, 2))
                J[:, 0] = (compute_F(x_cur + e1) - Fx0) / h
                J[:, 1] = (compute_F(x_cur + e2) - Fx0) / h
                
                try:
                    dx = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    break
                
                alpha = 1.0
                x_new = x_cur + alpha * dx
                Fx_new = compute_F(x_new)
                if np.linalg.norm(Fx_new) < norm_Fx:
                    x_cur = x_new
                else:
                    alpha = 0.5
                    x_new = x_cur + alpha * dx
                    Fx_new = compute_F(x_new)
                    if np.linalg.norm(Fx_new) < norm_Fx:
                        x_cur = x_new
                    else:
                        break

    # Final check
    if np.all(np.isfinite(best_x)):
        return best_x
    return np.array(x0, dtype=float)

# EVOLVE-BLOCK-END
