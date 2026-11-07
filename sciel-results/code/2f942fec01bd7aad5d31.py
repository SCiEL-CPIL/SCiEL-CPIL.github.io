import numpy as np
import time

class Optimizer:
    """
    Derivative-free Trust-Region BFGS (DF-TBFGS) with Random Restarts.
    Uses forward finite differences for gradients.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        
        n = self.dim
        budget = self.budget
        
        best_x = np.zeros(n)
        best_f = float('inf')
        calls_made = 0
        
        # Trust Region & BFGS State
        delta = 1.0
        delta_max = 10.0
        delta_min = 1e-6
        
        x_cur = np.zeros(n)
        f_cur = None
        grad_cur = None
        H_inv = np.eye(n)
        
        eps_fd = 1e-5
        iters_no_improve = 0
        max_iter_no_improve = 10
        
        def estimate_gradient_forward(x, f_val):
            nonlocal calls_made
            grad = np.zeros(n)
            for i in range(n):
                if calls_made >= budget:
                    return grad, calls_made
                x_plus = x.copy()
                x_plus[i] += eps_fd
                calls_made += 1
                f_plus = func(x_plus)
                grad[i] = (f_plus - f_val) / eps_fd
            return grad, calls_made

        def update_bfgs(x_old, x_new, g_old, g_new, H_inv_old):
            s = x_new - x_old
            y = g_new - g_old
            s_dot_y = np.dot(s, y)
            if s_dot_y <= 1e-12:
                return H_inv_old
            H_inv_s = H_inv_old @ s
            y_dot_H_inv_y = np.dot(y, H_inv_old @ y)
            if y_dot_H_inv_y <= 1e-12:
                return H_inv_old
            term1 = np.outer(s, s) / s_dot_y
            term2 = np.outer(H_inv_s, H_inv_s) / y_dot_H_inv_y
            H_inv_new = H_inv_old + term1 - term2
            return (H_inv_new + H_inv_new.T) / 2

        def solve_subproblem(grad, H_inv, delta):
            s_newton = -H_inv @ grad
            norm_s = np.linalg.norm(s_newton)
            if norm_s <= delta:
                return s_newton
            return delta * s_newton / norm_s

        # Initial Eval
        f_cur = func(x_cur)
        calls_made += 1
        best_f = f_cur
        best_x = x_cur.copy()
        
        # Initial Grad
        grad_cur, calls_made = estimate_gradient_forward(x_cur, f_cur)
        
        while calls_made < budget and time.time() < deadline:
            
            # 1. Step
            s = solve_subproblem(grad_cur, H_inv, delta)
            
            # 2. Eval
            if calls_made >= budget: break
            x_new = x_cur + s
            f_new = func(x_new)
            calls_made += 1
            
            # 3. Check Improvement
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
                iters_no_improve = 0
            else:
                iters_no_improve += 1
                
            # 4. Predicted Reduction & Ratio
            # Using heuristic for predicted reduction
            norm_grad = np.linalg.norm(grad_cur)
            if np.linalg.norm(s) < delta * 0.99:
                pred_reduction = 0.5 * np.dot(grad_cur, H_inv @ grad_cur)
            else:
                pred_reduction = -np.dot(grad_cur, s)
                
            if pred_reduction < 1e-12:
                pred_reduction = 1e-12
                
            rho = (f_cur - f_new) / pred_reduction
            
            # 5. Update Delta
            if rho > 0.75:
                delta = min(2 * delta, delta_max)
            elif rho < 0.25:
                delta = max(0.5 * delta, delta_min)
                
            # 6. Update BFGS & State
            if calls_made >= budget: break
            grad_new, calls_made = estimate_gradient_forward(x_new, f_new)
            
            if rho > 0.25:
                H_inv = update_bfgs(x_cur, x_new, grad_cur, grad_new, H_inv)
                x_cur = x_new
                f_cur = f_new
                grad_cur = grad_new
            else:
                # Reject step, do not update state
                pass
            
            # 7. Stagnation / Restart
            if iters_no_improve >= max_iter_no_improve or delta < delta_min:
                # Random Restart
                x_restart = np.random.uniform(-5, 5, n)
                if calls_made >= budget: break
                f_restart = func(x_restart)
                calls_made += 1
                
                if f_restart < best_f:
                    best_f = f_restart
                    best_x = x_restart.copy()
                
                x_cur = x_restart
                f_cur = f_restart
                grad_cur, calls_made = estimate_gradient_forward(x_cur, f_cur)
                H_inv = np.eye(n)
                delta = 1.0
                iters_no_improve = 0

        return best_x, best_f
