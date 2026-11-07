"""Black-box continuous optimizer for the rotated Rosenbrock (f9) task."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Gradient-based optimizer using finite difference gradient estimation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        def time_ok():
            return time.time() < deadline

        # Initial point
        x = rng.uniform(-2, 2, n)
        fx = func(x)
        calls = 1
        best_x = x.copy()
        best_f = fx

        # Parameters
        # Central difference step size
        h = 1e-5
        # Learning rate
        lr = 0.1
        min_lr = 1e-10
        max_lr = 1.0
        # Stagnation detection
        no_improve = 0
        stagnation_threshold = 20
        
        # Gradient estimation helper
        def estimate_gradient(x_curr, f_curr):
            grad = np.zeros(n)
            for i in range(n):
                x_plus = x_curr.copy()
                x_minus = x_curr.copy()
                x_plus[i] += h
                x_minus[i] -= h
                
                # Clip to box
                x_plus = np.clip(x_plus, -5, 5)
                x_minus = np.clip(x_minus, -5, 5)
                
                # If clipping changed the point, we might be at a boundary.
                # For Rosenbrock, the minimum is usually inside [-4,4], so boundaries are less critical
                # but we must handle them to avoid invalid gradient estimates if clipped.
                # A simple robust way: if x_plus[i] == x_curr[i], we are at upper bound.
                # If x_minus[i] == x_curr[i], we are at lower bound.
                # However, the problem states minimizer is in [-4, 4], so we likely won't hit [-5, 5] often.
                
                f_plus = func(x_plus)
                f_minus = func(x_minus)
                calls += 2
                
                # Central difference
                # Note: if clipping occurred, the effective step size might be 0 or smaller.
                # To keep it simple and robust, we use the nominal h. If we hit a wall, 
                # the gradient component might be inaccurate, but we'll rely on the 
                # fact that the optimum is internal.
                grad[i] = (f_plus - f_minus) / (2 * h)
                
                if not time_ok() or calls >= budget:
                    return grad, calls, False
            
            return grad, calls, True

        # Main loop
        while calls < budget and time_ok():
            # 1. Estimate Gradient
            grad, calls, completed = estimate_gradient(x, fx)
            if not completed:
                break
            
            # 2. Check for convergence/stagnation
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-8:
                # Converged or stuck in flat region
                no_improve += 1
            else:
                # 3. Gradient Descent Step with Backtracking
                # Direction: -grad
                direction = -grad
                
                # Normalize direction to have unit norm for consistent step size interpretation
                # Actually, standard GD uses lr * grad. Let's stick to that.
                # But we can scale lr based on grad norm to be robust?
                # Let's try a simple adaptive lr: if we fail, reduce lr. If we succeed, increase slightly?
                
                # Backtracking Line Search
                alpha = lr
                success = False
                max_backtracks = 10
                
                for _ in range(max_backtracks):
                    x_new = x + alpha * direction
                    x_new = np.clip(x_new, -5, 5)
                    f_new = func(x_new)
                    calls += 1
                    
                    if not time_ok() or calls >= budget:
                        break
                    
                    # Armijo condition (simplified): f_new < f_x
                    # Since we don't have a reliable Lipschitz constant, we just require improvement
                    # or a small decrease.
                    if f_new < fx:
                        # Success
                        x = x_new
                        fx = f_new
                        success = True
                        
                        # Update global best
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                            no_improve = 0
                        else:
                            no_improve += 1
                        
                        # Adapt learning rate
                        # If we took a large step, maybe we can increase lr?
                        # If we had to backtrack a lot, decrease lr.
                        backtracks_used = _
                        if backtracks_used > 5:
                            lr = max(lr * 0.5, min_lr)
                        elif backtracks_used == 0 and alpha > 0.1:
                            lr = min(lr * 1.1, max_lr)
                        else:
                            # Keep lr same or slight adjustment
                            pass
                        
                        break
                    else:
                        # Backtrack
                        alpha *= 0.5
                        
                if not time_ok() or calls >= budget:
                    break
                
                if not success:
                    # Failed to find a descent direction with current lr
                    # This suggests we are very close to a minimum or in a bad spot
                    # Reduce lr significantly
                    lr = max(lr * 0.5, min_lr)
                    no_improve += 1

            # Stagnation Restart Logic
            if no_improve >= stagnation_threshold:
                # Perturb the current point
                # Use a larger step to escape local minima (though Rosenbrock has no local minima, 
                # it has narrow valleys, so "stagnation" might mean we are moving too slowly 
                # along the valley or stuck in a flat area)
                
                # Random restart near best found so far
                x = best_x + rng.normal(0, 0.5, n)
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls += 1
                
                if not time_ok() or calls >= budget:
                    break
                
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                
                # Reset parameters
                lr = 0.1
                no_improve = 0
                h = 1e-5 # Reset h, could adapt this too but keep simple

        return best_x, best_f
# EVOLVE-BLOCK-END
