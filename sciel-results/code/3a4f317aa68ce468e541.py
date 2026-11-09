"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer for BBOB-style Schwefel f20 on [-5, 5]^5.
    
    Strategy:
      - Diverse initial points to find a good basin quickly.
      - (1+1)-ES with 1/5 success rule for fast, high-frequency updates (maximizes AOCC).
      - Stagnation detection with random restarts to escape local minima.
      - Final polish pass with scipy L-BFGS-B to refine the best solution.
      - Deadline guard to ensure we return within 330s wall-clock.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin
        
        rng = np.random.default_rng(42)
        
        # Initial points: origin, random, and structured points for diversity
        init_points = [np.zeros(dim)]
        for _ in range(6):
            init_points.append(rng.uniform(-5, 5, dim))
        # Add points at boundaries and midpoints
        init_points.append(np.full(dim, 4.0))
        init_points.append(np.full(dim, -4.0))
        init_points.append(np.full(dim, 2.0))
        init_points.append(np.full(dim, -2.0))
        
        # Evaluate initial points
        best_x = None
        best_f = np.inf
        for x0 in init_points:
            if time.time() > deadline or budget <= 0:
                break
            x0 = np.clip(x0, -5, 5)
            fx = func(x0)
            budget -= 1
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            budget -= 1
        
        # (1+1)-ES with stagnation detection
        sigma = 2.0
        sigma_min = 0.005
        sigma_max = 5.0
        stagnation_count = 0
        max_stagnation = 50
        restart_count = 0
        max_restarts = 3
        
        while budget > 0 and time.time() < deadline:
            # Generate offspring
            y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            budget -= 1
            
            if fy < best_f:
                best_x = y
                best_f = fy
                sigma *= 1.2  # Moderate increase on success
                stagnation_count = 0
            else:
                sigma *= 0.85  # Decrease step size on failure
                stagnation_count += 1
            
            sigma = np.clip(sigma, sigma_min, sigma_max)
            
            # Stagnation detection: if no improvement for max_stagnation steps, try a restart
            if stagnation_count >= max_stagnation and restart_count < max_restarts and budget > 30:
                # Try a random restart from a new point
                x_restart = np.clip(rng.uniform(-5, 5, dim), -5, 5)
                f_restart = func(x_restart)
                budget -= 1
                
                if f_restart < best_f:
                    best_x = x_restart
                    best_f = f_restart
                    sigma = 2.0
                    stagnation_count = 0
                    restart_count += 1
                else:
                    # No improvement from restart, increase step size to explore more
                    sigma = np.clip(sigma * 2.0, sigma_min, sigma_max)
                    stagnation_count = 0
                    restart_count += 1
        
        # Final polish pass with scipy L-BFGS-B
        # Only if we have enough budget and time left
        if budget > 10 and time.time() < deadline - 5:
            try:
                # Use remaining budget for L-BFGS-B
                # L-BFGS-B can use multiple function evaluations per iteration
                # We limit maxiter to be safe, but let it use the budget naturally via the wrapper
                
                calls_used_in_scipy = [0]
                
                def wrapped_func(x):
                    nonlocal budget
                    if budget <= 0 or time.time() > deadline:
                        return best_f  # Return current best if out of budget
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    budget -= 1
                    calls_used_in_scipy[0] += 1
                    return fx

                # Limit iterations to avoid getting stuck, but allow enough for convergence
                # 30 iterations is usually enough for 5D problems
                maxit = min(30, max(1, budget // 3))
                
                result = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': maxit, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 20}
                )

                if result.fun < best_f:
                    best_x = np.clip(result.x, -5, 5)
                    best_f = result.fun

            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
