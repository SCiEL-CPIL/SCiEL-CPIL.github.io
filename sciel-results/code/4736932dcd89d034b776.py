"""Multi-start Differential Evolution with adaptive (1+1)-ES polish for BBOB Weierstrass f16."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution

class Optimizer:
    """Multi-start DE with adaptive (1+1)-ES for the BBOB Weierstrass (f16) task."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0
        
        n_calls = [0]
        stop_flag = [False]

        def safe_func(x):
            if stop_flag[0] or n_calls[0] >= budget or time.time() > deadline:
                stop_flag[0] = True
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            n_calls[0] += 1
            try:
                return float(func(x))
            except:
                return np.inf

        bounds = [(-5.0, 5.0) for _ in range(dim)]
        best_x = np.zeros(dim)
        best_f = np.inf

        # Phase 1: Multi-run Differential Evolution for global exploration
        # Use multiple runs with decreasing budgets to find good basins
        # Early runs get more budget for thorough exploration
        runs = [250, 150, 60]
        
        for i, run_budget in enumerate(runs):
            if n_calls[0] >= budget or time.time() > deadline:
                break
            
            stop_flag[0] = False
            start_calls = n_calls[0]
            
            try:
                res = differential_evolution(
                    safe_func,
                    bounds,
                    maxiter=100,
                    popsize=15,
                    tol=1e-10,
                    seed=42 + i,
                    init='latinhypercube',
                    callback=lambda xk, convergence: True if (
                        stop_flag[0] or 
                        n_calls[0] - start_calls >= run_budget or 
                        time.time() > deadline
                    ) else False
                )
                
                if res.fun < best_f and np.isfinite(res.fun):
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 2: Adaptive (1+1)-ES with 1/5-rule step-size adaptation
        # This exploits local smoothness for rapid convergence in a good basin
        remaining = budget - n_calls[0]
        if remaining > 5 and time.time() < deadline:
            # Initialize step size based on problem scale
            # For Weierstrass, the function has features at multiple scales
            # Start with a moderate step size
            sigma = 0.1
            x = best_x.copy()
            f_x = best_f
            
            # 1/5-rule parameters
            mu = 1  # number of offspring (1+1-ES)
            c_sigma = 1.0 / (5 * dim)  # step-size adaptation rate
            rho = 1.0 / (5 * dim + 1)  # time constant for success rate
            success_rate = 0.5  # target success rate
            log_sigma = np.log(sigma)
            
            # Track for 1/5-rule
            n_success = 0
            n_total = 0
            window_size = 5 * dim + 1
            
            # Run (1+1)-ES until budget or time runs out
            while n_calls[0] < budget and time.time() < deadline and not stop_flag[0]:
                # Generate one offspring
                rng = np.random.default_rng(hash((n_calls[0], i)) % (2**32))
                z = rng.standard_normal(dim)
                x_trial = x + sigma * z
                
                # Evaluate
                f_trial = safe_func(x_trial)
                
                if not np.isfinite(f_trial):
                    break
                    
                # Update best if improved
                if f_trial < f_x:
                    x = x_trial.copy()
                    f_x = f_trial
                    best_x = x.copy()
                    best_f = f_x
                    n_success += 1
                else:
                    # Keep old x
                    pass
                    
                n_total += 1
                
                # Apply 1/5-rule step-size adaptation every window_size steps
                if n_total % window_size == 0:
                    if n_total > 0:
                        actual_success_rate = n_success / n_total
                        # Adapt step size
                        if actual_success_rate > success_rate:
                            sigma *= 1.5  # increase step size
                        else:
                            sigma /= 1.5  # decrease step size
                        # Clamp sigma to reasonable range
                        sigma = np.clip(sigma, 1e-6, 10.0)
                    
                    # Reset counters
                    n_success = 0
                    n_total = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
