"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """1+1-ES with adaptive step size, followed by Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin: 300s limit, keep 20s buffer

        dim = self.dim
        budget = self.budget
        calls_made = 0
        best_f = np.inf
        best_x = None

        def call_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            fx = float(func(x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def check_deadline():
            return (time.time() - start_time) > deadline

        # --- Phase 1: 1+1-ES ---
        # 1+1-ES is extremely budget-efficient: 1 evaluation per step.
        # This allows ~500 steps vs ~100-125 for CMA-ES.
        # Key for AOCC: early progress is critical. 1+1-ES can make small, steady improvements.
        
        rng = np.random.default_rng(42)
        
        # Initialize at origin (common heuristic for BBOB)
        x = np.zeros(dim)
        fx = call_func(x)
        
        # Adaptive sigma
        # Start with a moderate step size. 
        # The domain is [-5, 5]^5, so distance to corners is ~11.
        # A step of 1.0 is reasonable for initial exploration.
        sigma = 1.0
        
        # 1+1-ES parameters
        # Success rate control: target success rate p_target ~ 0.2 (typical for 1+1-ES)
        # If success rate > p_target, increase sigma; else decrease.
        # Simple update: sigma *= exp(0.1 * (success_rate - p_target))
        # Or use the standard (1,1)-ES rule with adaptation.
        
        # Let's use a robust adaptive scheme:
        # Keep a window of last 10 steps to estimate success rate.
        history = []
        window_size = 10
        p_target = 0.2
        
        # We have budget 500.
        # Reserve 100 calls for Nelder-Mead refinement.
        # So use 400 calls for 1+1-ES.
        es_budget = int(budget * 0.8)  # 400 calls
        nm_budget = budget - es_budget  # 100 calls
        
        for _ in range(es_budget):
            if calls_made >= budget or check_deadline():
                break
                
            # Generate candidate
            z = rng.standard_normal(dim)
            x_trial = x + sigma * z
            x_trial = np.clip(x_trial, -5, 5)
            
            f_trial = call_func(x_trial)
            
            # Update
            if f_trial < fx:
                # Success
                x = x_trial
                fx = f_trial
                history.append(1)
            else:
                # Failure
                history.append(0)
                
            # Adapt sigma based on recent success rate
            if len(history) >= window_size:
                recent = history[-window_size:]
                success_rate = np.mean(recent)
                # Simple exponential adaptation
                # If success rate is too high, increase sigma; too low, decrease.
                # Rate: 0.1 is a reasonable learning rate for sigma adaptation.
                sigma *= np.exp(0.1 * (success_rate - p_target))
                sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Clear history to avoid memory growth (optional, but good practice)
            if len(history) > 100:
                history = history[-window_size:]

        # --- Phase 2: Nelder-Mead Refinement ---
        # Use the remaining budget to polish the best point found by 1+1-ES.
        # Nelder-Mead is efficient for local minima and uses ~dim+1 evaluations per step.
        
        if best_x is not None and calls_made < budget and not check_deadline():
            from scipy.optimize import minimize as sp_minimize
            
            x0 = best_x.copy()
            remaining = budget - calls_made
            # Nelder-Mead uses dim+1 points per iteration.
            # Reserve a few extra calls for safety.
            max_iter_nm = max(10, remaining // (dim + 1) - 2)
            
            def objective(x):
                nonlocal calls_made
                if calls_made >= budget:
                    return np.inf
                calls_made += 1
                return float(func(x))
            
            try:
                res = sp_minimize(objective, x0, method='Nelder-Mead',
                                  options={'maxiter': max_iter_nm, 
                                           'xatol': 1e-10, 
                                           'fatol': 1e-10, 
                                           'maxfev': remaining})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5).copy()
            except Exception:
                pass

        return best_x, best_f

# EVOLVE-BLOCK-END
