import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s margin before 300s limit
        
        n = self.dim
        budget = self.budget
        
        # Global state
        best_x = np.zeros(n)
        best_f = np.inf
        eval_count = 0
        
        def safe_func(x):
            nonlocal best_x, best_f, eval_count
            if time.time() > deadline:
                raise TimeoutError
            if eval_count >= budget:
                raise BudgetExceeded
            f_val = func(np.array(x, dtype=np.float64))
            eval_count += 1
            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x, dtype=np.float64)
            return f_val

        # Phase 1: Coarse structured scan to find promising basins
        # For Lunacek bi-Rastrigin, we need to explore both the global and local minima
        rng = np.random.default_rng(42)
        
        # Try origin first
        safe_func(np.zeros(n))
        
        # Structured grid-like sampling for better coverage
        # Use a mix of random and quasi-random (Sobol-like) points
        n_scan = min(40, budget)
        
        for i in range(1, n_scan):
            # Mix of uniform random and scaled Halton-like points
            if i % 3 == 0:
                # Quasi-random: use golden ratio for better distribution
                golden = 0.6180339887498949
                x = np.array([
                    (i * golden + j * golden**2) % 1.0 * 10 - 5
                    for j in range(n)
                ])
            else:
                x = rng.uniform(-5, 5, n)
            safe_func(x)
            
            if eval_count >= budget:
                break

        if eval_count >= budget:
            return best_x, best_f

        # Phase 2: (1+1)-ES with restarts for efficient exploitation
        # This is faster than CMA-ES for dim=5 with 500 budget
        
        current_x = best_x.copy()
        current_f = best_f
        sigma = 0.5  # Initial step size
        
        # Adaptive (1+1)-ES with restarts
        while eval_count < budget and time.time() < deadline:
            # Generate candidate
            z = rng.standard_normal(n)
            candidate = np.clip(current_x + sigma * z, -5, 5)
            
            if eval_count >= budget:
                break
            if time.time() > deadline:
                break
            
            cand_f = safe_func(candidate)
            
            # Accept if better
            if cand_f < current_f - 1e-10:
                current_x = candidate
                current_f = cand_f
                # Slightly increase sigma on success (success rule)
                sigma *= 1.05
            else:
                # Decrease sigma on failure
                sigma *= 0.85
            
            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 0.001, 5.0)
            
            # Restart mechanism: if sigma is very small or no improvement for many steps
            if sigma < 0.005:
                # Restart with a larger sigma from a nearby point
                sigma = 1.0
                # Try a few random restarts
                for _ in range(min(5, budget - eval_count)):
                    x = np.clip(best_x + rng.normal(0, 2.0, n), -5, 5)
                    if eval_count >= budget:
                        break
                    safe_func(x)
                break
        
        # Phase 3: Final polish with smaller steps
        if eval_count < budget and time.time() < deadline:
            sigma = max(0.01, sigma * 0.1)
            polish_steps = min(20, budget - eval_count)
            for _ in range(polish_steps):
                if eval_count >= budget:
                    break
                if time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                candidate = np.clip(best_x + sigma * z, -5, 5)
                cand_f = safe_func(candidate)
                if cand_f < best_f:
                    best_x = candidate
                    best_f = cand_f
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                sigma = np.clip(sigma, 0.001, 0.1)
        
        return best_x, best_f
