"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+lambda)-ES with time guard for bent cigar optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Initial point near center
        x_best = np.zeros(n)
        fx_best = func(x_best)
        budget_used = 1
        sigma = 3.0
        lambda_pop = 4
        
        # Track step size adaptation
        successes = 0
        total_trials = 0
        target_success_rate = 1.0 / (3 * n)  # ~1/15 for n=5
        
        for iteration in range(budget - 1):
            if time.time() > deadline:
                break
            if budget_used >= budget:
                break
            
            # Generate lambda offspring
            candidates = []
            for _ in range(lambda_pop):
                y = np.clip(x_best + sigma * rng.standard_normal(n), -5, 5)
                fy = func(y)
                budget_used += 1
                if budget_used >= budget:
                    break
                candidates.append((fy, y))
                total_trials += 1
            
            if not candidates:
                break
            
            # Select best offspring
            candidates.sort(key=lambda t: t[0])
            fy_best, y_best = candidates[0]
            
            if fy_best < fx_best:
                x_best, fx_best = y_best, fy_best
                successes += 1
                # Increase sigma slightly on success
                sigma *= 1.1
            else:
                # Decrease sigma on failure
                sigma *= 0.92
            
            # Adaptive step size based on success rate over recent trials
            if total_trials >= 10:
                recent_rate = successes / total_trials
                if recent_rate > target_success_rate * 1.5:
                    sigma *= 1.05
                elif recent_rate < target_success_rate * 0.5:
                    sigma *= 0.95
            
            # Reset success counter periodically
            if total_trials % 20 == 0:
                successes = 0
                total_trials = 0
            
            # Clamp sigma to reasonable range
            sigma = np.clip(sigma, 0.01, 10.0)
        
        return x_best, fx_best
# EVOLVE-BLOCK-END
