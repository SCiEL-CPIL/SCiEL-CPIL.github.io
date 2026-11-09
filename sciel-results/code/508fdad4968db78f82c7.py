import numpy as np
import time
import sys

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._start_time = None
        self._deadline = None
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        self._start_time = time.time()
        # 330s limit, keep 5s margin
        self._deadline = self._start_time + 325
        self._call_count = 0
        self.budget_remaining = self.budget
        self._best_x = np.zeros(self.dim)
        self._best_f = np.inf
        self._rng = np.random.default_rng(42)

        def safe_call(x):
            if self._call_count >= self.budget or time.time() >= self._deadline:
                return 1e10
            self._call_count += 1
            self.budget_remaining -= 1
            val = func(x)
            if not np.isfinite(val):
                return 1e10
            if val < self._best_f:
                self._best_f = val
                self._best_x = x.copy()
            return val

        # Generate Sobol sequence for global points
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
            sobol_points = sampler.random(500)
            sobol_points = sobol_points * 10.0 - 5.0
        except ImportError:
            sobol_points = self._rng.uniform(-5, 5, (500, self.dim))

        # Phase 1: Global Exploration with Sobol points
        sobol_idx = 0
        
        # Evaluate first 30 Sobol points to find a decent basin
        for i in range(30):
            if self._call_count >= self.budget or time.time() >= self._deadline:
                break
            x = sobol_points[sobol_idx]
            sobol_idx += 1
            f = safe_call(x)

        # Phase 2: (1+1)-ES with Restarts
        # (1+1)-ES:
        # - Step size sigma
        # - If candidate better: accept, sigma *= 1.05
        # - If candidate worse: reject, sigma *= 0.95
        # - Restart if no improvement for K steps
        
        sigma = 1.0
        sigma_min = 1e-6
        sigma_max = 5.0
        stagnation_limit = 30
        stagnation_counter = 0
        
        # We will loop until budget or time is out
        while self._call_count < self.budget and time.time() < self._deadline:
            # Generate candidate
            # Use standard normal distribution
            noise = self._rng.standard_normal(self.dim)
            candidate_x = self._best_x + sigma * noise
            
            # Clip to box [-5, 5]
            candidate_x = np.clip(candidate_x, -5.0, 5.0)
            
            f_cand = safe_call(candidate_x)
            
            if f_cand < self._best_f:
                # Accept
                self._best_f = f_cand
                self._best_x = candidate_x
                stagnation_counter = 0
                # Increase step size slightly to explore
                sigma = min(sigma * 1.05, sigma_max)
            else:
                # Reject
                stagnation_counter += 1
                # Decrease step size
                sigma = max(sigma * 0.95, sigma_min)
                
                # Check for stagnation
                if stagnation_counter >= stagnation_limit:
                    # Restart: Pick a new point from Sobol sequence or random
                    # If Sobol points exhausted, use random
                    if sobol_idx < len(sobol_points):
                        new_x = sobol_points[sobol_idx]
                        sobol_idx += 1
                    else:
                        new_x = self._rng.uniform(-5, 5, self.dim)
                    
                    f_new = safe_call(new_x)
                    
                    # Update best if the new global point is better
                    if f_new < self._best_f:
                        self._best_f = f_new
                        self._best_x = new_x
                        # Reset sigma to a medium value for new basin
                        sigma = 1.0
                        stagnation_counter = 0
                    else:
                        # Keep current best but reset stagnation counter to try again
                        # Maybe increase sigma to jump out?
                        sigma = min(sigma * 2.0, sigma_max)
                        stagnation_counter = 0

        return self._best_x, self._best_f
