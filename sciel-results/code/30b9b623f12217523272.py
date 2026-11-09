import numpy as np
import time
from scipy.stats import qmc

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
            try:
                val = float(func(x))
            except Exception:
                return 1e10
            if not np.isfinite(val):
                return 1e10
            if val < self._best_f:
                self._best_f = val
                self._best_x = x.copy()
            return val

        # Phase 1: Global exploration with Sobol sequence
        n_sobol = 20
        try:
            sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
            sobol_points = sampler.random(n_sobol)
            sobol_points = sobol_points * 10.0 - 5.0
        except Exception:
            sobol_points = self._rng.uniform(-5, 5, (n_sobol, self.dim))

        # Evaluate Sobol points
        for i in range(n_sobol):
            if self._call_count >= self.budget or time.time() >= self._deadline:
                break
            x = sobol_points[i]
            safe_call(x)

        if self._call_count >= self.budget or time.time() >= self._deadline:
            return self._best_x, self._best_f

        # Phase 2: Multi-start (1+1)-ES refinement
        # Use remaining budget for multiple ES runs from best starting points
        # Strategy: allocate budget among several restarts, each running a (1+1)-ES
        
        # Determine number of restarts based on remaining budget
        remaining = self.budget_remaining
        # Each ES run needs at least ~20 evaluations to be useful
        min_es_budget = 20
        n_restarts = max(1, min(5, remaining // min_es_budget))
        
        # Get top starting points from Sobol evaluation
        # We need to re-evaluate or track them; for simplicity, use best found + perturbations
        start_points = [self._best_x.copy()]
        
        # Add diverse starting points by perturbing the best point
        for i in range(1, n_restarts):
            if time.time() >= self._deadline:
                break
            # Create diverse starts by adding random offset
            offset = self._rng.uniform(-1.5, 1.5, self.dim)
            start_point = np.clip(self._best_x + offset, -5, 5)
            start_points.append(start_point)
        
        # Run (1+1)-ES from each starting point
        for start_idx, start_x in enumerate(start_points):
            if self._call_count >= self.budget or time.time() >= self._deadline:
                break
            
            # Allocate budget for this ES run
            runs_left = n_restarts - start_idx
            es_budget = min(remaining // runs_left, self.budget_remaining)
            if es_budget < 5:
                break
            
            # (1+1)-ES parameters
            sigma = 1.0  # initial step size
            best_x = start_x.copy()
            best_f = safe_call(best_x)
            no_improve = 0
            max_no_improve = 10
            
            for _ in range(es_budget):
                if self._call_count >= self.budget or time.time() >= self._deadline:
                    break
                
                # Generate candidate
                candidate = best_x + sigma * self._rng.standard_normal(self.dim)
                candidate = np.clip(candidate, -5, 5)
                
                f_candidate = safe_call(candidate)
                
                if f_candidate < best_f:
                    # Improvement
                    best_f = f_candidate
                    best_x = candidate.copy()
                    no_improve = 0
                    # Increase step size slightly on success (optional)
                    sigma = min(sigma * 1.2, 3.0)
                else:
                    # No improvement
                    no_improve += 1
                    # Decrease step size
                    sigma = max(sigma * 0.8, 0.01)
                
                # Early stop if stuck
                if no_improve >= max_no_improve:
                    break
            
            remaining = self.budget_remaining
        
        return self._best_x, self._best_f
