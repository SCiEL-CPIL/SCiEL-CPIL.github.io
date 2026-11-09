import numpy as np
import time
import sys
from scipy.optimize import minimize

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

        # Strategy: Hybrid of Global Coverage (Sobol) and Local Refinement (Nelder-Mead)
        # 1. Generate a Sobol sequence for the entire budget to ensure space-filling coverage.
        # 2. Interleave: Evaluate a Sobol point, then do a quick local refinement on the current best.
        # 3. This avoids getting stuck in one local minimum (like pure NM) and provides
        #    better early progress (AOCC) than pure global search.

        # Generate Sobol sequence points
        # scipy.stats.qmc.Sobol
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
            # Generate enough points. We will use them in order.
            # We need to map [0,1]^d to [-5,5]^d
            sobol_points = sampler.random(1000)
            sobol_points = sobol_points * 10.0 - 5.0
        except ImportError:
            # Fallback to random if qmc is not available (unlikely in modern scipy)
            sobol_points = self._rng.uniform(-5, 5, (1000, self.dim))

        current_x = np.zeros(self.dim)
        current_f = np.inf
        sobol_idx = 0
        total_budget = self.budget
        
        # We will allocate budget in chunks:
        # - Global exploration: Evaluate next Sobol point
        # - Local exploitation: Run Nelder-Mead for a few iterations
        # 
        # Let's do:
        # - First 50 calls: Pure global exploration (Sobol) to get a good initial best.
        # - Then: Alternate between 1 Sobol point and 1 local refinement cycle.
        #   But Nelder-Mead consumes many calls. 
        #   Better: 
        #   1. Evaluate Sobol point.
        #   2. If improved, run Nelder-Mead from there for ~20 calls.
        #   3. Repeat.
        
        # Phase 1: Global Exploration
        # Use first 50 calls to find a decent starting point
        for i in range(min(50, total_budget)):
            if self._call_count >= total_budget or time.time() >= self._deadline:
                break
            x = sobol_points[sobol_idx]
            sobol_idx += 1
            f = safe_call(x)
            if f < current_f:
                current_x = x.copy()
                current_f = f
        
        # Phase 2: Hybrid Exploration/Exploitation with AOCC-optimized budget allocation
        # Key insight: NM with high maxiter can burn many calls without improving best,
        # which hurts AOCC. Use shorter NM cycles and more frequent global moves.
        
        last_improve_call = self._call_count
        
        while self._call_count < total_budget - 30 and time.time() < self._deadline:
            # Get next global point
            if sobol_idx >= len(sobol_points):
                x = self._rng.uniform(-5, 5, self.dim)
            else:
                x = sobol_points[sobol_idx]
                sobol_idx += 1
            
            f = safe_call(x)
            
            should_refine = False
            refine_x = self._best_x
            
            if f < self._best_f:
                should_refine = True
                refine_x = x
                last_improve_call = self._call_count
            elif self._call_count - last_improve_call > 8:
                # Stagnation: try refining current best
                should_refine = True
                refine_x = self._best_x
                last_improve_call = self._call_count

            if should_refine and self._call_count < total_budget - 15:
                try:
                    # Short NM: maxiter=5 (~10-15 calls) is more likely to improve
                    # per call than maxiter=20, and leaves budget for more Sobol points
                    res = minimize(
                        safe_call,
                        refine_x,
                        method='Nelder-Mead',
                        options={'maxiter': 5, 'xatol': 1e-3, 'fatol': 1e-3, 'adaptive': True}
                    )
                except Exception:
                    pass
        
        # Final Polish: use remaining budget for high-precision refinement
        if self._call_count < total_budget and time.time() < self._deadline:
            try:
                res = minimize(
                    safe_call,
                    self._best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 40, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
                )
            except Exception:
                pass

        return self._best_x, self._best_f
