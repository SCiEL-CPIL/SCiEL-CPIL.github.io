"""Optimized solver for BBOB f13 (Sharp ridge) using Sobol multi-start + L-BFGS-B."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Multi-start local search with Sobol initialization and L-BFGS-B refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim

        # Track function evaluations to stay within budget
        self._eval_count = 0
        self._best_f = np.inf
        self._best_x = None

        def tracked_func(x):
            if self._eval_count >= budget:
                return self._best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            self._eval_count += 1
            fx = float(func(x))
            if fx < self._best_f:
                self._best_f = fx
                self._best_x = x.copy()
            return fx

        # Phase 1: Sobol sampling for broad coverage
        # Use a good number of initial points for early progress in AOCC
        n_sobol = min(48, max(12, budget // 10))
        halton = qmc.Sobol(d=dim, scramble=True, seed=42)
        m = max(1, int(np.ceil(np.log2(n_sobol))))
        sobol_samples = halton.random(2 ** m)[:n_sobol]
        x_sobol = -5.0 + 10.0 * sobol_samples

        all_points = []
        all_values = []

        for x0 in x_sobol:
            if self._eval_count >= budget:
                break
            fx = tracked_func(x0)
            all_points.append(x0.copy())
            all_values.append(fx)

        # Phase 2: Sequential refinement of top candidates
        # Sort by objective value and refine the best few
        order = np.argsort(all_values)
        
        # Allocate remaining budget across refinements
        remaining = budget - self._eval_count
        # Refine top k candidates, each getting a share of remaining budget
        n_refine = min(len(order), max(1, remaining // 15))
        
        # Try a few top candidates
        for idx in order[:n_refine]:
            if self._eval_count >= budget:
                break
            
            x_start = all_points[idx].copy()
            
            # Use L-BFGS-B with tight tolerances for sharp ridge
            maxiter = max(1, min(80, (budget - self._eval_count) // 2))
            
            try:
                result = minimize(
                    tracked_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                
                if result.fun < self._best_f:
                    self._best_f = result.fun
                    self._best_x = result.x.copy()
                    
            except Exception:
                pass

        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
