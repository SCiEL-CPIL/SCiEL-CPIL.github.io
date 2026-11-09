"""Optimizer for BBOB-style f15 (Rastrigin rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, L_BFGS_B


class Optimizer:
    """Hybrid global-local optimizer: random restarts + L-BFGS-B local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit

        # Budget tracking
        calls = [0]

        def counted_func(x):
            calls[0] += 1
            return float(func(x))

        dim = self.dim
        budget = self.budget
        best_x = None
        best_f = np.inf

        # Phase 1: Random sampling to find a good basin
        # Use a fraction of budget for exploration
        n_explore = min(budget // 3, 150)
        
        rng = np.random.default_rng(42)
        
        for i in range(n_explore):
            if calls[0] >= budget:
                break
            if time.time() > deadline:
                break
                
            # Mix of uniform and slightly perturbed points
            if i == 0:
                x = np.zeros(dim)
            elif i % 3 == 0:
                x = rng.uniform(-4, 4, dim)
            else:
                # Random but biased towards center
                x = rng.uniform(-3, 3, dim)
            
            fx = counted_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Local optimization from best point and several other good points
        # We'll try L-BFGS-B from the best point and a few others
        if best_x is not None:
            candidates = [best_x.copy()]
            
            # Add a few other random points as candidates for local optimization
            # This helps if the first best is in a bad basin
            n_local_starts = min(5, (budget - calls[0]) // 50)
            for _ in range(n_local_starts):
                if calls[0] >= budget:
                    break
                if time.time() > deadline:
                    break
                x = rng.uniform(-4, 4, dim)
                candidates.append(x)

            # Try local optimization from each candidate
            for x0 in candidates:
                if calls[0] >= budget:
                    break
                if time.time() > deadline:
                    break
                
                try:
                    result = minimize(
                        counted_func,
                        x0,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 50, 'maxls': 20}
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass

        # Phase 3: If we have budget left, do a few more random restarts with local polish
        remaining = budget - calls[0]
        if remaining > 20 and best_x is not None:
            n_extra = min(remaining // 40, 3)
            for _ in range(n_extra):
                if calls[0] >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Perturb the best point
                x0 = best_x + rng.normal(0, 0.5, dim)
                x0 = np.clip(x0, -5, 5)
                
                try:
                    result = minimize(
                        counted_func,
                        x0,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 30, 'maxls': 10}
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x.copy()
                except Exception:
                    pass

        # Final: ensure we have a valid result
        if best_x is None:
            # Fallback: use center
            best_x = np.zeros(dim)
            best_f = counted_func(best_x) if calls[0] < budget else 0.0

        return best_x, best_f


# EVOLVE-BLOCK-END
