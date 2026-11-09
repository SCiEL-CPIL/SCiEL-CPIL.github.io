"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start adaptive (1+1)-ES with coarse-to-fine restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def safe_call(x):
            nonlocal calls_used
            if calls_used >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            try:
                val = float(func(np.asarray(x, dtype=np.float64)))
                calls_used += 1
                return val
            except Exception:
                calls_used += 1
                return np.inf

        # Phase 1: Coarse random sampling to find promising regions
        n_coarse = min(30, budget // 10)
        coarse_points = []
        coarse_vals = []
        for _ in range(n_coarse):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            if fx < np.inf:
                coarse_points.append(x)
                coarse_vals.append(fx)

        if not coarse_points:
            # Fallback: try origin
            x0 = np.zeros(dim)
            fx0 = safe_call(x0)
            if fx0 < np.inf:
                best_x = x0
                best_f = fx0
            else:
                best_x = np.zeros(dim)
                best_f = 0.0
            return best_x, best_f

        # Sort by fitness
        order = np.argsort(coarse_vals)
        
        # Select top k starting points
        k = min(5, len(coarse_points))
        top_indices = order[:k]

        # Initialize best
        best_x = coarse_points[top_indices[0]].copy()
        best_f = coarse_vals[top_indices[0]]

        # Phase 2: Local refinement from each top starting point
        # Distribute remaining budget among restarts
        remaining_budget = budget - calls_used
        # Give more budget to better starts
        weights = np.zeros(k)
        for i, idx in enumerate(top_indices):
            # Weight inversely proportional to rank (better gets more)
            weights[i] = 1.0 / (i + 1)
        weights /= weights.sum()

        for i, start_idx in enumerate(top_indices):
            if calls_used >= budget or time.time() > deadline:
                break

            x = coarse_points[start_idx].copy()
            fx = coarse_vals[start_idx]
            sigma = 1.5

            # Allocate budget for this restart
            alloc = int(remaining_budget * weights[i])
            alloc = max(5, min(alloc, budget - calls_used))

            for _ in range(alloc):
                if calls_used >= budget or time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85

                # Early stopping if sigma becomes too small
                if sigma < 1e-6:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
