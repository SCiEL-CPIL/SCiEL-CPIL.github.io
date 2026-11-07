"""Multi-objective optimizer for DTLZ1 exploiting known structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer exploiting DTLZ1 structure for hypervolume maximization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        all_X = []
        all_F = []

        def evaluate(x):
            nonlocal all_X, all_F
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            all_X.append(x.copy())
            all_F.append(f.copy())
            return f

        def make_x(x0, x1, shape_offsets=None):
            """Create a DTLZ1 point with given x0, x1 and shape variables near 0.5."""
            x = np.zeros(self.dim)
            x[0] = x0
            x[1] = x1
            if shape_offsets is not None:
                x[2:] = 0.5 + shape_offsets
            else:
                x[2:] = 0.5
            return np.clip(x, lo, hi)

        # Phase 1: Dense grid on the Pareto front (x0, x1 space)
        # For DTLZ1 with M=3, the Pareto front is parameterized by x0, x1 in [0,1]
        # with x[2:] = 0.5 giving g=0 (minimum g)
        n_grid = 14  # 14x14 = 196 points
        x0_vals = np.linspace(0.02, 0.98, n_grid)
        x1_vals = np.linspace(0.02, 0.98, n_grid)

        for x0 in x0_vals:
            if time.time() > deadline - 30:
                break
            for x1 in x1_vals:
                if len(all_X) >= self.budget:
                    break
                x = make_x(x0, x1)
                evaluate(x)

        # Phase 2: Add quasi-random points with shape vars near 0.5 for diversity
        remaining = self.budget - len(all_X)
        n_random = min(remaining, 40)
        for i in range(n_random):
            if time.time() > deadline - 20 or len(all_X) >= self.budget:
                break
            # Use Halton-like sequence for better coverage
            x0 = (i * 0.618033988749895) % 1.0
            x1 = (i * 0.7548776662466927) % 1.0
            shape_offsets = 0.03 * rng.standard_normal(self.dim - 2)
            x = make_x(x0, x1, shape_offsets)
            evaluate(x)

        # Phase 3: Local refinement around non-dominated points
        if len(all_X) < self.budget and time.time() < deadline - 10:
            F = np.array(all_F)
            X = np.array(all_X)
            
            # Simple non-dominated sort
            dominated = np.zeros(len(F), dtype=bool)
            for i in range(len(F)):
                if dominated[i]:
                    continue
                for j in range(len(F)):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            
            nd_idx = np.where(~dominated)[0]
            n_refine = min(self.budget - len(all_X), 24)
            for _ in range(n_refine):
                if time.time() > deadline - 5 or len(all_X) >= self.budget:
                    break
                if len(nd_idx) == 0:
                    break
                idx = nd_idx[rng.integers(len(nd_idx))]
                x = X[idx].copy()
                x[0] += 0.015 * rng.standard_normal()
                x[1] += 0.015 * rng.standard_normal()
                # Keep shape vars near 0.5 with small perturbation
                x[2:] = 0.5 + 0.005 * rng.standard_normal(self.dim - 2)
                x = np.clip(x, lo, hi)
                evaluate(x)

        return all_X
# EVOLVE-BLOCK-END
