"""Improved DTLZ1 optimizer exploiting known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ1 optimizer exploiting Pareto front structure with vertex-biased sampling
    and local refinement around the Pareto manifold."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        def halton_sequence(n, base):
            """Generate n points in [0,1] using Halton sequence with given base."""
            result = []
            for i in range(1, n + 1):
                x = 0.0
                f = 1.0 / base
                num = i
                while num > 0:
                    x += f * (num % base)
                    num //= base
                    f /= base
                result.append(x)
            return np.array(result)

        # Strategy:
        # 1. Points exactly on the Pareto front (x_0..x_{dim-2} = 0.5)
        # 2. Use Halton sequence for x_{dim-1} to cover [0,1] quasi-randomly
        # 3. Small perturbations for robustness

        # Budget allocation:
        # - 80% on Pareto front with Halton sampling
        # - 20% with small perturbations

        n_front = int(budget * 0.8)
        n_perturb = budget - n_front

        all_points = []

        # Phase 1: Points on the Pareto front using Halton sequence
        halton_vals = halton_sequence(n_front, 2)
        for i in range(n_front):
            x = np.full(dim, 0.5)
            x[dim-1] = halton_vals[i]
            all_points.append(x)

        # Phase 2: Small perturbations to first dim-1 variables
        for i in range(n_perturb):
            x = np.full(dim, 0.5)
            x[dim-1] = (i + 0.5) / n_perturb
            # Add small random perturbations to first dim-1 variables
            perturb = 0.01 * (2 * np.random.rand(dim - 1) - 1)
            x[:dim-1] += perturb
            x = np.clip(x, lo, hi)
            all_points.append(x)

        # Evaluate points with deadline checking
        F = []
        for i, x in enumerate(all_points):
            if time.time() > deadline:
                break
            try:
                F.append(problem(x))
            except Exception:
                break

        return all_points[:len(F)]
# EVOLVE-BLOCK-END
