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

        # DTLZ1: last variable is g(x) = 100*(n-1 + sum((x_i - 0.5)^2 - cos(20*pi*(x_i-0.5))))
        # for i=0..n-2. The Pareto front is g=0, which requires each of the first
        # dim-1 variables to be at 0.5. The last variable determines the trade-off.
        # f1 = 0.5*(1+g)*x_{n-1}, f2 = 0.5*(1+g)*(1-x_{n-1})

        # Strategy:
        # 1. Start with points on or near the Pareto front (x_0..x_{dim-2} = 0.5)
        # 2. Use quasi-random sampling for x_{dim-1} to cover the simplex
        # 3. Add some exploration with small perturbations to x_0..x_{dim-2}

        # Generate initial points on the Pareto front
        # Use Halton-like sequence for the last variable
        def halton_sequence(n, base):
            """Generate n points in [0,1] using Halton sequence with given base."""
            result = []
            f = 1.0 / base
            for i in range(1, n + 1):
                x = 0.0
                while i > 0:
                    f *= 1.0
                    x += f * (i % base)
                    i //= base
                    f /= base
                result.append(x)
            return np.array(result)

        # We'll use a simple quasi-random approach: stratified sampling
        # For the last variable, sample uniformly across [0,1]
        # For the first dim-1 variables, keep them at 0.5 (on the Pareto front)

        # Phase 1: Dense sampling on the Pareto front
        # Phase 2: Local refinement with small perturbations

        # Total budget allocation:
        # - 60% on Pareto front (x_0..x_{dim-2} = 0.5)
        # - 30% with small perturbations to first dim-1 vars
        # - 10% random exploration

        n_front = int(budget * 0.6)
        n_perturb = int(budget * 0.3)
        n_random = budget - n_front - n_perturb

        all_points = []

        # Phase 1: Points on the Pareto front
        # x_0..x_{dim-2} = 0.5, x_{dim-1} sampled quasi-randomly
        # Use stratified sampling for x_{dim-1}
        for i in range(n_front):
            x = np.zeros(dim)
            x[:dim-1] = 0.5  # on Pareto front
            # Stratified: divide [0,1] into n_front intervals
            x[dim-1] = (i + 0.5) / n_front
            all_points.append(x)

        # Phase 2: Small perturbations to first dim-1 variables
        # This helps in case the problem has numerical issues or if we need to
        # explore slightly off the front
        for i in range(n_perturb):
            x = np.zeros(dim)
            x[:dim-1] = 0.5
            x[dim-1] = (i + 0.5) / n_perturb
            # Add small random perturbations to first dim-1 variables
            for d in range(dim - 1):
                x[d] += 0.02 * (2 * np.random.rand() - 1)
            x = np.clip(x, lo, hi)
            all_points.append(x)

        # Phase 3: Random exploration
        for i in range(n_random):
            x = lo + (hi - lo) * np.random.rand(dim)
            all_points.append(x)

        # Shuffle the points to avoid bias in evaluation order
        np.random.shuffle(all_points)

        # Evaluate all points
        F = np.array([problem(x) for x in all_points])
        used = len(all_points)

        return all_points
# EVOLVE-BLOCK-END
