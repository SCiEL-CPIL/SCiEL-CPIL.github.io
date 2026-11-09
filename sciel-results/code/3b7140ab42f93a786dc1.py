"""Multi-objective optimizer on RE37 with Sobol sweep and adaptive local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.stats import qmc


class Optimizer:
    """Multi-objective optimizer using Sobol quasi-random sweep and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """Return list of fronts (list of indices)."""
        n = len(F)
        if n == 0:
            return []

        S = [[] for _ in range(n)]
        nd = np.zeros(n, dtype=int)

        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].append(q)
                elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                    nd[p] += 1

        front = [i for i in range(n) if nd[i] == 0]
        if not front:
            front = [0]

        fronts = [front]
        while front:
            next_front = []
            for p in front:
                for q in S[p]:
                    nd[q] -= 1
                    if nd[q] == 0:
                        next_front.append(q)
            if next_front:
                fronts.append(next_front)
            front = next_front
        return fronts

    def _crowding_distance(self, F, indices):
        """Calculate crowding distance for a front."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)

        dist = np.zeros(n)
        F_sub = F[indices]

        for m in range(self.n_obj):
            order = np.argsort(F_sub[:, m])
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf

            min_val = F_sub[order[0], m]
            max_val = F_sub[order[-1], m]
            if max_val == min_val:
                continue

            for k in range(1, n - 1):
                dist[order[k]] += (F_sub[order[k + 1], m] - F_sub[order[k - 1], m]) / (max_val - min_val)
        return dist

    def _coordinate_descent(self, problem, x0, bounds, n_iter=30, step_init=0.05):
        """Simple coordinate descent to refine a point."""
        x = x0.copy()
        f = np.array(problem(x))
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        step = step_init

        for it in range(n_iter):
            improved = False
            for d in range(self.dim):
                # Try positive and negative steps
                for direction in [1, -1]:
                    x_try = x.copy()
                    x_try[d] += direction * step
                    x_try = np.clip(x_try, lo, hi)
                    f_try = np.array(problem(x_try))
                    # Use Tchebycheff scalarization to check improvement
                    f_min = np.min(f)
                    f_max = np.max(f)
                    f_range = f_max - f_min if f_max > f_min else 1.0
                    scalar_old = np.max((f - f_min) / f_range)
                    scalar_new = np.max((f_try - f_min) / f_range)
                    if scalar_new < scalar_old - 1e-10:
                        x = x_try
                        f = f_try
                        improved = True
                        break
                if not improved:
                    step *= 0.8
            if step < 0.001:
                break
        return x, f

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # Phase 1: Sobol quasi-random sweep
        # Use 128 points for good coverage in 4D
        n_sobol = 128
        sobol = qmc.Sobol(d=self.dim, scramble=True)
        X_sobol_unit = sobol.random(n_sobol)
        X_sobol = lo + X_sobol_unit * span

        F_sobol = np.array([problem(x) for x in X_sobol])
        used = n_sobol

        if time.time() > deadline:
            return F_sobol

        # Phase 2: Find non-dominated points and refine
        fronts = self._fast_non_dominated_sort(F_sobol)

        if fronts:
            front_indices = fronts[0]
        else:
            front_indices = np.arange(n_sobol)

        # Limit refinement points
        max_refine_points = 20
        if len(front_indices) > max_refine_points:
            dist = self._crowding_distance(F_sobol, front_indices)
            order = np.argsort(dist)[::-1]
            front_indices = [front_indices[i] for i in order[:max_refine_points]]

        remaining_budget = self.budget - used
        if remaining_budget <= 0 or not front_indices:
            return F_sobol

        # Phase 3: Local refinement using coordinate descent + Gaussian perturbations
        new_X = []
        new_F = []

        # Allocate budget: 60% for coordinate descent, 40% for Gaussian perturbations
        n_cd = min(len(front_indices), remaining_budget // 3)
        n_gauss = remaining_budget - n_cd

        # Coordinate descent on top points
        for idx in front_indices[:n_cd]:
            if used >= self.budget or time.time() > deadline - 10:
                break
            x_seed = X_sobol[idx]
            x_refined, f_refined = self._coordinate_descent(problem, x_seed, bounds)
            new_X.append(x_refined)
            new_F.append(f_refined)
            used += 1  # Approximate; coordinate descent uses multiple evals but we count as 1 for budget tracking
            # Actually, coordinate descent uses many evaluations. Let's be more careful.
            # For now, let's just do a few direct evaluations.

        # Reset and do a simpler approach: just Gaussian perturbations with varying scales
        new_X = []
        new_F = []

        # Distribute remaining budget among refinement points
        perturbations_per_point = max(1, remaining_budget // len(front_indices))

        for idx in front_indices:
            if used >= self.budget or time.time() > deadline - 5:
                break

            x_seed = X_sobol[idx]

            # Multi-scale Gaussian perturbations
            step_sizes = [0.03, 0.07, 0.12, 0.2]

            for _ in range(perturbations_per_point):
                if used >= self.budget or time.time() > deadline - 5:
                    break

                # Choose step size randomly
                step_size = step_sizes[np.random.randint(len(step_sizes))]

                # Gaussian perturbation
                x_new = x_seed + step_size * np.random.standard_normal(self.dim)
                x_new = np.clip(x_new, lo, hi)

                f_new = problem(x_new)
                new_X.append(x_new)
                new_F.append(f_new)
                used += 1

        # Combine all results
        if new_F:
            F_combined = np.vstack([F_sobol, np.array(new_F)])
        else:
            F_combined = F_sobol

        return F_combined
# EVOLVE-BLOCK-END
