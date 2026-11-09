"""Multi-objective optimizer on RE37 using Sobol sampling + local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using Sobol quasi-random sampling + local Gaussian refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dominates(self, f1, f2):
        """Check if f1 dominates f2 (minimization)."""
        return np.all(f1 <= f2) and np.any(f1 < f2)

    def _is_non_dominated(self, f_new, F_list):
        """Check if f_new is non-dominated by any point in F_list."""
        for f in F_list:
            if self._dominates(f, f_new):
                return False
        return True

    def _sobol_sequence(self, n, dim, lo, hi):
        """Generate n quasi-random points using a simple Halton/Sobol-like sequence."""
        # Use Halton sequence for simplicity (deterministic, no external deps needed)
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        points = np.zeros((n, dim))
        for i in range(1, n + 1):
            for d in range(dim):
                # Use prime bases for each dimension
                bases = [2, 3, 5, 7, 11, 13, 17, 19]
                base = bases[d % len(bases)]
                points[i - 1, d] = lo[d] + halton(i, base) * (hi[d] - lo[d])
        return points

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        F_list = []
        X_list = []
        evals_used = 0

        def add_point(x, f):
            nonlocal evals_used, F_list, X_list
            if self._is_non_dominated(f, F_list):
                F_list = [f_old for f_old in F_list if not self._dominates(f, f_old)]
                F_list.append(f)
                X_list.append(x)

        # Phase 1: Sobol/Halton quasi-random sampling for uniform coverage
        n_init = 80
        X_sobol = self._sobol_sequence(n_init, self.dim, lo, hi)

        for x in X_sobol:
            f = problem(x)
            evals_used += 1
            add_point(x, f)

        if time.time() > deadline or evals_used >= self.budget:
            return np.array(F_list)

        # Phase 2: Local Gaussian refinement around non-dominated points
        # Use remaining budget to perturb each non-dominated point
        remaining = self.budget - evals_used
        if remaining <= 0:
            return np.array(F_list)

        # Adaptive sigma: start larger, shrink as we refine
        sigma_start = 0.15
        sigma_end = 0.02
        n_points = max(len(X_list), 1)
        evals_per_point = max(remaining // n_points, 1)
        # Cap evals per point to avoid over-focusing on one point
        evals_per_point = min(evals_per_point, 15)

        # Sort non-dominated points by their "potential" - use sum of objectives as proxy
        # Points with lower sum are closer to ideal, prioritize them
        if len(X_list) > 0:
            F_arr = np.array(F_list)
            sums = np.sum(F_arr, axis=1)
            order = np.argsort(sums)
            X_sorted = [X_list[i] for i in order]
            F_sorted = [F_list[i] for i in order]
        else:
            X_sorted = []
            F_sorted = []

        # Interleaved refinement: cycle through points, allocating budget
        point_idx = 0
        evals_for_point = 0
        total_points = len(X_sorted)

        while evals_used < self.budget and time.time() < deadline - 5.0:
            if total_points == 0:
                break

            if point_idx >= total_points:
                point_idx = 0
                # Shrink sigma as we cycle
                sigma_start = max(sigma_start * 0.7, sigma_end)

            x_base = X_sorted[point_idx]
            
            # Adaptive sigma based on progress
            progress = evals_used / self.budget
            sigma = sigma_start * (1 - progress) + sigma_end * progress
            
            x_trial = x_base + np.random.normal(0, sigma, self.dim)
            x_trial = np.clip(x_trial, lo, hi)
            
            f_trial = problem(x_trial)
            evals_used += 1
            evals_for_point += 1

            was_added = self._is_non_dominated(f_trial, F_list)
            if was_added:
                F_list = [f_old for f_old in F_list if not self._dominates(f_trial, f_old)]
                F_list.append(f_trial)
                X_list.append(x_trial)
                # Re-sort if needed (simplified: just track new point)
                # For efficiency, we'll re-sort occasionally
                if evals_used % 20 == 0 and len(X_list) > 0:
                    F_arr = np.array(F_list)
                    sums = np.sum(F_arr, axis=1)
                    order = np.argsort(sums)
                    X_sorted = [X_list[i] for i in order]
                    F_sorted = [F_list[i] for i in order]
                    total_points = len(X_sorted)
                    # Reset point_idx if it's out of bounds
                    if point_idx >= total_points:
                        point_idx = 0

            # Move to next point after a few evaluations on current
            if evals_for_point >= evals_per_point:
                point_idx += 1
                evals_for_point = 0
                total_points = len(X_sorted)  # May have changed
                if point_idx >= total_points:
                    point_idx = 0

        return np.array(F_list)
# EVOLVE-BLOCK-END
