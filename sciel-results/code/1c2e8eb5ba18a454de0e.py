"""Multi-objective optimizer on WFG4 using LHS + DE-based archive approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Latin Hypercube + Differential Evolution archive-based optimizer for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(12345)

        MAX_ARCHIVE = 100
        INIT_POP = 50  # Sobol sample size

        # Sobol sequence sampling for initial population
        # Use a deterministic Sobol sequence for low-discrepancy coverage
        sobol = qmc.Sobol(d=self.dim, scramble=False)
        # Use power-of-2 number of points for best Sobol properties
        n_sobol = 64  # 2^6
        sobol_points = sobol.random(n_sobol)
        # Map to [0, 1) and then to bounds
        X_init = lo + (hi - lo) * sobol_points
        # Only use the first INIT_POP points for evaluation
        X_init = X_init[:INIT_POP]
        F_init = np.array([problem(x) for x in X_init])
        evaluations = INIT_POP

        # Build initial archive
        archive_X = []
        archive_F = []

        def is_dominated(f_new, archive_F_list):
            """Check if f_new is dominated by any in archive_F_list."""
            for f in archive_F_list:
                if np.all(f <= f_new) and np.any(f < f_new):
                    return True
            return False

        def remove_dominated(new_f, archive_X_list, archive_F_list):
            """Remove points in archive dominated by new_f."""
            new_archive_X = []
            new_archive_F = []
            for i in range(len(archive_F_list)):
                f_i = archive_F_list[i]
                if np.all(new_f <= f_i) and np.any(new_f < f_i):
                    continue
                new_archive_X.append(archive_X_list[i])
                new_archive_F.append(f_i)
            return new_archive_X, new_archive_F

        def crowding_distance(F_arr, indices):
            """Compute crowding distance for a set of indices in F_arr."""
            if len(indices) <= 2:
                return {i: float('inf') for i in indices}
            distances = {i: 0.0 for i in indices}
            for m in range(self.n_obj):
                values = [F_arr[i][m] for i in indices]
                min_val = min(values)
                max_val = max(values)
                if max_val - min_val < 1e-12:
                    continue
                sorted_idx = sorted(indices, key=lambda i: F_arr[i][m])
                distances[sorted_idx[0]] = float('inf')
                distances[sorted_idx[-1]] = float('inf')
                for k in range(1, len(sorted_idx) - 1):
                    distances[sorted_idx[k]] += (
                        F_arr[sorted_idx[k + 1]][m] - F_arr[sorted_idx[k - 1]][m]
                    ) / (max_val - min_val)
            return distances

        def prune_archive(archive_X_list, archive_F_list, max_size):
            """Prune archive to max_size using crowding distance."""
            if len(archive_X_list) <= max_size:
                return archive_X_list, archive_F_list
            F_arr = np.array(archive_F_list)
            indices = list(range(len(archive_F_list)))
            cd = crowding_distance(F_arr, indices)
            sorted_indices = sorted(indices, key=lambda i: cd[i], reverse=True)
            kept_indices = sorted_indices[:max_size]
            kept_X = [archive_X_list[i] for i in kept_indices]
            kept_F = [archive_F_list[i] for i in kept_indices]
            return kept_X, kept_F

        # Populate initial archive
        for i in range(INIT_POP):
            if not is_dominated(F_init[i], archive_F):
                archive_X, archive_F = remove_dominated(F_init[i], archive_X, archive_F)
                archive_X.append(X_init[i])
                archive_F.append(F_init[i])

        archive_X, archive_F = prune_archive(archive_X, archive_F, MAX_ARCHIVE)

        # Differential Evolution parameters (proven from parent)
        CR = 0.9  # Crossover rate
        F_DE = 0.8  # Differential weight
        F_MUT = 0.1  # Mutation scaling for Gaussian perturbation

        # Main DE loop
        while evaluations < self.budget:
            if time.time() > deadline:
                break

            # Select 3 distinct parents from archive
            n_archive = len(archive_X)
            if n_archive < 3:
                x_new = lo + (hi - lo) * rng.random(self.dim)
            else:
                # DE mutation: x_trial = x_r1 + F * (x_r2 - x_r3)
                idxs = rng.choice(n_archive, size=3, replace=False)
                x_r1 = archive_X[idxs[0]]
                x_r2 = archive_X[idxs[1]]
                x_r3 = archive_X[idxs[2]]

                # Differential mutation
                x_m = x_r1 + F_DE * (x_r2 - x_r3)

                # Binary crossover with trial vector
                x_trial = x_r1.copy()
                for d in range(self.dim):
                    if rng.random() < CR or d == rng.integers(self.dim):
                        x_trial[d] = x_m[d]

                # Add small Gaussian perturbation for exploration
                x_trial = x_trial + F_MUT * (hi - lo) * rng.standard_normal(self.dim)

                x_new = x_trial

            # Clip to bounds
            x_new = np.clip(x_new, lo, hi)

            # Evaluate
            f_new = problem(x_new)
            evaluations += 1

            # Update archive
            if not is_dominated(f_new, archive_F):
                archive_X, archive_F = remove_dominated(f_new, archive_X, archive_F)
                archive_X.append(x_new)
                archive_F.append(f_new)

                if len(archive_X) > MAX_ARCHIVE:
                    archive_X, archive_F = prune_archive(archive_X, archive_F, MAX_ARCHIVE)

        # Final polish pass: local refinement around top archive points
        if evaluations < self.budget and time.time() < deadline:
            # Sort archive by sum of objectives (proxy for quality) to pick best points
            if len(archive_X) > 0:
                F_arr = np.array(archive_F)
                # Use normalized sum as quality indicator
                ranges = F_arr.max(axis=0) - F_arr.min(axis=0)
                ranges[ranges < 1e-12] = 1.0
                norm_F = (F_arr - F_arr.min(axis=0)) / ranges
                quality_scores = norm_F.sum(axis=1)
                sorted_idx = np.argsort(quality_scores)
                # Take top 8 for local refinement (increased from 5)
                n_polish = min(8, len(archive_X))
                polish_indices = sorted_idx[:n_polish]

                # Use remaining budget for local perturbations around these points
                remaining = self.budget - evaluations
                steps_per_point = max(1, remaining // n_polish)

                for pi in polish_indices:
                    if evaluations >= self.budget or time.time() > deadline:
                        break
                    base_x = archive_X[pi]
                    for _ in range(steps_per_point):
                        if evaluations >= self.budget or time.time() > deadline:
                            break
                        # Adaptive step size: start larger, decrease over iterations
                        step_size = 0.03 * (hi - lo)
                        x_trial = base_x + step_size * rng.standard_normal(self.dim)
                        x_trial = np.clip(x_trial, lo, hi)
                        f_trial = problem(x_trial)
                        evaluations += 1

                        if not is_dominated(f_trial, archive_F):
                            archive_X, archive_F = remove_dominated(f_trial, archive_X, archive_F)
                            archive_X.append(x_trial)
                            archive_F.append(f_trial)

                            if len(archive_X) > MAX_ARCHIVE:
                                archive_X, archive_F = prune_archive(archive_X, archive_F, MAX_ARCHIVE)

        return archive_X
# EVOLVE-BLOCK-END
