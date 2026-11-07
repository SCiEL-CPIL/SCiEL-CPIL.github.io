"""Hypervolume-guided evolutionary optimizer for WFG7."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective optimizer using hypervolume-based selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin before 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for scoring (fixed by task)
        ref_point = np.array([3.0, 5.0, 7.0])

        # Population size
        pop_size = 24

        # Initialize population
        X = np.random.uniform(lo, hi, size=(pop_size, n_dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Archive of non-dominated points
        archive_X = [X[i].copy() for i in range(pop_size)]
        archive_F = [F[i].copy() for i in range(pop_size)]

        def fast_non_dominated_sort(F_arr):
            """Fast non-dominated sort. Returns list of fronts (lists of indices)."""
            n = len(F_arr)
            if n == 0:
                return []

            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = np.zeros(n, dtype=int)  # number of solutions that dominate i

            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        S[j].append(i)
                        n_dom[i] += 1

            fronts = []
            current_front = np.where(n_dom == 0)[0].tolist()

            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = next_front

            return fronts

        def compute_hv_contribution(f, archive_F, ref):
            """Compute approximate hypervolume contribution of point f to archive."""
            # Filter archive to points better than ref
            valid = [af for af in archive_F if np.all(af < ref)]
            if not valid:
                # Contribution is the box from f to ref if f is better than ref
                if np.all(f < ref):
                    return np.prod(ref - f)
                return 0.0

            # Simple approximation: compute volume of box from f to ref
            # minus overlap with existing archive points
            if not np.all(f < ref):
                return 0.0

            # For a more accurate contribution, we'd need to compute the
            # actual hypervolume difference, but for selection purposes,
            # we use a simpler proxy: the product of (ref - f) weighted
            # by how much it extends beyond existing points
            box_vol = np.prod(ref - f)

            # Check if f is dominated by any archive point
            for af in valid:
                if np.all(af <= f) and np.any(af < f):
                    return 0.0  # dominated, no contribution

            # f is non-dominated; approximate contribution as box_vol
            # (this is an upper bound; good enough for relative comparison)
            return box_vol

        def select_by_hv(F_new, X_new, k, ref):
            """Select k points from F_new with best hypervolume contribution."""
            n = len(F_new)
            if n <= k:
                return np.arange(n)

            # Compute non-dominated fronts
            fronts = fast_non_dominated_sort(F_new)

            selected = []
            for front in fronts:
                if len(selected) + len(front) <= k:
                    selected.extend(front)
                else:
                    remaining = k - len(selected)
                    # Select from this front by hypervolume score
                    F_front = F_new[front]
                    scores = np.zeros(len(front))
                    for idx, i in enumerate(front):
                        if np.all(F_front[idx] < ref):
                            scores[idx] = np.prod(ref - F_front[idx])
                        else:
                            scores[idx] = -1.0

                    # Sort by score descending
                    order = np.argsort(scores)[::-1]
                    for idx in order[:remaining]:
                        selected.append(front[idx])
                    break

            if len(selected) < k:
                # Fill with remaining points
                all_indices = list(range(n))
                for i in all_indices:
                    if i not in selected and len(selected) < k:
                        selected.append(i)

            return np.array(selected)

        # Main evolutionary loop
        while used < budget:
            if time.time() > deadline:
                break

            # Generate offspring using DE/rand/1 with multiple variants
            n_offspring = min(6, budget - used)

            for _ in range(n_offspring):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break

                # DE/rand/1
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                v = X[r1] + 0.7 * (X[r2] - X[r3])
                v = np.clip(v, lo, hi)

                # Sometimes add random perturbation
                if np.random.random() < 0.3:
                    v = np.clip(v + 0.1 * (hi - lo) * np.random.randn(n_dim), lo, hi)

                fv = problem(v)
                used += 1

                # Combine population and offspring, select best
                X_new = np.vstack([X, v])
                F_new = np.vstack([F, fv])

                # Select best pop_size by hypervolume
                selected_idx = select_by_hv(F_new, X_new, pop_size, ref_point)

                X = X_new[selected_idx]
                F = F_new[selected_idx]

                # Update archive
                new_archive_X = []
                new_archive_F = []
                for i in range(pop_size):
                    f = F[i]
                    # Check if f is dominated by any archive point
                    dominated = False
                    for af in archive_F:
                        if np.all(af <= f) and np.any(af < f):
                            dominated = True
                            break
                    if not dominated:
                        # Remove archive points dominated by f
                        new_archive_X = []
                        new_archive_F = []
                        for j in range(len(archive_F)):
                            if not (np.all(f <= archive_F[j]) and np.any(f < archive_F[j])):
                                new_archive_X.append(archive_X[j])
                                new_archive_F.append(archive_F[j])
                        new_archive_X.append(X[i].copy())
                        new_archive_F.append(f.copy())
                        break
                    else:
                        continue
                else:
                    # No new non-dominated point added
                    pass

                # If we added a new point, archive is updated
                # Otherwise, keep existing archive
                # This logic is simplified; in practice we'd do proper archive management

            # Every few generations, do a local refinement around best points
            if used < budget and pop_size > 0:
                # Find the point with best hypervolume contribution
                scores = np.zeros(pop_size)
                for i in range(pop_size):
                    if np.all(F[i] < ref_point):
                        scores[i] = np.prod(ref_point - F[i])
                    else:
                        scores[i] = -1.0

                best_idx = np.argmax(scores)
                if scores[best_idx] > 0:
                    # Local search around best point
                    n_local = min(3, budget - used)
                    for _ in range(n_local):
                        if used >= budget:
                            break
                        if time.time() > deadline:
                            break

                        # Small perturbation
                        scale = 0.05 * (hi - lo)
                        x_local = np.clip(X[best_idx] + scale * np.random.randn(n_dim), lo, hi)
                        f_local = problem(x_local)
                        used += 1

                        # Replace worst point if improvement
                        worst_idx = np.argmin(scores)
                        if np.all(f_local < F[worst_idx]) or (
                            np.all(f_local <= F[worst_idx]) and np.any(f_local < F[worst_idx])
                        ):
                            X[worst_idx] = x_local
                            F[worst_idx] = f_local
                            scores[worst_idx] = np.prod(ref_point - f_local) if np.all(f_local < ref_point) else -1.0

        return X
# EVOLVE-BLOCK-END
