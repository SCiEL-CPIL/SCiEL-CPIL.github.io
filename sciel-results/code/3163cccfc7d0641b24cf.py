"""Multi-objective optimizer for RE21 using stratified sampling + adaptive local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools


class Optimizer:
    """Multi-objective optimizer for RE21 (4 vars, 2 objectives)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        ref_point = np.array([3000.0, 0.06])

        all_X = []
        all_F = []
        used = 0

        def eval_point(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = problem(x)
            f = np.array(f, dtype=np.float64)
            all_X.append(x.copy())
            all_F.append(f.copy())
            used += 1
            return f

        def time_ok():
            return time.time() < deadline

        # Phase 1: Stratified grid sampling (3^4 = 81 points)
        n_per_dim = 3
        levels = [np.linspace(lo[d], hi[d], n_per_dim) for d in range(self.dim)]
        for combo in itertools.product(*levels):
            if used >= 81 or not time_ok():
                break
            x = np.array(combo, dtype=np.float64)
            eval_point(x)

        # Fill remaining phase-1 budget with random points to reach 100
        while used < 100 and time_ok():
            x = lo + span * np.random.random(self.dim)
            eval_point(x)

        # Phase 2: Deterministic coordinate-descent local search on Pareto front
        if not all_F or not time_ok():
            return np.array(all_X) if all_X else np.array([])

        all_F_arr = np.array(all_F)
        all_X_arr = np.array(all_X)

        # Filter to valid points (better than ref in all objectives)
        valid_mask = np.all(all_F_arr < ref_point, axis=1)
        if np.any(valid_mask):
            valid_F = all_F_arr[valid_mask]
            valid_X = all_X_arr[valid_mask]
        else:
            valid_F = all_F_arr
            valid_X = all_X_arr

        if len(valid_F) == 0:
            return all_X_arr

        nd_mask = self._non_dominated_mask(valid_F)
        nd_F = valid_F[nd_mask]
        nd_X = valid_X[nd_mask]

        # Sort non-dominated points by first objective for diversity
        order = np.argsort(nd_F[:, 0])
        nd_F = nd_F[order]
        nd_X = nd_X[order]

        # Phase 2: Coordinate descent on each non-dominated point
        remaining = self.budget - used
        if remaining > 0 and time_ok():
            # Adaptive step sizes: start moderate, shrink as we converge
            base_step = 0.05 * span
            n_cd_rounds = min(5, max(1, (self.budget - used) // (len(nd_X) * self.dim * 2)))

            for round_num in range(n_cd_rounds):
                if used >= self.budget or not time_ok():
                    break

                # Recompute non-dominated set each round
                all_F_arr = np.array(all_F)
                all_X_arr = np.array(all_X)
                valid_mask = np.all(all_F_arr < ref_point, axis=1)
                if np.any(valid_mask):
                    valid_F = all_F_arr[valid_mask]
                    valid_X = all_X_arr[valid_mask]
                else:
                    valid_F = all_F_arr
                    valid_X = all_X_arr

                if len(valid_F) == 0:
                    break

                nd_mask = self._non_dominated_mask(valid_F)
                nd_F = valid_F[nd_mask]
                nd_X = valid_X[nd_mask]

                # Sort by first objective
                order = np.argsort(nd_F[:, 0])
                nd_F = nd_F[order]
                nd_X = nd_X[order]

                # Shrink step size each round
                step_scale = base_step * (0.5 ** round_num)

                for i in range(len(nd_X)):
                    if used >= self.budget or not time_ok():
                        break

                    x_center = nd_X[i].copy()

                    # Coordinate descent: try +/- along each axis
                    for d in range(self.dim):
                        if used >= self.budget or not time_ok():
                            break

                        for direction in [-1, 1]:
                            if used >= self.budget or not time_ok():
                                break

                            x_new = x_center.copy()
                            x_new[d] += direction * step_scale[d]
                            x_new = np.clip(x_new, lo, hi)

                            f_new = eval_point(x_new)

                            # Check if this point is non-dominated among current Pareto front
                            # If it improves hypervolume contribution, keep it as new center
                            all_F_arr = np.array(all_F)
                            all_X_arr = np.array(all_X)
                            valid_mask = np.all(all_F_arr < ref_point, axis=1)
                            if np.any(valid_mask):
                                valid_F = all_F_arr[valid_mask]
                                valid_X = all_X_arr[valid_mask]
                            else:
                                valid_F = all_F_arr
                                valid_X = all_X_arr

                            if len(valid_F) > 0:
                                # Check if new point is non-dominated
                                is_dominated = False
                                for j in range(len(valid_F)):
                                    if np.all(valid_F[j] <= f_new) and np.any(valid_F[j] < f_new):
                                        is_dominated = True
                                        break

                                if not is_dominated:
                                    # New point is on the Pareto front - update center
                                    x_center = x_new.copy()
                                else:
                                    # Dominated, revert to previous center
                                    pass
                            else:
                                x_center = x_new.copy()

        # Phase 3: Final polish - very fine coordinate search on top contributors
        if used < self.budget and time_ok():
            all_F_arr = np.array(all_F)
            all_X_arr = np.array(all_X)
            valid_mask = np.all(all_F_arr < ref_point, axis=1)
            if np.any(valid_mask):
                valid_F = all_F_arr[valid_mask]
                valid_X = all_X_arr[valid_mask]
            else:
                valid_F = all_F_arr
                valid_X = all_X_arr

            if len(valid_F) > 0:
                nd_mask = self._non_dominated_mask(valid_F)
                nd_F = valid_F[nd_mask]
                nd_X = valid_X[nd_mask]

                # Compute contribution proxy: distance to reference point
                distances = np.sum((ref_point - nd_F) ** 2, axis=1)
                top_k = min(5, len(nd_X))
                top_indices = np.argsort(distances)[:top_k]

                fine_step = 0.01 * span

                for idx in top_indices:
                    if used >= self.budget or not time_ok():
                        break
                    x_center = nd_X[idx].copy()

                    # Fine coordinate descent
                    for d in range(self.dim):
                        if used >= self.budget or not time_ok():
                            break

                        for direction in [-1, 1]:
                            if used >= self.budget or not time_ok():
                                break

                            x_new = x_center.copy()
                            x_new[d] += direction * fine_step[d]
                            x_new = np.clip(x_new, lo, hi)

                            f_new = eval_point(x_new)

                            # Check if non-dominated
                            all_F_arr = np.array(all_F)
                            all_X_arr = np.array(all_X)
                            valid_mask = np.all(all_F_arr < ref_point, axis=1)
                            if np.any(valid_mask):
                                valid_F = all_F_arr[valid_mask]
                                valid_X = all_X_arr[valid_mask]
                            else:
                                valid_F = all_F_arr
                                valid_X = all_X_arr

                            if len(valid_F) > 0:
                                is_dominated = False
                                for j in range(len(valid_F)):
                                    if np.all(valid_F[j] <= f_new) and np.any(valid_F[j] < f_new):
                                        is_dominated = True
                                        break

                                if not is_dominated:
                                    x_center = x_new.copy()

        # Return all evaluated points (evaluator records all objective vectors)
        return np.array(all_X) if all_X else np.array([])

    def _non_dominated_mask(self, F):
        """Return a boolean mask for non-dominated points in F (minimization)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)

        nd_mask = np.ones(n, dtype=bool)
        for i in range(n):
            if not nd_mask[i]:
                continue
            for j in range(n):
                if i == j or not nd_mask[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    nd_mask[i] = False
                    break
        return nd_mask
# EVOLVE-BLOCK-END
