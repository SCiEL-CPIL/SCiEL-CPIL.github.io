"""Multi-objective optimizer for RE21 using a direct grid + adaptive local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools


class Optimizer:
    """Direct multi-objective optimizer for RE21 (4 vars, 2 objectives)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 330s kill

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

        def is_valid(f):
            return np.all(f < ref_point)

        # Phase 1: Structured exploration with 3^4 = 81 grid points + 19 stratified random
        levels = [np.linspace(lo[d], hi[d], 3) for d in range(self.dim)]
        for combo in itertools.product(*levels):
            if used >= 81 or time.time() > deadline:
                break
            x = np.array(combo, dtype=np.float64)
            eval_point(x)

        # Fill remaining exploration budget with stratified random (Latin-hypercube-like)
        n_fill = 19
        for i in range(n_fill):
            if used >= 100 or time.time() > deadline:
                break
            # Stratified: divide each dim into n_fill bins, pick one bin per dim
            x = np.zeros(self.dim)
            for d in range(self.dim):
                bin_idx = (i + d) % n_fill
                frac_lo = bin_idx / n_fill
                frac_hi = (bin_idx + 1) / n_fill
                x[d] = lo[d] + (frac_lo + np.random.random() * (frac_hi - frac_lo)) * span[d]
            eval_point(x)

        if not all_F:
            return np.array([])

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
            return all_X_arr

        nd_mask = self._non_dominated_mask(valid_F)
        nd_F = valid_F[nd_mask]
        nd_X = valid_X[nd_mask]

        # Phase 2: Adaptive local search around non-dominated points
        # Use remaining budget (~100) for refinement
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline:
            # Sort ND points by hypervolume contribution (approximate by distance to ref)
            # For 2 objectives, sort by sum of normalized distances to ref
            norm_d = (ref_point[0] - nd_F[:, 0]) / ref_point[0] + (ref_point[1] - nd_F[:, 1]) / ref_point[1]
            order = np.argsort(-norm_d)  # Best first
            nd_X_sorted = nd_X[order]
            nd_F_sorted = nd_F[order]

            # Adaptive step: start moderate, shrink if no improvement
            base_scale = 0.08
            min_scale = 0.005
            max_iters_per_point = 12

            for i in range(len(nd_X_sorted)):
                if used >= self.budget or time.time() > deadline:
                    break

                x_best = nd_X_sorted[i].copy()
                f_best = nd_F_sorted[i].copy()
                scale = base_scale
                n_improved = 0

                for it in range(max_iters_per_point):
                    if used >= self.budget or time.time() > deadline:
                        break

                    # Try Gaussian perturbation
                    x_try = x_best + np.random.normal(0, 1, self.dim) * scale * span
                    x_try = np.clip(x_try, lo, hi)
                    f_try = eval_point(x_try)

                    # Check if this improves the local best (dominates or is non-dominated and better in HV)
                    if self._is_better_local(f_try, f_best, ref_point):
                        x_best = x_try.copy()
                        f_best = f_try.copy()
                        n_improved += 1
                        # Keep scale or slightly increase
                        scale = min(scale * 1.2, base_scale)
                    else:
                        # Shrink step
                        scale *= 0.7
                        if scale < min_scale:
                            break

                # After local search, also try a few axis-aligned moves for final polish
                if used < self.budget and time.time() < deadline:
                    for d in range(self.dim):
                        if used >= self.budget or time.time() > deadline:
                            break
                        for direction in [-1, 1]:
                            if used >= self.budget or time.time() > deadline:
                                break
                            x_try = x_best.copy()
                            step = 0.02 * span[d]
                            x_try[d] = x_best[d] + direction * step
                            x_try = np.clip(x_try, lo, hi)
                            f_try = eval_point(x_try)
                            if self._is_better_local(f_try, f_best, ref_point):
                                x_best = x_try.copy()
                                f_best = f_try.copy()

        # Final: recompute non-dominated set from all evaluations
        if not all_F:
            return np.array([])

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
            return all_X_arr

        nd_mask = self._non_dominated_mask(valid_F)
        nd_F = valid_F[nd_mask]
        nd_X = valid_X[nd_mask]

        return nd_X

    def _is_better_local(self, f_new, f_old, ref_point):
        """Check if f_new is better than f_old for local search purposes.
        Returns True if f_new dominates f_old, or if both are valid and f_new has
        better hypervolume contribution (for 2D, simpler check)."""
        # f_new dominates f_old
        if np.all(f_new <= f_old) and np.any(f_new < f_old):
            return True
        # f_old dominates f_new
        if np.all(f_old <= f_new) and np.any(f_old < f_new):
            return False
        # Both non-dominated: compare hypervolume contribution
        # For 2 objectives with reference point, a point contributes more HV if it's
        # closer to the ideal in at least one objective without being worse in the other
        # Simple heuristic: lower sum of (f - ideal) where ideal is min of the two
        # Better: check if f_new would add more HV than f_old given ref_point
        # For 2D: HV contribution of point p with ref r is (r0-p0)*(r1-p1) if both < r
        if np.all(f_new < ref_point) and np.all(f_old < ref_point):
            hv_new = (ref_point[0] - f_new[0]) * (ref_point[1] - f_new[1])
            hv_old = (ref_point[0] - f_old[0]) * (ref_point[1] - f_old[1])
            return hv_new > hv_old
        elif np.all(f_new < ref_point):
            return True
        elif np.all(f_old < ref_point):
            return False
        return False

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
