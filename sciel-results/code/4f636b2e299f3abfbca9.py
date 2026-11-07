# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-objective DE-style optimizer for RE21 (2 objectives, 4 vars)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hypervolume_2d(self, F, ref):
        """Exact 2D hypervolume of points strictly better than ref (minimization)."""
        pts = F[(F[:, 0] < ref[0]) & (F[:, 1] < ref[1])]
        if len(pts) == 0:
            return 0.0
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        hv = 0.0
        prev_y = ref[1]
        for i in range(len(pts)):
            x = pts[i, 0]
            y = pts[i, 1]
            # contribution: width in x times (prev_y - y) if y < prev_y
            if y < prev_y:
                x_left = (pts[i - 1, 0] if i > 0 else ref[0])
                hv += (x_left - x) * (prev_y - y)
                prev_y = y
        return hv

    def _hv_contributions(self, F, ref):
        """Per-point hypervolume contribution (marginal) for 2D, only for front points."""
        n = len(F)
        contrib = np.zeros(n)
        if n == 0:
            return contrib
        order = np.argsort(F[:, 0])
        sorted_idx = order
        # build sorted arrays
        sx = F[sorted_idx, 0]
        sy = F[sorted_idx, 1]
        # For 2D, contribution of point i (sorted by x asc) is:
        # (x_{i-1} - x_i) * (y_{i-1} - y_i) if y_i < y_{i-1} else 0, with x_{-1}=ref0
        prev_x = ref[0]
        prev_y = ref[1]
        for pos in range(n):
            x = sx[pos]
            y = sy[pos]
            if y < prev_y:
                contrib[sorted_idx[pos]] = (prev_x - x) * (prev_y - y)
                prev_x = x
                prev_y = y
            else:
                contrib[sorted_idx[pos]] = 0.0
        return contrib

    def _non_dominated_indices(self, F):
        """Return boolean mask of non-dominated points (minimization)."""
        n = len(F)
        mask = np.ones(n, dtype=bool)
        for i in range(n):
            if not mask[i]:
                continue
            # does any j dominate i?
            dom = (F <= F[i]) & np.any(F < F[i], axis=1)
            dom[i] = False
            if np.any(dom):
                mask[i] = False
        return mask

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 290.0  # safety margin before 300s

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        ref = np.array([3000.0, 0.06])

        budget = self.budget
        dim = self.dim

        # --- Initialization: reference-point-aligned deterministic grid ---
        # ref = [3000, 0.06] is highly asymmetric: obj1 dominates HV contribution.
        # Place grid points that saturate the asymmetric objective space immediately.
        import itertools

        # Strategy: bias the grid toward low values in the direction where
        # the reference point is far (obj1: ref=3000, so low obj1 is good).
        # Use levels that cluster near the lower end to cover the high-value region.
        # For 4D with budget 200, use a mixed approach:
        # - 2^4 = 16 corner-like points at asymmetric levels
        # - Additional points along the most influential dimensions

        # Asymmetric levels: more resolution near low values (where HV gain is high)
        levels = [0.05, 0.25, 0.5, 0.75, 0.95]

        # Build a structured grid: for each dim, pick from levels
        # Use a subset: 3 levels per dim = 3^4 = 81 points (within budget)
        subset_levels = [0.1, 0.5, 0.9]
        grid = []
        for combo in itertools.product(subset_levels, repeat=dim):
            grid.append(combo)

        # Add the extreme-low corner (most important for HV with ref=[3000, 0.06])
        grid.append([0.02] * dim)
        # Add center
        grid.append([0.5] * dim)
        # Add a few points biased toward low obj1 region
        for _ in range(8):
            pt = [0.05 + 0.3 * np.random.default_rng(0).random() for _ in range(dim)]
            grid.append(pt)

        rng = np.random.default_rng(0)
        X = lo + span * np.array(grid, dtype=float)
        X = np.clip(X, lo, hi)

        # Add random points to reach init population
        init_pop = min(50, budget // 4)
        while len(X) < init_pop:
            X = np.vstack([X, lo + span * rng.random(dim)])
        X = X[:init_pop]
        F = np.array([problem(x) for x in X], dtype=float)
        used = len(X)

        # Record all evaluated points for final hypervolume
        all_F = F.copy()

        # --- Main loop ---
        # Population size for DE
        pop_size = max(10, min(30, budget // 10))
        total_steps = budget - used
        step_count = 0

        while used < budget:
            if time.time() >= deadline:
                break

            # Dynamically shrinking mutation factor:
            # Start with larger exploration, shrink to exploit high-value region
            progress = step_count / max(1, total_steps)
            # F_scale shrinks from 0.8 to 0.15 as we progress
            F_scale = 0.8 - 0.65 * progress
            # Add slight noise to prevent stagnation
            F_scale *= (1.0 + 0.1 * rng.standard_normal())
            F_scale = np.clip(F_scale, 0.05, 1.0)

            # Identify non-dominated front
            nd_mask = self._non_dominated_indices(F)
            nd_idx = np.where(nd_mask)[0]
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(F))

            # HV contributions on the front
            front_F = F[nd_idx]
            contrib = self._hv_contributions(front_F, ref)
            total = contrib.sum()
            if total <= 1e-15:
                weights = np.ones(len(nd_idx)) / len(nd_idx)
            else:
                weights = contrib / total

            # Select parent(s) via DE-style
            n_nd = len(nd_idx)
            if n_nd < 3:
                # Fallback: random selection
                parent_idx = nd_idx[rng.integers(n_nd)]
                x1 = X[parent_idx]
                x2 = X[parent_idx]
                x3 = X[parent_idx]
            else:
                # Weighted selection for a, b, c
                a = nd_idx[rng.choice(n_nd, p=weights)]
                b = nd_idx[rng.choice(n_nd, p=weights)]
                c = nd_idx[rng.choice(n_nd, p=weights)]
                # Ensure distinct if possible
                if n_nd >= 3:
                    while a == b:
                        b = nd_idx[rng.choice(n_nd, p=weights)]
                    while c == a or c == b:
                        c = nd_idx[rng.choice(n_nd, p=weights)]
                x1, x2, x3 = X[a], X[b], X[c]

            # DE mutation: v = x1 + F*(x2 - x3)
            v = x1 + F_scale * (x2 - x3)
            # Perturbation also shrinks with progress (less exploration later)
            perturb_scale = 0.15 * (1.0 - 0.7 * progress)
            v = v + perturb_scale * span * rng.standard_normal(dim)
            v = np.clip(v, lo, hi)

            fv = problem(v)
            used += 1
            step_count += 1
            all_F = np.vstack([all_F, fv])

            # Replacement: worst point by HV contribution (or random if all zero)
            # Compute contributions for current population front
            cur_nd = self._non_dominated_indices(F)
            cur_nd_idx = np.where(cur_nd)[0]
            if len(cur_nd_idx) > 0:
                cur_contrib = self._hv_contributions(F[cur_nd_idx], ref)
                # Map back: worst = point with smallest contribution among front
                worst_local = np.argmin(cur_contrib)
                worst_idx = cur_nd_idx[worst_local]
            else:
                worst_idx = rng.integers(len(F))

            # Only replace if child is better or non-dominated
            # Simple rule: replace worst if child dominates worst OR child has better HV
            if (fv[0] < F[worst_idx, 0] and fv[1] < F[worst_idx, 1]) or \
               (fv[0] <= F[worst_idx, 0] and fv[1] <= F[worst_idx, 1] and
                (fv[0] < F[worst_idx, 0] or fv[1] < F[worst_idx, 1])):
                X[worst_idx] = v
                F[worst_idx] = fv
            else:
                # Replace anyway to maintain diversity if budget is running low
                if used > budget * 0.75:
                    X[worst_idx] = v
                    F[worst_idx] = fv

        # Final polish: if we have time left, do a few local refinements
        # around the best HV-contributing point
        remaining_time = deadline - time.time()
        if remaining_time > 2.0 and used < budget:
            # Find the point with highest HV contribution
            cur_nd = self._non_dominated_indices(F)
            cur_nd_idx = np.where(cur_nd)[0]
            if len(cur_nd_idx) > 0:
                cur_contrib = self._hv_contributions(F[cur_nd_idx], ref)
                best_local = np.argmax(cur_contrib)
                best_idx = cur_nd_idx[best_local]
                best_x = X[best_idx].copy()
                best_f = F[best_idx].copy()

                # Local polish with very small steps
                polish_budget = min(budget - used, 15)
                for _ in range(polish_budget):
                    if time.time() >= deadline:
                        break
                    v = best_x + 0.02 * span * rng.standard_normal(dim)
                    v = np.clip(v, lo, hi)
                    fv = problem(v)
                    used += 1
                    all_F = np.vstack([all_F, fv])
                    if fv[0] < best_f[0] and fv[1] < best_f[1]:
                        best_x = v
                        best_f = fv
                    elif fv[0] < best_f[0] or fv[1] < best_f[1]:
                        # Update best_x if one objective improved
                        if fv[0] < best_f[0]:
                            best_x = v.copy()
                            best_f = fv.copy()
                        elif fv[1] < best_f[1]:
                            best_x = v.copy()
                            best_f = fv.copy()

        return X
# EVOLVE-BLOCK-END
