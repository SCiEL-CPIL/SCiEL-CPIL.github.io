# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-objective DE-style optimizer for RE21 (2 objectives, 4 vars)."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_contributions_2d(self, F, ref):
        """Per-point hypervolume contribution (marginal) for 2D, only for front points."""
        n = len(F)
        contrib = np.zeros(n)
        if n == 0:
            return contrib
        order = np.argsort(F[:, 0])
        sorted_idx = order
        sx = F[sorted_idx, 0]
        sy = F[sorted_idx, 1]
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

        # --- Initialization: stratified grid + random ---
        import itertools
        levels = [0.15, 0.85]
        grid = []
        for combo in itertools.product(levels, repeat=dim):
            grid.append(combo)
        grid.append([0.5] * dim)

        rng = np.random.default_rng(0)
        X = lo + span * np.array(grid)
        X = np.clip(X, lo, hi)

        init_pop = min(40, budget // 5)
        while len(X) < init_pop:
            X = np.vstack([X, lo + span * rng.random(dim)])
        X = X[:init_pop]
        F = np.array([problem(x) for x in X], dtype=float)
        used = len(X)

        # Record all evaluated points for final hypervolume
        all_F = F.copy()

        # --- Main loop ---
        pop_size = max(10, min(30, budget // 10))

        # Adaptive F parameter
        F_param = 0.5
        C_rate = 0.3  # crossover rate

        while used < budget:
            if time.time() >= deadline:
                break

            # Identify non-dominated front
            nd_mask = self._non_dominated_indices(F)
            nd_idx = np.where(nd_mask)[0]
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(F))

            # HV contributions on the front
            front_F = F[nd_idx]
            contrib = self._hv_contributions_2d(front_F, ref)
            total = contrib.sum()
            if total <= 1e-15:
                weights = np.ones(len(nd_idx)) / len(nd_idx)
            else:
                # Use softmax-like weights for smoother selection
                max_c = contrib.max()
                exp_contrib = np.exp((contrib - max_c) / (max_c + 1e-10))
                weights = exp_contrib / exp_contrib.sum()

            # Select parent(s) via DE-style
            n_nd = len(nd_idx)
            if n_nd < 3:
                parent_idx = nd_idx[rng.integers(n_nd)]
                x1 = X[parent_idx]
                x2 = X[parent_idx]
                x3 = X[parent_idx]
            else:
                a = nd_idx[rng.choice(n_nd, p=weights)]
                b = nd_idx[rng.choice(n_nd, p=weights)]
                c = nd_idx[rng.choice(n_nd, p=weights)]
                if n_nd >= 3:
                    while a == b:
                        b = nd_idx[rng.choice(n_nd, p=weights)]
                    while c == a or c == b:
                        c = nd_idx[rng.choice(n_nd, p=weights)]
                x1, x2, x3 = X[a], X[b], X[c]

            # Adaptive mutation scale
            spread = np.max(front_F, axis=0) - np.min(front_F, axis=0)
            F_scale = 0.5
            if np.all(spread < 1e-3):
                F_scale = 1.0

            # DE mutation: v = x1 + F*(x2 - x3)
            v = x1 + F_scale * (x2 - x3)
            # Add small random perturbation
            v = v + 0.1 * span * rng.standard_normal(dim)
            v = np.clip(v, lo, hi)

            fv = problem(v)
            used += 1
            all_F = np.vstack([all_F, fv])

            # Replacement: worst point by HV contribution
            cur_nd = self._non_dominated_indices(F)
            cur_nd_idx = np.where(cur_nd)[0]
            if len(cur_nd_idx) > 0:
                cur_contrib = self._hv_contributions_2d(F[cur_nd_idx], ref)
                worst_local = np.argmin(cur_contrib)
                worst_idx = cur_nd_idx[worst_local]
            else:
                worst_idx = rng.integers(len(F))

            # Replace if child dominates worst or has better HV contribution
            if (fv[0] < F[worst_idx, 0] and fv[1] < F[worst_idx, 1]) or \
               (fv[0] <= F[worst_idx, 0] and fv[1] <= F[worst_idx, 1] and
                (fv[0] < F[worst_idx, 0] or fv[1] < F[worst_idx, 1])):
                X[worst_idx] = v
                F[worst_idx] = fv
            else:
                # Replace anyway to maintain diversity if budget is running low
                if used > budget * 0.8:
                    X[worst_idx] = v
                    F[worst_idx] = fv

        return X
# EVOLVE-BLOCK-END
