import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        rng = np.random.default_rng(0)

        evals_used = 0
        max_evals = self.budget

        def evaluate(x):
            nonlocal evals_used
            if evals_used >= max_evals:
                return None
            x_clipped = np.clip(x, lo, hi)
            f = np.array(problem(x_clipped), dtype=float)
            evals_used += 1
            return x_clipped, f

        def nondominated_indices(F):
            """Vectorized non-dominated front extraction. F is (n, n_obj) array."""
            n = F.shape[0]
            if n == 0:
                return np.array([], dtype=int)
            if n == 1:
                return np.array([0])
            # For each point i, check if any j dominates i
            # j dominates i if F[j] <= F[i] elementwise and F[j] < F[i] in at least one
            # Vectorized: for each i, count how many j dominate i
            # F[j] <= F[i] for all dims: (F <= F[i:i+1]).all(axis=1)
            # F[j] < F[i] in at least one dim: (F < F[i:i+1]).any(axis=1)
            # But we need to exclude j=i itself
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                # Check if any j != i dominates i
                le = (F <= F[i]).all(axis=1)  # F[j] <= F[i] for all dims
                lt = (F < F[i]).any(axis=1)   # F[j] < F[i] in at least one dim
                dom_mask = le & lt
                dom_mask[i] = False  # exclude self
                if dom_mask.any():
                    dominated[i] = True
            return np.where(~dominated)[0]

        # ---------------- Phase 1: Latin Hypercube initial population ----------
        pop_size = int(min(25, max(8, self.budget // 8)))
        # LHS: divide each dimension into pop_size equal intervals, pick one random point per interval
        X = np.zeros((pop_size, self.dim))
        for d in range(self.dim):
            perm = rng.permutation(pop_size)
            for i in range(pop_size):
                X[i, d] = lo[d] + span[d] * (perm[i] + rng.random()) / pop_size

        F = np.empty((pop_size, self.n_obj))
        alive = 0
        for i in range(pop_size):
            res = evaluate(X[i])
            if res is None:
                break
            X[i] = res[0]
            F[i] = res[1]
            alive += 1

        # ---------------- Phase 2: DE-style evolutionary loop -----------------
        # Use differential evolution mutation: x_child = x_a + F*(x_b - x_c)
        F_factor = 0.8  # differential weight
        CR = 0.5  # crossover rate

        while evals_used < max_evals and time.time() < deadline:
            if alive < 3:
                # Need at least 3 parents for DE
                break

            # Get non-dominated set for selection
            nd_idx = nondominated_indices(F[:alive])
            if len(nd_idx) == 0:
                nd_idx = np.arange(alive)

            # Select 3 distinct parents
            if len(nd_idx) >= 3:
                p1, p2, p3 = rng.choice(nd_idx, size=3, replace=False)
            else:
                p1 = nd_idx[rng.integers(len(nd_idx))]
                p2 = nd_idx[rng.integers(len(nd_idx))]
                p3 = nd_idx[rng.integers(len(nd_idx))]
                if p1 == p2:
                    p2 = (p2 + 1) % alive
                if p3 in (p1, p2):
                    p3 = (p3 + 2) % alive

            # DE mutation
            progress = evals_used / max(1, max_evals)
            F_adaptive = F_factor * (1.0 - 0.4 * progress)  # shrink over time
            v = X[p1] + F_adaptive * (X[p2] - X[p3])

            # Binary crossover
            trial = X[p1].copy()
            cross_mask = rng.random(self.dim) < CR
            if not cross_mask.any():
                cross_mask[rng.integers(self.dim)] = True
            trial[cross_mask] = v[cross_mask]
            trial = np.clip(trial, lo, hi)

            res = evaluate(trial)
            if res is None:
                break
            x_new, f_new = res

            # Replace worst member (largest sum of objectives among dominated, or random if all ND)
            sums = np.sum(F[:alive], axis=1)
            # Prefer replacing dominated points
            dom_mask = np.zeros(alive, dtype=bool)
            for i in range(alive):
                le = (F[:alive] <= F[i]).all(axis=1)
                lt = (F[:alive] < F[i]).any(axis=1)
                dm = le & lt
                dm[i] = False
                if dm.any():
                    dom_mask[i] = True

            if dom_mask.any():
                worst = int(np.argmax(sums * dom_mask + np.where(dom_mask, 0, -1)))
            else:
                worst = int(np.argmax(sums))

            X[worst] = x_new
            F[worst] = f_new

        # ---------------- Phase 3: Extract non-dominated front -----------------
        front_idx = nondominated_indices(F[:alive])
        if len(front_idx) == 0:
            front_idx = np.array([int(np.argmin(np.sum(F[:alive], axis=1)))])
        front = [(X[i].copy(), F[i].copy()) for i in front_idx]

        # ---------------- Phase 4: Adaptive multi-objective hill polish --------
        reserved = 25

        def polish(x0, f0, max_iters=80):
            """Multi-objective hill climb with adaptive step size."""
            nonlocal evals_used
            cx = x0.copy()
            cf = f0.copy()
            step = 0.15
            best_f = cf.copy()
            best_x = cx.copy()

            for _ in range(max_iters):
                if evals_used >= max_evals - reserved:
                    break
                if time.time() > deadline:
                    break
                improved = False
                n_attempts = 8
                for _ in range(n_attempts):
                    if evals_used >= max_evals - reserved:
                        break
                    if time.time() > deadline:
                        break
                    dx = rng.standard_normal(self.dim) * step
                    xn = np.clip(cx + dx, lo, hi)
                    res = evaluate(xn)
                    if res is None:
                        break
                    xn, fn = res
                    # Accept if it dominates current (strict) OR is non-dominated vs current
                    dominates = np.all(fn <= cf) and np.any(fn < cf)
                    non_dom = (not (np.all(cf <= fn) and np.any(cf < fn)))
                    if dominates or non_dom:
                        cx, cf = xn, fn
                        improved = True
                        step *= 0.85
                        if np.all(fn <= best_f) and np.any(fn < best_f):
                            best_f = fn.copy()
                            best_x = xn.copy()
                        break
                if not improved:
                    step *= 0.55
                    if step < 0.005:
                        break
            return best_x, best_f

        # Polish each front point
        polished_front = []
        for x0, f0 in front:
            if evals_used >= max_evals - reserved:
                break
            if time.time() > deadline:
                break
            cx, cf = polish(x0, f0)
            polished_front.append((cx, cf))

        # Combine original front and polished
        all_points = front + polished_front

        # ---------------- Phase 5: Micro-refinement pass -----------------------
        # Re-extract front from all points
        if len(all_points) > 0:
            F_all = np.array([f for _, f in all_points])
            front2_idx = nondominated_indices(F_all)
            if len(front2_idx) > 0:
                for idx in front2_idx[:5]:  # Limit to top 5 for time
                    if evals_used >= max_evals - 5:
                        break
                    if time.time() > deadline:
                        break
                    x0, f0 = all_points[idx]
                    cx, cf = polish(x0, f0, max_iters=40)
                    all_points.append((cx, cf))

        # Final non-dominated extraction
        if len(all_points) > 0:
            F_final = np.array([f for _, f in all_points])
            final_idx = nondominated_indices(F_final)
            if len(final_idx) == 0:
                final_idx = np.array([0])
            return [all_points[i][0] for i in final_idx]
        else:
            return [X[0].copy()]
