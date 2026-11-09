"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer for BBOB f13 (Sharp Ridge):
    1. Low-discrepancy Halton initialization for broad early coverage (AOCC).
    2. Adaptive (1+1)-ES with ridge-aware step scaling.
    3. Nelder-Mead refinement for final convergence on narrow ridges.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton_sequence(self, n, dim, skip=1):
        """Generate n Halton sequence points in [0,1)^dim."""
        points = []
        for i in range(skip, skip + n):
            point = []
            for j in range(1, dim + 1):
                base = j + 1
                f = 1.0
                result = 0.0
                i_val = i
                while i_val > 0:
                    f /= base
                    result += f * (i_val % base)
                    i_val //= base
                point.append(result)
            points.append(point)
        return np.array(points)

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        box_low = -5.0
        box_high = 5.0
        calls_made = 0

        def clip(x):
            return np.clip(x, box_low, box_high)

        def evaluate(x):
            nonlocal calls_made
            if calls_made >= budget or time.time() > deadline:
                return None
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = x.reshape(dim)
            x = clip(x)
            f = func(x)
            calls_made += 1
            return x, f

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Halton initialization
        n_init = max(10, min(50, budget // 5))
        halton_points = self._halton_sequence(n_init, dim, skip=1)
        init_points = halton_points * 10.0 - 5.0

        all_points = []
        all_fvals = []

        for p in init_points:
            if time.time() > deadline:
                break
            res = evaluate(p)
            if res is None:
                break
            x, f = res
            all_points.append(x)
            all_fvals.append(f)
            update_best(x, f)

        # Add structured points
        structured = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.array([1, -1, 1, -1, 1]) * 2.0,
            np.array([3, -3, 0, 3, -3]),
        ]
        for x in structured:
            if time.time() > deadline - 60:
                break
            res = evaluate(x)
            if res is None:
                break
            x, f = res
            all_points.append(x)
            all_fvals.append(f)
            update_best(x, f)

        if not all_fvals:
            return best_x, best_f

        # Phase 2: Adaptive (1+1)-ES refinement on top candidates
        sorted_idx = np.argsort(all_fvals)
        top_k = min(5, len(sorted_idx))
        remaining_budget = budget - calls_made

        if remaining_budget > 0:
            for rank in range(top_k):
                if time.time() > deadline - 30:
                    break

                idx = sorted_idx[rank]
                x = all_points[idx].copy()
                fx = all_fvals[idx]

                alloc = max(1, int(remaining_budget * (0.6 ** rank) / sum(0.6 ** r for r in range(top_k))))
                sigma = 1.0
                successes = 0
                trials = 0
                no_improve = 0

                for _ in range(alloc):
                    if time.time() > deadline - 30 or calls_made >= budget:
                        break

                    # Ridge-aware: slightly larger steps to follow ridges
                    y = np.clip(x + sigma * np.random.default_rng(int(time.time()*1000) + rank).standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls_made += 1

                    if fy < fx:
                        x, fx = y, fy
                        successes += 1
                        sigma *= 1.2
                        no_improve = 0
                    else:
                        sigma *= 0.9
                        no_improve += 1

                    trials += 1

                    if trials >= 10:
                        success_rate = successes / trials
                        if success_rate > 0.3:
                            sigma *= 1.1
                        elif success_rate < 0.1:
                            sigma *= 0.85
                        successes = 0
                        trials = 0

                    update_best(x, fx)

                    if no_improve > 20:
                        break

        # Phase 3: Nelder-Mead on the best point
        if best_x is not None and calls_made < budget and time.time() < deadline - 20:
            x_nm, f_nm = self._nelder_mead(best_x, best_f, func, clip, evaluate, update_best, deadline)
            if f_nm < best_f:
                best_f = f_nm
                best_x = x_nm

        return best_x, best_f

    def _nelder_mead(self, x0, f0, func, clip, evaluate, update_best, deadline):
        dim = self.dim
        simplex = [x0.copy()]
        fvals = [f0]

        step = 0.5
        for d in range(dim):
            if time.time() > deadline - 20:
                break
            x_new = x0.copy()
            x_new[d] += step
            x_new = clip(x_new)
            res = evaluate(x_new)
            if res is None:
                break
            x_new, f_new = res
            simplex.append(x_new)
            fvals.append(f_new)
            update_best(x_new, f_new)

        if len(simplex) < dim + 1:
            return x0, f0

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

        for _ in range(200):
            if time.time() > deadline - 10:
                break

            order = np.argsort(fvals)
            simplex = [simplex[i].copy() for i in order]
            fvals = [fvals[i] for i in order]

            centroid = np.mean(simplex[:-1], axis=0)

            x_r = clip(centroid + alpha * (centroid - simplex[-1]))
            res = evaluate(x_r)
            if res is None:
                break
            x_r, f_r = res
            update_best(x_r, f_r)

            if f_r < fvals[0]:
                x_e = clip(centroid + gamma * (x_r - centroid))
                res = evaluate(x_e)
                if res is None:
                    break
                x_e, f_e = res
                update_best(x_e, f_e)
                if f_e < f_r:
                    simplex[-1], fvals[-1] = x_e, f_e
                else:
                    simplex[-1], fvals[-1] = x_r, f_r
            elif f_r < fvals[-2]:
                simplex[-1], fvals[-1] = x_r, f_r
            else:
                x_c = clip(centroid + rho * (simplex[-1] - centroid))
                res = evaluate(x_c)
                if res is None:
                    break
                x_c, f_c = res
                update_best(x_c, f_c)
                if f_c < f_r:
                    simplex[-1], fvals[-1] = x_c, f_c
                else:
                    for i in range(1, len(simplex)):
                        simplex[i] = clip(simplex[0] + sigma * (simplex[i] - simplex[0]))
                        res = evaluate(simplex[i])
                        if res is None:
                            return simplex[0], fvals[0]
                        simplex[i], fvals[i] = res
                        update_best(simplex[i], fvals[i])

            if abs(fvals[-1] - fvals[0]) < 1e-10:
                break

        best_idx = np.argmin(fvals)
        return simplex[best_idx].copy(), fvals[best_idx]

# EVOLVE-BLOCK-END
