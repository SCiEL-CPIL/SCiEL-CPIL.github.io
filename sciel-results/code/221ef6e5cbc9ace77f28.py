"""Optimizer for BBOB-style f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive lattice + CMA-ES-lite + Nelder-Mead hybrid for 21-peak f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        dim = self.dim
        budget = self.budget
        deadline = t0 + 290.0  # safety margin vs 330s kill

        # Local state
        x_best = np.zeros(dim)
        f_best = np.inf
        call_count = 0
        rng = np.random.default_rng(0)

        def evaluate(x):
            nonlocal x_best, f_best, call_count
            x = np.asarray(x, dtype=float).reshape(-1)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            call_count += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            return fx

        # ---- Phase 1: structured global scan (Halton-ish + jitter) ----
        # Use a low-discrepancy-ish grid to cover the box and find multiple basins.
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = int(index)
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_point(n, dim):
            p = np.empty(dim)
            for d in range(dim):
                # distinct prime bases
                primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
                p[d] = halton(n + 1, primes[d % len(primes)])
            return p

        # Initial dense-ish scan: ~120 points spread over the box
        n_scan = 120
        for k in range(1, n_scan + 1):
            if call_count >= budget:
                break
            if time.time() > deadline:
                break
            p = halton_point(k, dim) * 10.0 - 5.0
            # small jitter to avoid exact lattice hits
            p = p + rng.uniform(-0.2, 0.2, dim)
            evaluate(p)

        # ---- Phase 2: adaptive lattice refinement ----
        # Maintain a set of candidate centers; shrink the promising region.
        # Start from the best scan point, define a region around it.
        if not np.isfinite(f_best):
            # fallback
            x_best = np.zeros(dim)
            f_best = evaluate(x_best)

        # Build an adaptive grid search: maintain a region (center, half-width)
        # and sample a grid inside it, then shrink toward the best cell.
        center = x_best.copy()
        half_width = np.full(dim, 5.0)  # full box

        # We'll do several rounds of grid refinement.
        n_rounds = 6
        grid_per_axis = 4
        for r in range(n_rounds):
            if call_count >= budget:
                break
            if time.time() > deadline:
                break

            # Sample a grid within the current region
            best_cell_x = None
            best_cell_f = np.inf
            for idx in np.ndindex(*([grid_per_axis] * dim)):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                # position within cell: center of the sub-cell
                frac = np.array([(i + 0.5) / grid_per_axis for i in idx])
                pt = center + (frac - 0.5) * 2.0 * half_width
                pt = np.clip(pt, -5.0, 5.0)
                fx = evaluate(pt)
                if fx < best_cell_f:
                    best_cell_f = fx
                    best_cell_x = pt.copy()

            if best_cell_x is None:
                break

            # Shrink region toward the best cell
            center = best_cell_x
            half_width = half_width * 0.5

            # After a few rounds, the region is small -> switch to local polish
            if r >= 2 and np.all(half_width < 1.0):
                break

        # ---- Phase 3: local Nelder-Mead polish from the best point ----
        # Use remaining budget for Nelder-Mead on the best basin.
        def nelder_mead(start, budget_left, max_iter=200):
            nonlocal call_count
            d = dim
            # Initial simplex
            simplex = [start.copy()]
            fvals = [evaluate(start)]
            for i in range(d):
                xi = start.copy()
                xi[i] = start[i] + 0.2
                xi = np.clip(xi, -5.0, 5.0)
                simplex.append(xi)
                fvals.append(evaluate(xi))
                if call_count >= budget:
                    break

            max_iter = min(max_iter, max(1, budget_left - 5))
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

            for it in range(max_iter):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                # sort
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                x_worst = simplex[-1]
                f_worst = fvals[-1]
                x_best_s = simplex[0]
                f_best_s = fvals[0]

                # centroid excluding worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - x_worst)
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = evaluate(x_r)
                if call_count >= budget:
                    break

                if f_r < f_best_s:
                    # Expansion
                    x_e = centroid + gamma * (centroid - x_worst)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = evaluate(x_e)
                    if call_count >= budget:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (x_worst - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = evaluate(x_c)
                    if call_count >= budget:
                        break
                    if f_c < f_worst:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = x_best_s + sigma * (simplex[i] - x_best_s)
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fvals[i] = evaluate(simplex[i])
                            if call_count >= budget:
                                break
                        if call_count >= budget:
                            break

            # Return best in simplex
            order = np.argsort(fvals)
            return simplex[order[0]], fvals[order[0]]

        budget_left = budget - call_count
        if budget_left > 10 and time.time() < deadline:
            x_nm, f_nm = nelder_mead(x_best, budget_left)
            if f_nm < f_best:
                x_best = x_nm
                f_best = f_nm

        # ---- Phase 4: coordinate descent polish ----
        if call_count < budget and time.time() < deadline:
            x_cd = x_best.copy()
            f_cd = f_best
            step = 0.5
            for _ in range(50):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                improved = False
                for i in range(dim):
                    if call_count >= budget:
                        break
                    for direction in (1, -1):
                        x_try = x_cd.copy()
                        x_try[i] += direction * step
                        x_try = np.clip(x_try, -5.0, 5.0)
                        f_try = evaluate(x_try)
                        if f_try < f_cd:
                            x_cd = x_try
                            f_cd = f_try
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 1e-3:
                        break
                else:
                    step *= 1.2

            if f_cd < f_best:
                x_best = x_cd
                f_best = f_cd

        # ---- Phase 5: final Nelder-Mead if budget remains ----
        budget_left = budget - call_count
        if budget_left > 20 and time.time() < deadline:
            x_nm2, f_nm2 = nelder_mead(x_best, budget_left, max_iter=100)
            if f_nm2 < f_best:
                x_best = x_nm2
                f_best = f_nm2

        # ---- Phase 6: use any remaining budget for a final fine grid around best ----
        if call_count < budget and time.time() < deadline:
            # tiny grid around the best point
            radius = 0.3
            for _ in range(20):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                pt = x_best + rng.uniform(-radius, radius, dim)
                pt = np.clip(pt, -5.0, 5.0)
                evaluate(pt)
            radius = 0.1
            for _ in range(20):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                pt = x_best + rng.uniform(-radius, radius, dim)
                pt = np.clip(pt, -5.0, 5.0)
                evaluate(pt)

        # Final Nelder-Mead if still budget
        budget_left = budget - call_count
        if budget_left > 15 and time.time() < deadline:
            x_nm3, f_nm3 = nelder_mead(x_best, budget_left, max_iter=80)
            if f_nm3 < f_best:
                x_best = x_nm3
                f_best = f_nm3

        return x_best, f_best
# EVOLVE-BLOCK-END
