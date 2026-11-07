"""Rewritten optimizer for BBOB f7 (Step ellipsoidal).
Structural change: LHS initialization + Nelder-Mead simplex search with
adaptive restarts. NM navigates flat plateaus better than (1+λ)-ES for
the step-ellipsoidal landscape in low dimensions with limited budget."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """LHS + Nelder-Mead with adaptive restarts for step-ellipsoidal f7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _lhs(self, n, dim, rng):
        """Latin Hypercube Sampling."""
        pts = np.zeros((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            pts[:, d] = (perm + rng.random(n)) / n * 10.0 - 5.0
        return pts

    def _nelder_mead(self, func, x0, calls_made, max_calls, deadline,
                     rng, max_iter=80, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.1):
        """Nelder-Mead simplex optimization. Returns (best_x, best_f, calls_made)."""
        dim = self.dim
        n = dim + 1
        # Build initial simplex
        simplex = [x0.copy()]
        for i in range(dim):
            pt = x0.copy()
            pt[i] += sigma
            simplex.append(np.clip(pt, -5, 5))

        # Evaluate initial simplex
        fvals = []
        for i in range(n):
            if time.time() >= deadline or calls_made >= max_calls:
                break
            fvals.append(func(simplex[i]))
            calls_made += 1

        if len(fvals) < n:
            # Not enough budget; return best found
            idx = np.argmin(fvals)
            return simplex[idx], fvals[idx], calls_made

        best_f = min(fvals)
        best_x = simplex[np.argmin(fvals)].copy()

        for _ in range(max_iter):
            if time.time() >= deadline or calls_made >= max_calls:
                break

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
            fr = func(xr)
            calls_made += 1

            if fr < fvals[0]:
                # Expansion
                xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                fe = func(xe)
                calls_made += 1
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                xc = np.clip(centroid + rho * (simplex[-1] - centroid), -5, 5)
                fc = func(xc)
                calls_made += 1
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n):
                        simplex[i] = np.clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]), -5, 5)
                        fvals[i] = func(simplex[i])
                        calls_made += 1
                    if calls_made >= max_calls or time.time() >= deadline:
                        break

            # Track best
            cur_best_idx = np.argmin(fvals)
            if fvals[cur_best_idx] < best_f:
                best_f = fvals[cur_best_idx]
                best_x = simplex[cur_best_idx].copy()

            # Convergence check
            spread = np.std(fvals)
            if spread < 1e-10:
                break

        return best_x, best_f, calls_made

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0
        max_calls = self.budget  # 500
        dim = self.dim

        def elapsed_ok():
            return time.time() < deadline and calls_made < max_calls

        # Phase 1: LHS initialization - spread out initial points
        n_lhs = 15
        lhs_pts = self._lhs(n_lhs, dim, rng)
        lhs_fvals = []
        for i in range(n_lhs):
            if not elapsed_ok():
                break
            fx = func(lhs_pts[i])
            calls_made += 1
            lhs_fvals.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = lhs_pts[i].copy()

        if not lhs_fvals:
            return best_x, best_f

        # Phase 2: Nelder-Mead from best LHS point, then from other promising points
        # Sort LHS points by function value
        lhs_order = np.argsort(lhs_fvals)

        # NM from the best point
        nm_sigma = 0.5
        best_x, best_f, calls_made = self._nelder_mead(
            func, lhs_pts[lhs_order[0]].copy(), calls_made, max_calls,
            deadline, rng, max_iter=60, sigma=nm_sigma
        )

        # NM from second-best LHS point if budget allows
        if elapsed_ok() and calls_made < max_calls - 20:
            x2 = lhs_pts[lhs_order[1]].copy()
            # Don't restart if too close to current best
            if np.linalg.norm(x2 - best_x) > 0.5:
                best_x2, best_f2, calls_made = self._nelder_mead(
                    func, x2, calls_made, max_calls,
                    deadline, rng, max_iter=40, sigma=0.3
                )
                if best_f2 < best_f:
                    best_f = best_f2
                    best_x = best_x2

        # Phase 3: Additional NM restarts from random points if budget allows
        n_restarts = 3
        for r in range(n_restarts):
            if not elapsed_ok() or calls_made >= max_calls - 15:
                break
            # Random restart point, biased toward better regions
            if best_x is not None:
                x_r = np.clip(best_x + rng.standard_normal(dim) * 2.0, -5, 5)
            else:
                x_r = rng.uniform(-4, 4, dim)
            
            # Only restart if far enough from current best
            if best_x is not None and np.linalg.norm(x_r - best_x) < 0.3:
                x_r = np.clip(x_r + rng.standard_normal(dim) * 1.0, -5, 5)

            best_xr, best_fr, calls_made = self._nelder_mead(
                func, x_r, calls_made, max_calls,
                deadline, rng, max_iter=30, sigma=0.2
            )
            if best_fr < best_f:
                best_f = best_fr
                best_x = best_xr

        # Phase 4: Final local refinement around global best
        if elapsed_ok() and calls_made < max_calls - 5 and best_x is not None:
            # Fine NM with small sigma
            best_x, best_f, calls_made = self._nelder_mead(
                func, best_x.copy(), calls_made, max_calls,
                deadline, rng, max_iter=20, sigma=0.05
            )

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
