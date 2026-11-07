"""Optimized optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Pattern search with adaptive step size, DFTR local model, and multi-start."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _pattern_search(self, func, rng, x0, max_calls, deadline):
        """
        Pattern search (Hooke-Jeeves style) with adaptive step size.
        Explores orthogonal and diagonal directions systematically.
        """
        dim = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        calls = 1

        # Initial step size
        step = 1.5

        # Directions: standard basis + diagonals
        directions = []
        # Standard basis
        for i in range(dim):
            e = np.zeros(dim)
            e[i] = 1.0
            directions.append(e)
        # Pairwise diagonals
        for i in range(dim):
            for j in range(i + 1, dim):
                d1 = np.zeros(dim)
                d1[i] = 1.0
                d1[j] = 1.0
                d1 /= np.linalg.norm(d1)
                directions.append(d1)

                d2 = np.zeros(dim)
                d2[i] = 1.0
                d2[j] = -1.0
                d2 /= np.linalg.norm(d2)
                directions.append(d2)

        # Add a few full diagonal directions
        for signs in [(1, 1, 1, 1, 1), (1, 1, 1, -1, -1), (1, -1, -1, 1, -1), (-1, 1, -1, -1, 1)]:
            d = np.array(signs, dtype=float)
            d /= np.linalg.norm(d)
            directions.append(d)

        no_improve_count = 0
        max_no_improve = 30

        while calls < max_calls:
            if time.time() > deadline - 5.0:
                break

            improved = False

            # Try each direction
            for d in directions:
                if calls >= max_calls or time.time() > deadline - 5.0:
                    break

                # Try positive direction
                x_new = np.clip(x + step * d, -5, 5)
                if np.allclose(x_new, x):
                    continue
                fx_new = func(x_new)
                calls += 1

                if fx_new < fx - 1e-10:
                    x = x_new
                    fx = fx_new
                    improved = True
                    no_improve_count = 0
                    # Extend in this direction
                    for _ in range(3):
                        if calls >= max_calls or time.time() > deadline - 5.0:
                            break
                        x_try = np.clip(x + step * d, -5, 5)
                        if np.allclose(x_try, x):
                            break
                        fx_try = func(x_try)
                        calls += 1
                        if fx_try < fx - 1e-10:
                            x = x_try
                            fx = fx_try
                        else:
                            break
                    break  # Restart direction scan after improvement

                # Try negative direction
                x_new_neg = np.clip(x - step * d, -5, 5)
                if np.allclose(x_new_neg, x):
                    continue
                fx_new_neg = func(x_new_neg)
                calls += 1

                if fx_new_neg < fx - 1e-10:
                    x = x_new_neg
                    fx = fx_new_neg
                    improved = True
                    no_improve_count = 0
                    # Extend in negative direction
                    for _ in range(3):
                        if calls >= max_calls or time.time() > deadline - 5.0:
                            break
                        x_try = np.clip(x - step * d, -5, 5)
                        if np.allclose(x_try, x):
                            break
                        fx_try = func(x_try)
                        calls += 1
                        if fx_try < fx - 1e-10:
                            x = x_try
                            fx = fx_try
                        else:
                            break
                    break  # Restart direction scan after improvement

            if not improved:
                no_improve_count += 1
                # Reduce step size
                step *= 0.7
                if no_improve_count >= max_no_improve:
                    break

            # Cap step size
            step = min(step, 10.0)
            step = max(step, 1e-8)

        return x, fx, calls

    def _dftr_local(self, func, rng, x0, delta0, max_calls, deadline):
        """
        Derivative-Free Trust-Region: build local quadratic model from 2n+1 points,
        step toward model minimizer within trust region.
        Returns (best_x, best_f, calls_used).
        """
        dim = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        calls = 1
        delta = delta0
        no_improve = 0

        while calls < max_calls:
            if time.time() > deadline - 1.5:
                break

            # Build 2n+1 point stencil: center + n forward + n backward
            pts = [x.copy()]
            fvals = [fx]
            for i in range(dim):
                if calls >= max_calls or time.time() > deadline - 1.5:
                    break
                xp = x.copy()
                xp[i] += delta
                xp = np.clip(xp, -5, 5)
                fp = func(xp)
                calls += 1
                pts.append(xp.copy())
                fvals.append(fp)
                if fp < fx:
                    fx = fp
                    x = xp.copy()

                if calls >= max_calls or time.time() > deadline - 1.5:
                    break
                xm = x.copy()
                xm[i] -= delta
                xm = np.clip(xm, -5, 5)
                fm = func(xm)
                calls += 1
                pts.append(xm.copy())
                fvals.append(fm)
                if fm < fx:
                    fx = fm
                    x = xm.copy()

            n_pts = len(pts)
            if n_pts < dim + 1:
                break

            # Estimate gradient via finite differences
            grad = np.zeros(dim)
            for i in range(dim):
                fwd_idx = None
                bwd_idx = None
                for j in range(1, n_pts):
                    dx = pts[j] - x
                    if abs(dx[i]) > 1e-12:
                        if dx[i] > 0:
                            fwd_idx = j
                        else:
                            bwd_idx = j
                if fwd_idx is not None and bwd_idx is not None:
                    h_fwd = pts[fwd_idx][i] - x[i]
                    h_bwd = pts[bwd_idx][i] - x[i]
                    grad[i] = (fvals[fwd_idx] - fvals[bwd_idx]) / (h_fwd - h_bwd)
                elif fwd_idx is not None:
                    h = pts[fwd_idx][i] - x[i]
                    grad[i] = (fvals[fwd_idx] - fvals[0]) / h if h != 0 else 0
                elif bwd_idx is not None:
                    h = pts[bwd_idx][i] - x[i]
                    grad[i] = (fvals[bwd_idx] - fvals[0]) / h if h != 0 else 0

            # Compute diagonal Hessian approximation
            diag_hess = np.zeros(dim)
            for i in range(dim):
                fwd_idx = None
                bwd_idx = None
                for j in range(1, n_pts):
                    dx = pts[j] - x
                    if abs(dx[i]) > 1e-12:
                        if dx[i] > 0:
                            fwd_idx = j
                        else:
                            bwd_idx = j
                if fwd_idx is not None and bwd_idx is not None:
                    h_fwd = pts[fwd_idx][i] - x[i]
                    h_bwd = pts[bwd_idx][i] - x[i]
                    if abs(h_fwd) > 1e-12 and abs(h_bwd) > 1e-12:
                        diag_hess[i] = (fvals[fwd_idx] - 2 * fvals[0] + fvals[bwd_idx]) / (h_fwd * h_bwd)
                if diag_hess[i] <= 0:
                    diag_hess[i] = 1e-6

            # Newton step with diagonal Hessian
            step = -grad / diag_hess
            step_norm = np.linalg.norm(step)

            if step_norm < 1e-10:
                delta *= 0.7
                no_improve += 1
                if no_improve > 5:
                    break
                continue

            # Trust region constraint
            scale = min(1.0, delta / step_norm)
            x_new = x + scale * step
            x_new = np.clip(x_new, -5, 5)

            # Evaluate new point
            if calls >= max_calls or time.time() > deadline - 1.5:
                break
            f_new = func(x_new)
            calls += 1

            if f_new < fx:
                x = x_new
                fx = f_new
                no_improve = 0
                delta = min(10.0, delta * 1.3)
            else:
                delta *= 0.5
                no_improve += 1
                if no_improve > 10:
                    break

            delta = max(1e-6, min(delta, 10.0))

        return x, fx, calls

    def _coordinate_refine(self, func, rng, x0, max_calls, deadline):
        """
        Cyclic coordinate descent with adaptive step sizes for final polishing.
        """
        dim = self.dim
        x = np.array(x0, dtype=float)
        fx = func(x)
        calls = 1
        step_sizes = np.full(dim, 0.5)
        no_improve_total = 0

        while calls < max_calls:
            if time.time() > deadline - 1.0:
                break

            improved = False
            for i in range(dim):
                if calls >= max_calls or time.time() > deadline - 1.0:
                    break

                x_try = x.copy()
                x_try[i] += step_sizes[i]
                x_try = np.clip(x_try, -5, 5)
                f_try = func(x_try)
                calls += 1

                if f_try < fx:
                    x = x_try
                    fx = f_try
                    improved = True
                    step_sizes[i] = min(2.0, step_sizes[i] * 1.2)
                else:
                    x_try2 = x.copy()
                    x_try2[i] -= step_sizes[i]
                    x_try2 = np.clip(x_try2, -5, 5)
                    f_try2 = func(x_try2)
                    calls += 1

                    if f_try2 < fx:
                        x = x_try2
                        fx = f_try2
                        improved = True
                        step_sizes[i] = min(2.0, step_sizes[i] * 1.2)
                    else:
                        step_sizes[i] = max(1e-6, step_sizes[i] * 0.7)

            if not improved:
                no_improve_total += 1
                if no_improve_total > 3:
                    break
            else:
                no_improve_total = 0

        return x, fx, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        dim = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        total_calls = 0
        budget = self.budget

        # Phase 1: Multi-start with pattern search
        n_starts = 4
        calls_per_start = max(30, budget // (n_starts + 2))

        start_points = [
            rng.uniform(-1, 1, dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-5, 5, dim),
            rng.uniform(-3, 3, dim),
        ]

        for i, x0 in enumerate(start_points):
            if total_calls >= budget or time.time() > deadline - 30.0:
                break

            x, fx, calls_used = self._pattern_search(
                func, rng, x0,
                min(calls_per_start, budget - total_calls),
                deadline
            )
            total_calls += calls_used

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: DFTR local refinement on best point
        if best_x is not None and total_calls < budget:
            remaining = budget - total_calls
            if remaining > 20:
                x, fx, calls_used = self._dftr_local(
                    func, rng, best_x, 1.0,
                    min(remaining, 100), deadline
                )
                total_calls += calls_used
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Phase 3: Coordinate descent polish
        if best_x is not None and total_calls < budget:
            remaining = budget - total_calls
            if remaining > 10:
                x, fx, calls_used = self._coordinate_refine(
                    func, rng, best_x,
                    min(remaining, 50), deadline
                )
                total_calls += calls_used
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        # Ensure we've made at least one call
        if total_calls == 0:
            x = rng.uniform(-5, 5, dim)
            best_f = func(x)
            best_x = x
            total_calls = 1

        return best_x, best_f
# EVOLVE-BLOCK-END
