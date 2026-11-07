"""Optimizer for BBOB-style f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Parallel coordinate descent with adaptive golden-section 1D refinement and multi-start restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        phi = (np.sqrt(5.0) - 1.0) / 2.0

        calls = 0
        best_x = None
        best_f = np.inf
        last_improvement_call = 0

        def evaluate(x):
            nonlocal calls, best_f, best_x, last_improvement_call
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                last_improvement_call = calls
            return f

        def golden_section_1d(x_base, coord, lo, hi, max_iter, tol=1e-10):
            """Minimize along coordinate `coord` of x_base in [lo, hi] using golden-section.
            Returns (best_x, best_f)."""
            a, b = lo, hi
            if b - a < 1e-12:
                return x_base.copy(), np.inf

            c = b - phi * (b - a)
            d = a + phi * (b - a)

            xc = x_base.copy()
            xc[coord] = c
            fc = evaluate(xc)

            xd = x_base.copy()
            xd[coord] = d
            fd = evaluate(xd)

            best_val = c if fc < fd else d
            best_f_local = fc if fc < fd else fd

            for _ in range(max(0, max_iter - 2)):
                if calls >= budget:
                    break
                if b - a < tol:
                    break
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - phi * (b - a)
                    xc = x_base.copy()
                    xc[coord] = c
                    fc = evaluate(xc)
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + phi * (b - a)
                    xd = x_base.copy()
                    xd[coord] = d
                    fd = evaluate(xd)

                if fc < fd:
                    if fc < best_f_local:
                        best_f_local = fc
                        best_val = c
                else:
                    if fd < best_f_local:
                        best_f_local = fd
                        best_val = d

            best_x_local = x_base.copy()
            best_x_local[coord] = best_val
            return best_x_local, best_f_local

        def coordinate_descent_adaptive(x_start, n_sweeps, gs_iters, window, min_window=0.01):
            """Perform coordinate descent sweeps with adaptive window. Returns (best_x, best_f)."""
            x = x_start.copy()
            fx = evaluate(x) if calls < budget else np.inf
            if calls >= budget:
                return x, fx

            coords = list(range(dim))
            coord_improvement = np.zeros(dim)

            for sweep in range(n_sweeps):
                if calls >= budget:
                    break
                if sweep > 0:
                    # Sort by recent improvement (descending)
                    order = np.argsort(-coord_improvement)
                    coords = order.tolist()
                    coord_improvement *= 0.5  # Decay

                for coord in coords:
                    if calls >= budget:
                        break

                    lo = max(-5.0, x[coord] - window)
                    hi = min(5.0, x[coord] + window)

                    if hi - lo < 1e-12:
                        continue

                    x_new, fx_new = golden_section_1d(x, coord, lo, hi, gs_iters)

                    if fx_new < fx:
                        improvement = fx - fx_new
                        coord_improvement[coord] += improvement
                        x = x_new
                        fx = fx_new

            return x, fx

        # Phase 1: Structured initial exploration for fast early AOCC
        initial_points = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.array([2.0, -2.0, 2.0, -2.0, 2.0]),
            np.array([-2.0, 2.0, -2.0, 2.0, -2.0]),
            np.array([4.0, -4.0, 4.0, -4.0, 4.0]),
            np.array([-4.0, 4.0, -4.0, 4.0, -4.0]),
            rng.uniform(-3, 3, dim),
            rng.uniform(-3, 3, dim),
            rng.uniform(-3, 3, dim),
        ]

        for x_init in initial_points:
            if calls >= budget:
                break
            x_init = np.clip(x_init, -5, 5)
            fx = evaluate(x_init)

            # Quick wide-window coordinate sweep
            if calls < budget:
                x_sw, fx_sw = coordinate_descent_adaptive(x_init, n_sweeps=1, gs_iters=6, window=4.0)
                if fx_sw < best_f:
                    best_f = fx_sw
                    best_x = x_sw.copy()

        # Phase 2: Adaptive multi-scale refinement
        if calls < budget and best_x is not None:
            scales = [
                (1.2, 10, 3),
                (0.6, 12, 4),
                (0.3, 14, 5),
                (0.15, 16, 6),
                (0.08, 18, 7),
                (0.04, 20, 8),
            ]

            for window, gs_iters, n_starts in scales:
                if calls >= budget:
                    break

                for s in range(n_starts):
                    if calls >= budget:
                        break

                    if s == 0:
                        x_start = best_x.copy()
                    else:
                        # Adaptive perturbation size
                        pert_scale = window * 0.5
                        x_start = best_x + rng.normal(0, pert_scale, dim)
                        x_start = np.clip(x_start, -5, 5)

                    fx_start = evaluate(x_start)

                    if calls < budget:
                        x_sw, fx_sw = coordinate_descent_adaptive(x_start, n_sweeps=3, gs_iters=gs_iters, window=window)
                        if fx_sw < best_f:
                            best_f = fx_sw
                            best_x = x_sw.copy()

        # Phase 3: Final high-precision polish with multiple passes
        if calls < budget and best_x is not None:
            for _ in range(6):
                if calls >= budget:
                    break
                x_sw, fx_sw = coordinate_descent_adaptive(best_x, n_sweeps=3, gs_iters=12, window=0.02)
                if fx_sw < best_f:
                    best_f = fx_sw
                    best_x = x_sw.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
