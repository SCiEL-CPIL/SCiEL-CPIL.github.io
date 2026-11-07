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

        def evaluate(x):
            nonlocal calls, best_f, best_x
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def golden_section_1d(x_base, coord, lo, hi, max_iter):
            """Minimize along coordinate `coord` of x_base in [lo, hi] using golden-section.
            Returns (best_x, best_f, num_calls_used)."""
            a, b = lo, hi
            if b - a < 1e-12:
                return x_base.copy(), np.inf, 0

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

        def coordinate_descent(x_start, n_sweeps, gs_iters, window):
            """Perform coordinate descent sweeps. Returns (best_x, best_f)."""
            x = x_start.copy()
            fx = evaluate(x) if calls < budget else np.inf
            if calls >= budget:
                return x, fx

            coords = list(range(dim))

            for sweep in range(n_sweeps):
                if calls >= budget:
                    break
                if sweep > 0:
                    rng.shuffle(coords)

                for coord in coords:
                    if calls >= budget:
                        break

                    lo = max(-5.0, x[coord] - window)
                    hi = min(5.0, x[coord] + window)

                    if hi - lo < 1e-12:
                        continue

                    x_new, fx_new = golden_section_1d(x, coord, lo, hi, gs_iters)

                    if fx_new < fx:
                        x = x_new
                        fx = fx_new

            return x, fx

        # Phase 1: Aggressive exploration - rapid initial points for early AOCC
        n_initial = 8
        for i in range(n_initial):
            if calls >= budget:
                break

            if i == 0:
                x = np.zeros(dim)
            elif i == 1:
                x = rng.uniform(-4, 4, dim)
            elif i == 2:
                x = rng.uniform(-3, 3, dim)
            elif i == 3:
                x = rng.uniform(-2, 2, dim)
            else:
                # Perturb best found so far
                x = best_x + rng.normal(0, 1.5, dim)
                x = np.clip(x, -5, 5)

            fx = evaluate(x)

            # Quick coordinate sweep with wide window for immediate improvement
            if calls < budget:
                x_sw, fx_sw = coordinate_descent(x, n_sweeps=1, gs_iters=5, window=3.5)
                if fx_sw < best_f:
                    best_f = fx_sw
                    best_x = x_sw.copy()

        # Phase 2: Multi-scale refinement around best point
        if calls < budget and best_x is not None:
            scales = [
                (1.5, 8, 2),   # window, gs_iters, n_starts
                (0.8, 10, 3),
                (0.4, 12, 4),
                (0.2, 14, 5),
                (0.1, 16, 6),
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
                        x_start = best_x + rng.normal(0, window * 0.4, dim)
                        x_start = np.clip(x_start, -5, 5)

                    fx_start = evaluate(x_start)

                    if calls < budget:
                        x_sw, fx_sw = coordinate_descent(x_start, n_sweeps=2, gs_iters=gs_iters, window=window)
                        if fx_sw < best_f:
                            best_f = fx_sw
                            best_x = x_sw.copy()

        # Phase 3: Final high-precision polish
        if calls < budget and best_x is not None:
            # Multiple narrow-window passes
            for _ in range(4):
                if calls >= budget:
                    break
                x_sw, fx_sw = coordinate_descent(best_x, n_sweeps=2, gs_iters=10, window=0.05)
                if fx_sw < best_f:
                    best_f = fx_sw
                    best_x = x_sw.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
