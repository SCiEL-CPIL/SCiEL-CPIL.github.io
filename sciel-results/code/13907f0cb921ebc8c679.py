"""Optimizer for BBOB-style f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Parallel coordinate descent with adaptive golden-section 1D refinement and multi-start restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        phi = (np.sqrt(5.0) - 1.0) / 2.0

        calls = 0
        best_x = None
        best_f = np.inf

        def golden_section_1d(x_base, coord, lo, hi, max_iter):
            """Minimize along coordinate `coord` of x_base in [lo, hi] using golden-section.
            Returns (best_x, best_f, num_calls_used)."""
            nonlocal calls
            a, b = lo, hi
            if b - a < 1e-10:
                return x_base.copy(), func(x_base), 0

            c = b - phi * (b - a)
            d = a + phi * (b - a)

            xc = x_base.copy()
            xc[coord] = c
            fc = func(xc)
            calls += 1

            xd = x_base.copy()
            xd[coord] = d
            fd = func(xd)
            calls += 1

            best_f_local = fc if fc < fd else fd
            best_val = c if fc < fd else d

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
                    fc = func(xc)
                    calls += 1
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + phi * (b - a)
                    xd = x_base.copy()
                    xd[coord] = d
                    fd = func(xd)
                    calls += 1

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
            return best_x_local, best_f_local, calls

        def coordinate_sweep(x_start, fx_start, n_sweeps, gs_iters, window):
            """Perform coordinate descent sweeps. Returns (best_x, best_f, calls_used)."""
            nonlocal calls
            x = x_start.copy()
            fx = fx_start
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

                    if hi - lo < 1e-10:
                        continue

                    x_new, fx_new, _ = golden_section_1d(x, coord, lo, hi, gs_iters)

                    if fx_new < fx:
                        x = x_new
                        fx = fx_new

                    if calls >= budget:
                        break

            return x, fx, calls

        # Phase 1: Quick exploration with wide windows and fewer iterations
        # This gives early progress for AOCC scoring
        n_initial = 5
        for i in range(n_initial):
            if calls >= budget:
                break

            if i == 0:
                x = rng.uniform(-5, 5, dim)
            else:
                # Perturb best found so far
                x = best_x + rng.normal(0, 2.0, dim)
                x = np.clip(x, -5, 5)

            fx = func(x)
            calls += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Do a quick coordinate sweep with wide window
            if calls < budget:
                x_sw, fx_sw, _ = coordinate_sweep(x, fx, n_sweeps=1, gs_iters=6, window=4.0)
                if fx_sw < best_f:
                    best_f = fx_sw
                    best_x = x_sw.copy()

        # Phase 2: Refined search around best point with narrower windows
        if calls < budget and best_x is not None:
            # Multiple restarts from best point with decreasing windows
            windows = [2.0, 1.0, 0.5, 0.25]
            gs_iters_list = [10, 12, 14, 16]

            for w, gi in zip(windows, gs_iters_list):
                if calls >= budget:
                    break

                # Try a few perturbations
                n_perturb = 3
                for p in range(n_perturb):
                    if calls >= budget:
                        break

                    if p == 0:
                        x_start = best_x.copy()
                    else:
                        x_start = best_x + rng.normal(0, w * 0.5, dim)
                        x_start = np.clip(x_start, -5, 5)

                    fx_start = func(x_start)
                    calls += 1

                    if fx_start < best_f:
                        best_f = fx_start
                        best_x = x_start.copy()

                    if calls < budget:
                        x_sw, fx_sw, _ = coordinate_sweep(x_start, fx_start, n_sweeps=2, gs_iters=gi, window=w)
                        if fx_sw < best_f:
                            best_f = fx_sw
                            best_x = x_sw.copy()

        # Phase 3: Final polish with very narrow window
        if calls < budget and best_x is not None:
            x_sw, fx_sw, _ = coordinate_sweep(best_x, best_f, n_sweeps=3, gs_iters=8, window=0.1)
            if fx_sw < best_f:
                best_f = fx_sw
                best_x = x_sw.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
