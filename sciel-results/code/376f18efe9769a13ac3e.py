"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start coordinate descent optimizer for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0

        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        def eval_and_track(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            calls += 1
            try:
                val = float(func(x))
            except Exception:
                return best_f
            if np.isnan(val) or np.isinf(val):
                return best_f
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(42)

        # Phase 1: Random search to find basins
        n_random = 80
        candidates = []
        for i in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, size=dim)
            f_val = eval_and_track(x)
            candidates.append((f_val, x.copy()))

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Coordinate descent refinement on top candidates
        def golden_section_1d(x_fixed, coord_idx, lo, hi, max_evals):
            """1D golden-section search along one coordinate."""
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return x_fixed[coord_idx]
            
            phi = (1.0 + 5.0**0.5) / 2.0
            inv_phi = 1.0 / phi
            inv_phi2 = inv_phi * inv_phi
            
            a = lo
            b = hi
            c = b - inv_phi * (b - a)
            d = a + inv_phi * (b - a)
            
            x_c = x_fixed.copy()
            x_d = x_fixed.copy()
            x_c[coord_idx] = c
            x_d[coord_idx] = d
            
            f_c = eval_and_track(x_c)
            f_d = eval_and_track(x_d)
            
            for _ in range(max_evals):
                if calls >= budget or time.time() > deadline:
                    break
                if f_c < f_d:
                    b = d
                    d = c
                    f_d = f_c
                    c = b - inv_phi * (b - a)
                    x_c = x_fixed.copy()
                    x_c[coord_idx] = c
                    f_c = eval_and_track(x_c)
                else:
                    a = c
                    c = d
                    f_c = f_d
                    d = a + inv_phi * (b - a)
                    x_d = x_fixed.copy()
                    x_d[coord_idx] = d
                    f_d = eval_and_track(x_d)
            
            # Return midpoint of final interval
            mid = (a + b) / 2.0
            return mid

        def coordinate_descent(x0, max_rounds=10, evals_per_coord=20):
            """Full coordinate descent: sweep all coordinates multiple times."""
            nonlocal calls, best_f, best_x
            x = x0.copy()
            f_current = eval_and_track(x)
            
            for rnd in range(max_rounds):
                if calls >= budget or time.time() > deadline:
                    break
                improved = False
                # Shuffle coordinate order to avoid systematic bias
                coord_order = list(range(dim))
                rng.shuffle(coord_order)
                
                for ci in coord_order:
                    if calls >= budget or time.time() > deadline:
                        break
                    # Determine search range around current value
                    x_curr = x[ci]
                    half_width = max(0.5, 0.1 * (5.0 - abs(x_curr)))
                    lo = max(-5.0, x_curr - half_width)
                    hi = min(5.0, x_curr + half_width)
                    
                    # Golden section search
                    new_val = golden_section_1d(x, ci, lo, hi, evals_per_coord)
                    
                    if abs(new_val - x_curr) > 1e-10:
                        x[ci] = new_val
                        improved = True
                        f_new = eval_and_track(x)
                        if f_new < best_f:
                            best_f = f_new
                            best_x = x.copy()
                    
                    if calls >= budget or time.time() > deadline:
                        break
                
                if not improved:
                    break
            
            return x

        # Refine top candidates
        top_k = 8
        for i in range(top_k):
            if calls >= budget or time.time() > deadline:
                break
            if budget - calls < 15:
                break
            
            x0 = candidates[i][1]
            # Allocate more evals to better candidates
            remaining = budget - calls
            max_rounds = min(15, max(3, remaining // (dim * 25)))
            evals_per_coord = min(25, max(10, remaining // (dim * max_rounds)))
            
            try:
                coordinate_descent(x0, max_rounds=max_rounds, evals_per_coord=evals_per_coord)
            except Exception:
                pass

        # Phase 3: Final polish with tighter tolerance on the best point
        if calls < budget and time.time() < deadline:
            remaining = budget - calls
            if remaining > 10:
                # Do a few more rounds of coordinate descent with tighter search
                x_best = best_x.copy()
                for rnd in range(5):
                    if calls >= budget or time.time() > deadline:
                        break
                    for ci in range(dim):
                        if calls >= budget or time.time() > deadline:
                            break
                        x_curr = x_best[ci]
                        half_width = max(0.1, 0.05 * (5.0 - abs(x_curr)))
                        lo = max(-5.0, x_curr - half_width)
                        hi = min(5.0, x_curr + half_width)
                        new_val = golden_section_1d(x_best, ci, lo, hi, 15)
                        if abs(new_val - x_curr) > 1e-10:
                            x_best[ci] = new_val
                            f_new = eval_and_track(x_best)
                            if f_new < best_f:
                                best_f = f_new
                                best_x = x_best.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
