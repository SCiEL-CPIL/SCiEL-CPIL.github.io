"""Optimizer for BBOB f22 (Gallagher 21 peaks) using Powell's method with
deterministic axis rotation, multi-start from Halton samples, and a final
coordinate-descent polish.  Powell's method builds an implicit search-direction
basis that adapts to the local geometry without any covariance model, which
should exploit the 21-peaks landscape more efficiently than Nelder-Mead."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330 s kill
        n = self.dim
        budget = self.budget
        total_calls = 0
        best_x = np.zeros(n)
        best_f = np.inf

        def try_eval(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget:
                return None
            if time.time() > deadline - 2:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            f = float(func(x))
            total_calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # ---------- Halton sequence helpers ----------
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        _primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        def halton_vec(idx, dim):
            pt = np.empty(dim)
            for d in range(dim):
                pt[d] = halton(idx, _primes[d % len(_primes)])
            return pt

        # ---------- Phase 1: broad Halton sampling ----------
        n_phase1 = min(40, budget // 10)
        for i in range(1, n_phase1 + 1):
            if total_calls >= budget or time.time() > deadline - 15:
                break
            x_try = -4.0 + 8.0 * halton_vec(i, n)
            try_eval(x_try)

        # A few structured points
        for p in [np.zeros(n), 2.0 * np.ones(n), -2.0 * np.ones(n)]:
            if total_calls >= budget or time.time() > deadline - 15:
                break
            try_eval(p)

        # ---------- Powell's method ----------
        def powell(x0, max_iter=200):
            """Deterministic Powell method with axis rotation.
            Returns nothing; updates best_x / best_f via try_eval."""
            nonlocal best_x, best_f, total_calls
            if total_calls >= budget:
                return

            x = x0.copy()
            fx = try_eval(x)
            if fx is None:
                return

            # Initial directions: standard basis
            directions = np.eye(n)

            for it in range(max_iter):
                if total_calls >= budget or time.time() > deadline - 5:
                    return

                x_old = x.copy()
                fx_old = fx
                delta = 0.0

                # Line searches along each direction
                for j in range(n):
                    if total_calls >= budget or time.time() > deadline - 5:
                        return
                    d = directions[j]
                    # Find best step along d via golden-section or parabolic
                    # Simple: try several steps and pick best
                    best_step = 0.0
                    best_f_line = fx
                    # Try positive and negative steps with shrinking
                    for step in [1.0, 0.5, 0.25, 0.1, 0.05, -1.0, -0.5, -0.25, -0.1, -0.05]:
                        if total_calls >= budget or time.time() > deadline - 5:
                            return
                        x_try = x + step * d
                        x_try = np.clip(x_try, -5.0, 5.0)
                        f_try = try_eval(x_try)
                        if f_try is None:
                            return
                        if f_try < best_f_line:
                            best_f_line = f_try
                            best_step = step

                    if best_step != 0.0:
                        x = x + best_step * d
                        fx = best_f_line
                        delta += abs(fx - fx_old)
                        fx_old = fx

                # Check convergence
                if abs(fx - fx_old) < 1e-10 * (1.0 + abs(fx)):
                    return

                # Build new direction
                x_new = x.copy()
                fx_new = fx
                for j in range(n):
                    if total_calls >= budget or time.time() > deadline - 5:
                        return
                    d = directions[j]
                    best_step = 0.0
                    best_f_line = fx_new
                    for step in [1.0, 0.5, 0.25, 0.1, -1.0, -0.5, -0.25, -0.1]:
                        if total_calls >= budget or time.time() > deadline - 5:
                            return
                        x_try = x_new + step * d
                        x_try = np.clip(x_try, -5.0, 5.0)
                        f_try = try_eval(x_try)
                        if f_try is None:
                            return
                        if f_try < best_f_line:
                            best_f_line = f_try
                            best_step = step
                    if best_step != 0.0:
                        x_new = x_new + best_step * d
                        fx_new = best_f_line

                # Replace the direction that contributed most to improvement
                # (simplified: replace the last direction)
                new_dir = x_new - x_old
                norm_new = np.linalg.norm(new_dir)
                if norm_new > 1e-12:
                    # Find direction with largest contribution
                    max_j = 0
                    max_val = 0.0
                    # We didn't track per-direction deltas, so just replace last
                    directions[n - 1] = new_dir

                    # If new direction improves, do a final line search along it
                    if fx_new < fx:
                        x = x_new
                        fx = fx_new

            # Final polish at the end
            if total_calls < budget and time.time() < deadline - 3:
                for coord in range(n):
                    if total_calls >= budget or time.time() > deadline - 3:
                        break
                    for step in [0.1, 0.05, 0.01, -0.1, -0.05, -0.01]:
                        if total_calls >= budget or time.time() > deadline - 3:
                            break
                        x_try = best_x + step * np.eye(n)[coord]
                        x_try = np.clip(x_try, -5.0, 5.0)
                        f_try = try_eval(x_try)
                        if f_try is not None and f_try < best_f:
                            best_f = f_try
                            best_x = x_try.copy()

        # ---------- Phase 2: Powell from best point ----------
        if total_calls < budget - 20 and time.time() < deadline - 20:
            powell(best_x.copy(), max_iter=150)

        # ---------- Phase 3: Powell from a few other promising starts ----------
        rng = np.random.default_rng(123)
        for _ in range(3):
            if total_calls >= budget - 30 or time.time() > deadline - 30:
                break
            x_start = best_x + rng.normal(0, 0.5, n)
            x_start = np.clip(x_start, -4.0, 4.0)
            f_start = try_eval(x_start)
            if f_start is not None and f_start < best_f + 1.0:
                if total_calls < budget - 20 and time.time() < deadline - 20:
                    powell(x_start, max_iter=80)

        # ---------- Phase 4: final coordinate-descent polish ----------
        if total_calls < budget and time.time() < deadline - 3:
            step = 0.2
            for _ in range(5):
                if total_calls >= budget or time.time() > deadline - 3:
                    break
                improved = False
                for coord in range(n):
                    if total_calls >= budget or time.time() > deadline - 3:
                        break
                    for direction in [-1, 1]:
                        if total_calls >= budget or time.time() > deadline - 3:
                            break
                        x_try = best_x.copy()
                        x_try[coord] += direction * step
                        x_try = np.clip(x_try, -5.0, 5.0)
                        f_try = try_eval(x_try)
                        if f_try is not None and f_try < best_f:
                            best_f = f_try
                            best_x = x_try.copy()
                            improved = True
                step *= 0.5

        return best_x, best_f
