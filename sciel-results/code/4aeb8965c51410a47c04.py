"""Combined optimizer for BBOB f22:
Halton-based broad sampling + Powell's method (conjugate direction) + Nelder-Mead refinement + coordinate polish.
Powell's method is used as the primary local refiner because it maintains an implicit search direction basis,
exploiting local convexity with fewer evaluations than Nelder-Mead in smooth regions.
"""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin of 45s before 330s kill
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

        # Halton sequence for stratified sampling
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_vector(index, dims):
            point = []
            bases = [2, 3, 5, 7, 11]
            for d in range(dims):
                base = bases[d % len(bases)]
                point.append(halton(index, base))
            return np.array(point)

        # Phase 1: Stratified sampling with Halton sequence
        n_phase1 = min(40, budget // 7)
        phase1_results = []
        for i in range(1, n_phase1 + 1):
            if total_calls >= budget or time.time() > deadline - 10:
                break
            h_point = halton_vector(i, n)
            x_try = -4.0 + 8.0 * h_point
            f_try = try_eval(x_try)
            if f_try is None:
                break
            phase1_results.append((f_try, x_try.copy()))

        # Phase 1b: Add structured points
        structured = [
            np.zeros(n),
            np.ones(n) * 2.0,
            -np.ones(n) * 2.0,
            np.ones(n) * -2.0,
            np.ones(n) * 4.0,
            -np.ones(n) * 4.0,
            np.ones(n) * 1.0,
            -np.ones(n) * 1.0,
        ]
        for p in structured:
            if total_calls >= budget or time.time() > deadline - 10:
                break
            f_s = try_eval(p)
            if f_s is not None:
                phase1_results.append((f_s, p.copy()))

        # Powell's method: conjugate direction search
        def powell_refine(x0, max_calls):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget or x0 is None:
                return best_x.copy(), best_f

            # Initial directions: standard basis
            directions = [np.zeros(n) for _ in range(n)]
            for i in range(n):
                directions[i][i] = 1.0

            x = x0.copy()
            x = np.clip(x, -5.0, 5.0)

            # Evaluate starting point
            if total_calls >= budget or time.time() > deadline - 5:
                return best_x.copy(), best_f
            f = try_eval(x)
            if f is None:
                return best_x.copy(), best_f

            # Line search along a direction
            def line_search(x_start, direction):
                nonlocal total_calls, best_x, best_f
                # Try a few step sizes to find a good minimum along the line
                step_sizes = [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, -0.5, -0.25, -0.1, -0.05, -0.02, -0.01]
                best_x_line = x_start.copy()
                best_f_line = try_eval(x_start)
                if best_f_line is None:
                    return x_start, np.inf

                for step in step_sizes:
                    if total_calls >= budget or time.time() > deadline - 3:
                        break
                    x_try = x_start + step * direction
                    x_try = np.clip(x_try, -5.0, 5.0)
                    f_try = try_eval(x_try)
                    if f_try is None:
                        break
                    if f_try < best_f_line:
                        best_f_line = f_try
                        best_x_line = x_try.copy()

                return best_x_line, best_f_line

            # Powell's main loop
            for iteration in range(30):
                if total_calls >= budget or time.time() > deadline - 5:
                    break

                x_old = x.copy()
                f_old = try_eval(x)
                if f_old is None:
                    break

                min_f = np.inf
                min_dir = 0
                max_f = -np.inf
                max_dir = 0

                for i in range(n):
                    if total_calls >= budget or time.time() > deadline - 3:
                        break
                    x_new, f_new = line_search(x, directions[i])
                    if f_new < min_f:
                        min_f = f_new
                        min_dir = i
                    if f_new > max_f:
                        max_f = f_new
                        max_dir = i
                    x = x_new.copy()

                # If no improvement, stop
                if f_old - min_f < 1e-8:
                    break

                # Compute new conjugate direction
                x_new = x_old + (x - x_old)
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = try_eval(x_new)
                if f_new is None:
                    break

                # If the new point is better than the worst point in the previous iteration
                if f_new < f_old - (max_f - min_f) / 2:
                    # Replace the direction with the largest change
                    directions[max_dir] = (x - x_old)
                    # Shift other directions
                    for i in range(n - 1, 0, -1):
                        if i != max_dir:
                            directions[i] = directions[i - 1]
                    directions[0] = directions[max_dir]

                x = x_new.copy()

            # Update global best
            f_final = try_eval(x)
            if f_final is not None and f_final < best_f:
                best_f = f_final
                best_x = x.copy()

            return best_x.copy(), best_f

        # Nelder-Mead for final polish
        def nelder_mead(x0, max_calls):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget or x0 is None:
                return best_x.copy(), best_f

            alpha = 0.5
            simplex = [x0.copy()]
            for i in range(n):
                xi = x0.copy()
                xi[i] += alpha if x0[i] < 4.0 else -alpha
                simplex.append(xi)

            simplex_f = []
            for i in range(n + 1):
                if total_calls >= budget or time.time() > deadline - 5:
                    return best_x.copy(), best_f
                f_i = try_eval(simplex[i])
                if f_i is None:
                    return best_x.copy(), best_f
                simplex_f.append(f_i)

            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]

            gamma = 1.5
            rho = 2.0
            sigma_nm = 0.5
            no_improve = 0
            patience = 20

            while total_calls < budget and time.time() < deadline - 5:
                f_range = simplex_f[-1] - simplex_f[0]
                x_range = max(np.linalg.norm(simplex[-1] - simplex[0]), 1e-12)
                if f_range < 1e-7 and x_range < 1e-5:
                    break
                if no_improve >= patience:
                    break

                centroid = np.mean([simplex[i] for i in range(n)], axis=0)
                x_r = centroid + gamma * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                if total_calls >= budget or time.time() > deadline - 3:
                    break
                f_r = try_eval(x_r)
                if f_r is None:
                    break

                if simplex_f[0] <= f_r < simplex_f[-1]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                    no_improve = 0 if f_r < simplex_f[-2] else no_improve + 1
                elif f_r < simplex_f[0]:
                    x_e = centroid + rho * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    if total_calls >= budget or time.time() > deadline - 3:
                        break
                    f_e = try_eval(x_e)
                    if f_e is None:
                        break
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                    no_improve = 0
                else:
                    x_c = centroid + sigma_nm * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    if total_calls >= budget or time.time() > deadline - 3:
                        break
                    f_c = try_eval(x_c)
                    if f_c is None:
                        break
                    if f_c < simplex_f[-1]:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                        no_improve += 1
                    else:
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            if total_calls >= budget or time.time() > deadline - 3:
                                return best_x.copy(), best_f
                            f_i = try_eval(simplex[i])
                            if f_i is None:
                                return best_x.copy(), best_f
                            simplex_f[i] = f_i
                        no_improve += 1

                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

            best_idx = int(np.argmin(simplex_f))
            if simplex_f[best_idx] < best_f:
                best_f = simplex_f[best_idx]
                best_x = simplex[best_idx].copy()
            return best_x.copy(), best_f

        # Phase 2: Run Powell's method from top diverse points
        if phase1_results:
            phase1_results.sort(key=lambda x: x[0])

            powell_starts = []
            for f_val, x_val in phase1_results[:5]:
                is_distinct = True
                for _, x_existing in powell_starts:
                    if np.linalg.norm(x_val - x_existing) < 0.8:
                        is_distinct = False
                        break
                if is_distinct and total_calls < budget - 30:
                    powell_starts.append((f_val, x_val))
                if len(powell_starts) >= 3:
                    break

            for f_start, x_start in powell_starts:
                if total_calls < budget - 20 and time.time() < deadline - 15:
                    powell_refine(x_start.copy(), budget - total_calls)

        # Phase 3: Nelder-Mead polish from global best
        if total_calls < budget - 10 and best_x is not None and time.time() < deadline - 10:
            nelder_mead(best_x.copy(), budget - total_calls)

        # Phase 4: Coordinate descent polish
        if total_calls < budget and best_x is not None and time.time() < deadline - 5:
            for step in [0.3, 0.15, 0.075, 0.04, 0.02, 0.01]:
                if total_calls >= budget or time.time() > deadline - 3:
                    break
                for coord in range(n):
                    if total_calls >= budget or time.time() > deadline - 3:
                        break
                    for direction in [-1, 1]:
                        x_try = best_x.copy()
                        x_try[coord] += direction * step
                        x_try = np.clip(x_try, -5.0, 5.0)
                        f_try = try_eval(x_try)
                        if f_try is None:
                            break
                        if f_try < best_f:
                            best_f = f_try
                            best_x = x_try.copy()
                            for _ in range(3):
                                if total_calls >= budget or time.time() > deadline - 3:
                                    break
                                x_next = best_x.copy()
                                x_next[coord] += direction * step
                                x_next = np.clip(x_next, -5.0, 5.0)
                                f_next = try_eval(x_next)
                                if f_next is None:
                                    break
                                if f_next < best_f:
                                    best_f = f_next
                                    best_x = x_next.copy()
                                else:
                                    break

        # Phase 5: Final random perturbations
        if total_calls < budget and best_x is not None and time.time() < deadline - 2:
            rng_final = np.random.default_rng(999)
            for scale in [0.05, 0.02, 0.01]:
                if total_calls >= budget or time.time() > deadline - 2:
                    break
                for _ in range(5):
                    if total_calls >= budget or time.time() > deadline - 2:
                        break
                    x_perturb = best_x + rng_final.normal(0, scale, n)
                    x_perturb = np.clip(x_perturb, -5.0, 5.0)
                    f_perturb = try_eval(x_perturb)
                    if f_perturb is not None and f_perturb < best_f:
                        best_f = f_perturb
                        best_x = x_perturb.copy()

        return best_x, best_f
