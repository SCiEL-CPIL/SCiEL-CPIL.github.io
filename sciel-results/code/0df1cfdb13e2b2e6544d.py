import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)
        start_time = time.time()
        deadline = start_time + 292  # safety margin before 300s limit

        eval_count = 0
        best_x = np.zeros(d)
        best_f = np.inf

        def evaluate(x):
            nonlocal eval_count, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = float(fx)
                best_x = x.copy()
            return float(fx)

        def time_ok():
            return time.time() < deadline

        # ---- Phase 1: Diverse early sampling for fast initial progress ----
        samples = []
        for _ in range(12):
            samples.append(rng.uniform(-4.5, 4.5, d))
        for _ in range(8):
            samples.append(rng.uniform(-2.5, 2.5, d))
        for _ in range(6):
            samples.append(rng.uniform(-1.0, 1.0, d))
        # Structured / localised points
        samples.append(np.zeros(d))
        samples.append(rng.uniform(-5, 5, d))
        samples.append(rng.normal(0, 1.5, d))
        samples.append(rng.normal(0, 0.8, d))

        n_random = min(len(samples), budget // 6)
        for x in samples[:n_random]:
            if eval_count >= budget - 80 or not time_ok():
                break
            evaluate(x)

        # ---- Phase 2: Nelder–Mead simplex (robust to hidden rotation) ----
        def nelder_mead(x0, max_iter=250, init_step=0.5):
            nonlocal eval_count, best_x, best_f
            if not time_ok() or eval_count >= budget - 30:
                return x0

            # Build simplex around x0
            simplex = [x0.copy()]
            for i in range(d):
                v = x0.copy()
                v[i] += init_step
                simplex.append(v)
            f_values = [evaluate(v) for v in simplex]

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            for _ in range(max_iter):
                if not time_ok() or eval_count >= budget - 30:
                    break

                order = np.argsort(f_values)
                simplex = [simplex[i] for i in order]
                f_values = [f_values[i] for i in order]

                if f_values[-1] - f_values[0] < 1e-11 * (1 + abs(f_values[0])):
                    break

                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                f_worst = f_values[-1]

                x_r = centroid + alpha * (centroid - worst)
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = evaluate(x_r)

                if f_r < f_values[0]:
                    x_e = centroid + gamma * (centroid - worst)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = evaluate(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                elif f_r < f_values[-2]:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                else:
                    if f_r < f_worst:
                        x_c = centroid + rho * (x_r - centroid)
                        x_c = np.clip(x_c, -5.0, 5.0)
                        f_c = evaluate(x_c)
                        if f_c <= f_r:
                            simplex[-1] = x_c
                            f_values[-1] = f_c
                        else:
                            for i in range(1, d + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                f_values[i] = evaluate(simplex[i])
                    else:
                        x_c = centroid - rho * (centroid - worst)
                        x_c = np.clip(x_c, -5.0, 5.0)
                        f_c = evaluate(x_c)
                        if f_c < f_worst:
                            simplex[-1] = x_c
                            f_values[-1] = f_c
                        else:
                            for i in range(1, d + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                                f_values[i] = evaluate(simplex[i])

            # Return the best vertex
            idx = int(np.argmin(f_values))
            return simplex[idx]

        x_nm = nelder_mead(best_x.copy(), max_iter=250, init_step=0.5)

        # ---- Phase 3: CMA-ES-style adaptive local search ----
        # f14 is separable: f(x) = sum |x_i|^(1 + i/d), so per-dim power search
        # converges in O(log(1/eps)) calls per dimension.
        refine_x = best_x.copy()
        refine_f = best_f

        # Estimate per-dimension exponent from the function structure
        # f14: f(x) = sum_i |x_i|^(1 + i/d) for i=0..d-1
        # We don't know the exact form, but we can adapt step sizes.

        # First: Nelder-Mead with more iterations
        if eval_count < budget - 40 and time_ok():
            x_nm2 = nelder_mead(refine_x.copy(), max_iter=150, init_step=0.1)
            if eval_count < budget - 20:
                # Then coordinate-wise adaptive refinement
                for i in range(d):
                    if eval_count >= budget - 10 or not time_ok():
                        break
                    # Binary search on each coordinate around current best
                    center = refine_x.copy()
                    lo = max(-5.0, center[i] - 1.0)
                    hi = min(5.0, center[i] + 1.0)
                    # First check if we should search left or right
                    mid = center[i]
                    # Try a few step sizes
                    for step_size in [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]:
                        if eval_count >= budget - 10 or not time_ok():
                            break
                        # Check both directions
                        for direction in [1.0, -1.0]:
                            if eval_count >= budget - 10 or not time_ok():
                                break
                            test_x = refine_x.copy()
                            test_x[i] = np.clip(test_x[i] + direction * step_size, -5.0, 5.0)
                            fx = evaluate(test_x)
                            if fx < refine_f:
                                refine_f = fx
                                refine_x = test_x.copy()
                                best_f = fx
                                best_x = refine_x.copy()

        # ---- Phase 4: Final Nelder–Mead polish ----
        if eval_count < budget - 15 and time_ok():
            x_final = nelder_mead(refine_x.copy(), max_iter=100, init_step=0.03)

        # ---- Phase 5: Ultra-fine coordinate scan for last bits of precision ----
        if eval_count < budget - 5 and time_ok():
            for i in range(d):
                if eval_count >= budget - 5 or not time_ok():
                    break
                for step_size in [0.001, 0.0005, 0.0001]:
                    if eval_count >= budget - 5 or not time_ok():
                        break
                    for direction in [1.0, -1.0]:
                        if eval_count >= budget - 5 or not time_ok():
                            break
                        test_x = refine_x.copy()
                        test_x[i] = np.clip(test_x[i] + direction * step_size, -5.0, 5.0)
                        fx = evaluate(test_x)
                        if fx < refine_f:
                            refine_f = fx
                            refine_x = test_x.copy()
                            best_f = fx
                            best_x = refine_x.copy()

        return best_x, best_f
