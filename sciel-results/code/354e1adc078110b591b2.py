import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        d = self.dim
        N = self.budget
        deadline = self.start_time + 280.0

        def time_check():
            return time.time() < deadline

        def safe_eval(x):
            if self.calls >= N or not time_check():
                return np.inf
            self.calls += 1
            try:
                val = float(func(x))
                return val
            except Exception:
                return np.inf

        # Track best globally
        best_x = np.zeros(d)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Initial evaluation at origin
        x0 = np.zeros(d)
        f0 = safe_eval(x0)
        update_best(x0, f0)

        # Phase 1: L-BFGS-B from multiple starting points
        rng = np.random.default_rng(42)
        start_points = [x0.copy()]
        for _ in range(5):
            start_points.append(rng.uniform(-4, 4, d))

        # Wrap func for scipy
        def wrapped_func(x):
            if self.calls >= N or not time_check():
                return np.inf
            return safe_eval(x)

        bounds = [(-5.0, 5.0)] * d

        for sp in start_points:
            if not time_check() or self.calls >= N:
                break
            
            f_sp = safe_eval(sp)
            update_best(sp, f_sp)
            
            try:
                result = minimize(
                    wrapped_func,
                    sp,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                if result.fun < best_f:
                    update_best(result.x, result.fun)
            except Exception:
                pass

        # Phase 2: Refine with Nelder-Mead from the best point found
        if self.calls < N - 10 and time_check():
            def nelder_mead(x_start, f_start, max_iters=100):
                nonlocal best_f, best_x
                if not time_check():
                    return

                delta = 0.5
                simplex = [x_start.copy()]
                f_vals = [f_start]

                for i in range(d):
                    if not time_check() or self.calls >= N:
                        return
                    x_i = x_start.copy()
                    x_i[i] += delta
                    f_i = safe_eval(x_i)
                    update_best(x_i, f_i)
                    simplex.append(x_i)
                    f_vals.append(f_i)

                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma = 0.5
                tol = 1e-10

                for _ in range(max_iters):
                    if not time_check() or self.calls >= N:
                        break

                    combined = list(zip(f_vals, simplex))
                    combined.sort(key=lambda x: x[0])
                    f_vals = [c[0] for c in combined]
                    simplex = [c[1] for c in combined]

                    x_worst = simplex[-1]
                    f_worst = f_vals[-1]
                    x_best = simplex[0]
                    f_best = f_vals[0]

                    if abs(f_worst - f_best) < tol:
                        break

                    centroid = np.sum(simplex[:-1], axis=0) / d

                    x_r = centroid + alpha * (centroid - x_worst)
                    f_r = safe_eval(x_r)
                    update_best(x_r, f_r)

                    if f_r < f_best:
                        x_e = centroid + gamma * (centroid - x_worst)
                        f_e = safe_eval(x_e)
                        update_best(x_e, f_e)
                        if f_e < f_r:
                            simplex[-1] = x_e
                            f_vals[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            f_vals[-1] = f_r
                    elif f_r < f_vals[-2]:
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                    else:
                        x_c = centroid + rho * (x_worst - centroid)
                        f_c = safe_eval(x_c)
                        update_best(x_c, f_c)
                        if f_c < f_worst:
                            simplex[-1] = x_c
                            f_vals[-1] = f_c
                        else:
                            for i in range(1, d + 1):
                                if not time_check() or self.calls >= N:
                                    break
                                simplex[i] = x_best + sigma * (simplex[i] - x_best)
                                f_vals[i] = safe_eval(simplex[i])
                                update_best(simplex[i], f_vals[i])

            nelder_mead(best_x, best_f, max_iters=80)

        return best_x, best_f
