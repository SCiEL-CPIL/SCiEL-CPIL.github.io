import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = 280.0  # safety margin before 300s

        call_count = 0
        best_x = None
        best_f = np.inf

        def tracked_func(x):
            nonlocal call_count, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError()
            if call_count >= self.budget:
                raise ValueError("budget exceeded")
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            if x.shape != (self.dim,):
                x = np.resize(x, self.dim)
            f = func(x)
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Nelder-Mead
        x0 = np.zeros(self.dim)
        try:
            f0 = tracked_func(x0)
        except:
            return best_x, best_f

        simplex = [x0.copy()]
        simplex_f = [f0]

        for i in range(self.dim):
            xi = x0.copy()
            xi[i] = 2.0
            try:
                fi = tracked_func(xi)
            except:
                return best_x, best_f
            simplex.append(xi)
            simplex_f.append(fi)

        alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
        max_nm_iter = 200

        for _ in range(max_nm_iter):
            if call_count >= self.budget - 60 or time.time() > deadline - 20:
                break

            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]

            if abs(simplex_f[-1] - simplex_f[0]) < 1e-10 * (abs(simplex_f[0]) + 1e-10):
                break

            centroid = np.mean(simplex[:-1], axis=0)
            x_r = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)

            try:
                f_r = tracked_func(x_r)
            except:
                break

            if f_r < simplex_f[0]:
                x_e = np.clip(centroid + gamma * (x_r - centroid), -5, 5)
                try:
                    f_e = tracked_func(x_e)
                except:
                    break
                if f_e < f_r:
                    simplex[-1], simplex_f[-1] = x_e, f_e
                else:
                    simplex[-1], simplex_f[-1] = x_r, f_r
            elif f_r < simplex_f[-2]:
                simplex[-1], simplex_f[-1] = x_r, f_r
            else:
                if f_r < simplex_f[-1]:
                    x_c = np.clip(centroid + rho * (x_r - centroid), -5, 5)
                else:
                    x_c = np.clip(centroid - rho * (simplex[-1] - centroid), -5, 5)
                try:
                    f_c = tracked_func(x_c)
                except:
                    break
                if f_c < min(f_r, simplex_f[-1]):
                    simplex[-1], simplex_f[-1] = x_c, f_c
                else:
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        try:
                            simplex_f[i] = tracked_func(simplex[i])
                        except:
                            break
                    else:
                        continue
                    break

        # Phase 2: L-BFGS-B
        if call_count < self.budget - 5 and time.time() < deadline - 10:
            best_x_nm = simplex[0]
            try:
                result = minimize(
                    tracked_func,
                    best_x_nm,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 30,
                        'maxfun': self.budget - call_count - 5,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                    }
                )
            except:
                pass

        return best_x, best_f
