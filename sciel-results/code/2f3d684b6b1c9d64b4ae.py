import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                raise TimeoutError()
            # Ensure x is a proper numpy array
            x = np.asarray(x, dtype=float)
            # Clip to box [-5, 5]
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls += 1
            if not np.isfinite(fx):
                fx = float('inf')
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global Search using Differential Evolution
        # DE is a population-based stochastic optimizer that adaptively explores
        # the search space, better suited for Rosenbrock's narrow valley than
        # simple uniform random sampling.
        try:
            de_result = differential_evolution(
                safe_func,
                bounds=[(-5.0, 5.0)] * dim,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                mut=(0.3, 1.0),
                recomb=0.7,
                polish=False,
                init='latinhypercube',
                update_interval=1
            )
            if de_result.success and np.isfinite(de_result.fun):
                if de_result.fun < best_f:
                    best_f = de_result.fun
                    best_x = de_result.x.copy()
        except Exception:
            pass

        # Phase 2: Local Refinement using L-BFGS-B on best point found
        # L-BFGS-B is efficient for Rosenbrock once in the valley.
        remaining = budget - calls
        if remaining > 20:
            try:
                res = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        'maxiter': 200,
                        'ftol': 1e-14,
                        'gtol': 1e-8,
                        'maxls': 30
                    }
                )
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: Final Polish with Nelder-Mead if budget remains
        remaining = budget - calls
        if remaining > 20:
            x0 = best_x
            sigma = 0.1

            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                if xi[i] + sigma > 5:
                    xi[i] -= 2 * sigma
                elif xi[i] - sigma < -5:
                    xi[i] += 2 * sigma
                else:
                    xi[i] += sigma
                simplex.append(xi)

            simplex_f = []
            for i in range(dim + 1):
                if calls >= budget - 5 or time.time() > deadline:
                    break
                simplex_f.append(safe_func(simplex[i]))

            if len(simplex_f) == dim + 1:
                simplex_data = list(zip(simplex_f, simplex))

                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma_shrink = 0.5

                while calls < budget - 5 and time.time() < deadline:
                    simplex_data.sort(key=lambda item: item[0])
                    f_worst = simplex_data[-1][0]
                    x_worst = simplex_data[-1][1]
                    f_best = simplex_data[0][0]
                    x_best = simplex_data[0][1]

                    if f_best < best_f:
                        best_f = f_best
                        best_x = x_best.copy()

                    f_spread = f_worst - f_best
                    if f_spread < 1e-10:
                        break

                    centroid = np.mean([item[1] for item in simplex_data[:-1]], axis=0)

                    x_r = centroid + alpha * (centroid - x_worst)
                    x_r = np.clip(x_r, -5, 5)
                    f_r = safe_func(x_r)

                    if f_r < f_best:
                        x_e = centroid + gamma * (centroid - x_worst)
                        x_e = np.clip(x_e, -5, 5)
                        f_e = safe_func(x_e)
                        if f_e < f_r:
                            simplex_data[-1] = (f_e, x_e)
                        else:
                            simplex_data[-1] = (f_r, x_r)
                    elif f_r < simplex_data[-2][0]:
                        simplex_data[-1] = (f_r, x_r)
                    else:
                        if f_r < f_worst:
                            x_c = centroid + rho * (x_r - centroid)
                        else:
                            x_c = centroid - rho * (centroid - x_worst)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = safe_func(x_c)
                        if f_c <= f_r:
                            simplex_data[-1] = (f_c, x_c)
                        else:
                            for i in range(1, len(simplex_data)):
                                x_s = x_best + sigma_shrink * (simplex_data[i][1] - x_best)
                                x_s = np.clip(x_s, -5, 5)
                                f_s = safe_func(x_s)
                                simplex_data[i] = (f_s, x_s)

        return best_x, best_f
