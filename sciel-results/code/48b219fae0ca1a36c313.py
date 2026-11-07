import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = None

    def __call__(self, func):
        self.deadline = time.time() + 275
        self.calls = 0
        self.best_x = np.zeros(self.dim)
        self.best_f = np.inf

        dim = self.dim
        budget = self.budget

        def wrapped_func(x):
            if self.calls >= budget:
                return np.inf
            if time.time() > self.deadline:
                return np.inf
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5, 5)
            self.calls += 1
            f = float(func(x))
            if not np.isfinite(f):
                return f
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
            return f

        from scipy.optimize import minimize

        # Custom Nelder-Mead to avoid scipy overhead for final polish
        def nelder_mead(f, x0, max_iter=100, tol=1e-10):
            n = len(x0)
            alpha, gamma, rho, sigma = 0.5, 2.0, 1.0, 0.5
            simplex = [x0.copy()]
            f0 = f(x0)
            simplex.append(f0)
            for i in range(n):
                xi = x0.copy()
                xi[i] = x0[i] + 0.1
                simplex.append(xi)
                simplex.append(f(xi))
            
            for _ in range(max_iter):
                if time.time() > self.deadline - 2:
                    break
                # Sort simplex
                idx = np.argsort(simplex[1::2])
                simplex = [simplex[0]] + [simplex[i+1] for i in idx]
                
                f_best = simplex[1]
                f_worst = simplex[-1]
                if abs(f_worst - f_best) < tol:
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[0:-1:2], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-2])
                fr = f(xr)
                if f_best <= fr < f_worst:
                    simplex[-2] = xr
                    simplex[-1] = fr
                elif fr < f_best:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-2])
                    fe = f(xe)
                    if fe < fr:
                        simplex[-2] = xe
                        simplex[-1] = fe
                    else:
                        simplex[-2] = xr
                        simplex[-1] = fr
                else:
                    # Contraction
                    xc = centroid + sigma * (simplex[-2] - centroid)
                    fc = f(xc)
                    if fc < f_worst:
                        simplex[-2] = xc
                        simplex[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex), 2):
                            if i != 1:
                                simplex[i-1] = simplex[0] + 0.5 * (simplex[i-1] - simplex[0])
                                simplex[i] = f(simplex[i-1])
            
            # Return best
            idx = np.argmin(simplex[1::2])
            return simplex[idx*2], simplex[idx*2+1]

        # Generate diverse starting points
        np.random.seed(42)
        starts = [np.zeros(dim)]
        for _ in range(4):
            starts.append(np.random.uniform(-4, 4, dim))
        starts.append(np.random.uniform(-5, -3, dim))
        starts.append(np.random.uniform(3, 5, dim))

        num_starts = len(starts)

        for i in range(num_starts):
            if self.calls >= budget - 10:
                break
            if time.time() > self.deadline - 15:
                break

            x0 = starts[i].copy()
            f0 = wrapped_func(x0)

            if not np.isfinite(f0):
                continue

            # Phase 1: L-BFGS-B for efficient gradient-based optimization
            if self.calls < budget - 20 and time.time() < self.deadline - 10:
                try:
                    remaining = min(80, budget - self.calls - 10)
                    if remaining > 20:
                        res = minimize(wrapped_func, x0, method='L-BFGS-B',
                                       bounds=[(-5, 5)] * dim,
                                       options={'maxiter': max(1, remaining // 8),
                                                'gtol': 1e-8, 'ftol': 1e-10,
                                                'maxls': 50})
                        if np.isfinite(res.fun) and res.fun < self.best_f:
                            self.best_f = res.fun
                            self.best_x = np.clip(res.x, -5, 5).copy()
                except Exception:
                    pass

            # Phase 2: Coordinate descent for fine-tuning
            if self.calls < budget - 10 and time.time() < self.deadline - 5:
                try:
                    x = self.best_x.copy()
                    f = self.best_f
                    step = 0.5
                    for coord in range(dim):
                        if self.calls >= budget - 5:
                            break
                        if time.time() > self.deadline - 3:
                            break
                        for _ in range(15):
                            if self.calls >= budget - 5:
                                break
                            if time.time() > self.deadline - 3:
                                break
                            x_new = x.copy()
                            x_new[coord] = x[coord] + step
                            f_new = wrapped_func(x_new)
                            if np.isfinite(f_new) and f_new < f:
                                x = x_new
                                f = f_new
                            else:
                                x_new = x.copy()
                                x_new[coord] = x[coord] - step
                                f_new = wrapped_func(x_new)
                                if np.isfinite(f_new) and f_new < f:
                                    x = x_new
                                    f = f_new
                                else:
                                    step *= 0.5
                                    if step < 1e-4:
                                        break
                        if step < 1e-4:
                            break
                    if np.isfinite(f) and f < self.best_f:
                        self.best_f = f
                        self.best_x = np.clip(x, -5, 5).copy()
                except Exception:
                    pass

        # Final polish with custom Nelder-Mead around the best point
        if self.calls < budget - 5 and time.time() < self.deadline - 5:
            try:
                remaining = min(40, budget - self.calls - 3)
                if remaining > 10:
                    max_iter = max(1, remaining // 3)
                    x_opt, f_opt = nelder_mead(wrapped_func, self.best_x.copy(), max_iter=max_iter, tol=1e-12)
                    if np.isfinite(f_opt) and f_opt < self.best_f:
                        self.best_f = f_opt
                        self.best_x = np.clip(x_opt, -5, 5).copy()
            except Exception:
                pass

        return self.best_x, self.best_f
