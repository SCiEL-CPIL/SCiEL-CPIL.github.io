import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280
        dim = self.dim
        budget = self.budget
        calls = [0]
        best_x = None
        best_f = np.inf

        def wrapped(x):
            calls[0] += 1
            if calls[0] > budget:
                raise RuntimeError("budget exceeded")
            if time.time() > deadline:
                raise RuntimeError("time exceeded")
            return float(func(x))

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        rng = np.random.default_rng(42)

        # 1. Initial diverse starts
        starts = [np.zeros(dim)]
        for _ in range(5):
            starts.append(rng.uniform(-5, 5, dim))
        
        for s in starts:
            if calls[0] >= budget or time.time() > deadline:
                break
            try:
                f = wrapped(s)
                update_best(s, f)
            except Exception:
                pass

        if best_x is None:
            best_x = rng.uniform(-5, 5, dim)
            best_f = wrapped(best_x)

        # 2. Global Search: Differential Evolution
        # Use 60% of remaining budget
        remaining_budget = budget - calls[0]
        de_calls = int(remaining_budget * 0.6)
        
        if de_calls > 20 and time.time() < deadline:
            try:
                popsize = 10
                max_iter = max(1, de_calls // popsize)
                res_de = differential_evolution(
                    wrapped,
                    bounds=[(-5, 5)] * dim,
                    maxiter=max_iter,
                    popsize=popsize,
                    tol=1e-8,
                    seed=42,
                    polish=False,
                    init='sobol'
                )
                if res_de.fun < best_f:
                    update_best(res_de.x, res_de.fun)
            except Exception:
                pass

        # 3. Local Polish: L-BFGS-B
        if calls[0] < budget and time.time() < deadline:
            try:
                res_lbfgs = minimize(
                    wrapped,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 30, 'maxls': 15}
                )
                update_best(res_lbfgs.x, res_lbfgs.fun)
            except Exception:
                pass

        # 4. Local Polish: Nelder-Mead
        if calls[0] < budget and time.time() < deadline:
            try:
                res_nm = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-12}
                )
                update_best(res_nm.x, res_nm.fun)
            except Exception:
                pass

        return best_x, best_f
