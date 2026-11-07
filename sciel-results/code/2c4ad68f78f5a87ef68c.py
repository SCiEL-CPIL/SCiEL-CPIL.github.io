import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325  # 5s margin before 330s kill

        best_x = None
        best_f = np.inf
        calls = 0
        dim = self.dim
        budget = self.budget

        def wrapper(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                return 1e9
            if calls >= budget:
                return 1e9
            calls += 1
            x = np.clip(x, -5, 5)
            try:
                f = float(func(x))
            except:
                f = 1e9

            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 0: Sobol quasi-random sampling for rapid early coverage
        # This gives fast initial AOCC by uniformly covering the space
        try:
            from scipy.stats import qmc
            sobol = qmc.Sobol(d=dim, seed=42)
            # Use 100 Sobol points for initial exploration
            n_sobol = min(100, budget)
            sobol_points = sobol.random(n_sobol)
            # Map from [0,1]^dim to [-5,5]^dim
            sobol_points = sobol_points * 10.0 - 5.0

            for pt in sobol_points:
                if calls >= budget or time.time() > deadline:
                    break
                wrapper(pt)
        except ImportError:
            # Fallback: random sampling if scipy.stats.qmc not available
            for _ in range(min(100, budget)):
                if calls >= budget or time.time() > deadline:
                    break
                x0 = np.random.uniform(-5, 5, dim)
                wrapper(x0)

        # Phase 1: Differential Evolution for global search
        remaining_budget = budget - calls
        if remaining_budget > 30 and best_x is not None:
            de_budget = int(remaining_budget * 0.6)
            popsize = 20
            maxiter = max(1, (de_budget // popsize) - 1)
            bounds = [(-5, 5) for _ in range(dim)]

            try:
                res_de = differential_evolution(
                    wrapper, bounds,
                    popsize=popsize,
                    maxiter=maxiter,
                    tol=1e-6,
                    seed=42,
                    init='sobol' if 'sobol' in str(differential_evolution.__init__.__doc__ or '') else 'latinhypercube'
                )
            except:
                res_de = None

        # Phase 2: Nelder-Mead refinement from best point found
        remaining_budget = budget - calls
        if remaining_budget > 10 and best_x is not None:
            nm_budget = remaining_budget
            nm_maxiter = max(1, nm_budget // 6)

            try:
                res_nm = minimize(
                    wrapper, best_x, method='Nelder-Mead',
                    options={'maxiter': nm_maxiter, 'xatol': 1e-10, 'fatol': 1e-12}
                )
            except:
                res_nm = None

        # Phase 3: Final polish with tighter Nelder-Mead if budget remains
        remaining_budget = budget - calls
        if remaining_budget > 5 and best_x is not None:
            polish_maxiter = max(1, remaining_budget // 6)

            try:
                res_polish = minimize(
                    wrapper, best_x, method='Nelder-Mead',
                    options={'maxiter': polish_maxiter, 'xatol': 1e-12, 'fatol': 1e-14}
                )
            except:
                res_polish = None

        return best_x, best_f
