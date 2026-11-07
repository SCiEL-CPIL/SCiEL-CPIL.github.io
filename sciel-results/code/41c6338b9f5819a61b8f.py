import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 300 - 30  # 270s safety margin

        calls_made = 0
        best_x = np.random.default_rng(123).uniform(-5, 5, self.dim)
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def wrapper(x):
            nonlocal calls_made, best_x, best_f
            if time.time() - start_time > time_limit:
                return best_f
            if calls_made >= self.budget:
                return best_f
            calls_made += 1
            try:
                f_val = func(x)
            except Exception:
                return best_f
            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x, dtype=float)
            return f_val

        # Phase 1: Global Search with Differential Evolution
        # Use a larger population and more iterations for better global coverage
        try:
            res_de = differential_evolution(
                wrapper,
                bounds=[(-5, 5)] * self.dim,
                popsize=16,
                maxiter=30,
                seed=42,
                tol=1e-8,
                init='latinhypercube',
                mutation=(0.4, 1.0),
                recombination=0.7,
                update_interval=1
            )
            if res_de.fun < best_f:
                best_f = res_de.fun
                best_x = np.array(res_de.x, dtype=float)
        except Exception:
            pass

        # Phase 2: Local Polish with L-BFGS-B
        # Tighter convergence criteria
        try:
            res_lbfgsb = minimize(
                wrapper,
                best_x,
                method='L-BFGS-B',
                bounds=[(-5, 5)] * self.dim,
                options={'maxiter': 200, 'gtol': 1e-14, 'ftol': 1e-14}
            )
            if res_lbfgsb.fun < best_f:
                best_f = res_lbfgsb.fun
                best_x = np.array(res_lbfgsb.x, dtype=float)
        except Exception:
            pass

        # Phase 3: Multi-start Nelder-Mead refinement
        # More diverse starting points to escape local minima
        if calls_made < self.budget - 50:
            starts = [best_x.copy()]
            
            # Add perturbations at multiple scales
            for scale in [0.1, 0.3, 0.8, 1.5, 2.5]:
                perturb = best_x + rng.normal(0, scale, self.dim)
                perturb = np.clip(perturb, -5, 5)
                starts.append(perturb)
            
            # Add a few fully random points
            for _ in range(3):
                random_pt = rng.uniform(-5, 5, self.dim)
                starts.append(random_pt)

            for start in starts:
                if calls_made >= self.budget - 10:
                    break
                if time.time() - start_time > time_limit - 15:
                    break
                try:
                    res_nm = minimize(
                        wrapper,
                        start,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 100,
                            'xatol': 1e-12,
                            'fatol': 1e-14,
                            'adaptive': True
                        }
                    )
                    if res_nm.fun < best_f:
                        best_f = res_nm.fun
                        best_x = np.array(res_nm.x, dtype=float)
                except Exception:
                    pass

        # Phase 4: Final polish with L-BFGS-B if budget allows
        if calls_made < self.budget - 5 and time.time() - start_time < time_limit - 5:
            try:
                res_final = minimize(
                    wrapper,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 50, 'gtol': 1e-15, 'ftol': 1e-15}
                )
                if res_final.fun < best_f:
                    best_f = res_final.fun
                    best_x = np.array(res_final.x, dtype=float)
            except Exception:
                pass

        return best_x, best_f
