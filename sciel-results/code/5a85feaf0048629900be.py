"""Direct solver for BBOB f9 (rotated Rosenbrock) exploiting valley structure."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution


class Optimizer:
    """Multi-start L-BFGS-B with Rosenbrock-aware start points and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 270.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim

        call_count = [0]
        best_x = [None]
        best_f = [np.inf]

        def counted_func(x):
            call_count[0] += 1
            if call_count[0] > budget:
                return best_f[0]
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            if f < best_f[0]:
                best_f[0] = f
                best_x[0] = x_clipped.copy()
            return f

        # Phase 1: Differential Evolution global search
        # DE uses a population-based approach which is robust to the unknown rotation/shift
        # of the Rosenbrock valley. We limit the max iterations and use a small population
        # to stay within the budget.
        
        # Estimate max iterations for DE. Each iteration uses popsize evaluations.
        # popsize=15 (default is 15*dim), so each iter uses ~75 evals.
        # Budget is 500. Let's use ~300 for DE, leaving 200 for polish.
        # 300 / 75 = 4 iterations. That's very few.
        # Let's reduce popsize to 10. 10 * 5 = 50 evals/iter.
        # 300 / 50 = 6 iterations. Still few.
        # Maybe we should just let DE run with a tight budget.
        
        # Alternatively, we can use a smaller population and more iterations if we control it.
        # But differential_evolution doesn't expose population size directly in all versions easily?
        # Actually, it does: popsize parameter.
        
        # Let's try popsize=8. 8*5=40 evals/iter.
        # Budget 250 for DE -> ~6 iterations.
        # This might be too few for DE to converge globally.
        
        # Let's try a different approach: use DE with a very tight maxiter and rely on the polish.
        # Or, use a smaller budget for DE and more for polish?
        # The valley is narrow. DE might miss it if population is small.
        
        # Let's try popsize=15 (default) and maxiter=5. 5 * 75 = 375 evals.
        # Leaves 125 for polish.
        
        try:
            de_result = differential_evolution(
                counted_func,
                bounds,
                maxiter=5,
                popsize=15,
                seed=42,
                tol=1e-6,
                polish=False  # We will do our own polish
            )
            if de_result.fun < best_f[0]:
                best_f[0] = de_result.fun
                best_x[0] = np.clip(de_result.x, -5, 5)
        except Exception:
            pass

        # Ensure we have a best point
        if best_x[0] is None:
            best_x[0] = np.zeros(dim)
            best_f[0] = counted_func(best_x[0])

        # Phase 2: Local optimization with L-BFGS-B
        # Start from the best point found by DE.
        starts_for_local = [best_x[0].copy()]
        
        # Add a few perturbations to handle potential local minima or DE convergence to a suboptimal point
        rng = np.random.default_rng(123)
        for _ in range(2):
            perturbed = best_x[0] + rng.normal(0, 1.0, dim)
            starts_for_local.append(np.clip(perturbed, -5, 5))

        for x0 in starts_for_local:
            if call_count[0] >= budget - 50:
                break
            if time.time() > deadline - 20.0:
                break

            try:
                res = minimize(
                    counted_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res.fun < best_f[0]:
                    best_f[0] = res.fun
                    best_x[0] = np.clip(res.x, -5, 5)
            except Exception:
                continue

        # Phase 3: Final high-precision polish
        if best_x[0] is not None and call_count[0] < budget - 10 and time.time() < deadline - 5.0:
            try:
                res = minimize(
                    counted_func,
                    best_x[0],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                if res.fun < best_f[0]:
                    best_f[0] = res.fun
                    best_x[0] = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Ensure we have a valid result
        if best_x[0] is None:
            best_x[0] = np.zeros(dim)
            if call_count[0] < budget:
                best_f[0] = counted_func(best_x[0])
            else:
                best_f[0] = 0.0

        return best_x[0], best_f[0]
