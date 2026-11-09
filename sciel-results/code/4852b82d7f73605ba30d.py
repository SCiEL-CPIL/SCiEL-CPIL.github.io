"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize


class Optimizer:
    """Differential Evolution + Nelder-Mead polish for BBOB f15."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Use a subset of budget for DE, rest for polishing
        de_budget = int(budget * 0.7)
        polish_budget = budget - de_budget

        def bounded_func(x):
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            return func(x)

        # Differential Evolution with moderate population
        # maxiter controls number of generations; popsize controls population size
        # Total evaluations ≈ popsize * dim * maxiter
        popsize = 8
        # Estimate maxiter: each generation uses popsize * dim evaluations roughly
        # But DE actually uses popsize evaluations per generation (one new candidate per member)
        # So total evals ≈ popsize * maxiter
        # We want to use about de_budget evaluations
        maxiter = max(1, de_budget // popsize)

        try:
            result = differential_evolution(
                bounded_func,
                bounds=[(-5.0, 5.0)] * dim,
                maxiter=maxiter,
                popsize=popsize,
                seed=42,
                tol=1e-8,
                atol=1e-8,
                polish=False,
                init='sobol',
                updating='immediate',
                workers=1
            )
            x_best = result.x
            f_best = result.fun
        except Exception:
            # Fallback: simple random search + local refinement
            x_best = rng.uniform(-5, 5, dim)
            f_best = bounded_func(x_best)

        # Polish with Nelder-Mead using remaining budget
        if polish_budget > 5:
            try:
                nm_result = minimize(
                    bounded_func,
                    x_best,
                    method='Nelder-Mead',
                    options={
                        'maxiter': polish_budget,
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if nm_result.fun < f_best:
                    x_best = nm_result.x
                    f_best = nm_result.fun
            except Exception:
                pass

        return x_best, f_best
# EVOLVE-BLOCK-END
