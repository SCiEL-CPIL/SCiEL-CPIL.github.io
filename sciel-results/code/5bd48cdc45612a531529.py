"""Rewritten optimizer for BBOB-style f3 (Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution


class Optimizer:
    """Global search via differential evolution + local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        # Wrap func to count calls and enforce deadline
        call_count = [0]
        max_calls = self.budget

        def wrapped(x):
            call_count[0] += 1
            if call_count[0] > max_calls:
                return 1e300
            # Check deadline
            if time.time() > deadline:
                return 1e300
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            return float(func(x))

        # Use differential evolution for global exploration
        # Budget allocation: ~80% for DE, ~20% for local refinement
        de_budget = int(self.budget * 0.75)
        local_budget = self.budget - de_budget

        # Configure DE: use a moderate population for 5D problem
        # popsize=15 means 75 candidates per generation
        # maxiter = de_budget // (popsize * dim) roughly
        # But DE's maxfevs is more direct
        popsize = 15
        # Each generation evaluates popsize * dim candidates roughly
        # maxfevs controls total function evaluations
        result = differential_evolution(
            wrapped,
            bounds=[(-5.0, 5.0)] * self.dim,
            maxiter=1000,
            maxfevs=de_budget,
            popsize=popsize,
            tol=1e-10,
            atol=1e-12,
            init='sobol',
            seed=42,
            polish=False,  # We'll do our own polish
            update_interval=1,
            callback=None,
            disp=False,
            init_kwargs={'n_skip': 0},
        )

        best_x = np.clip(result.x, -5.0, 5.0)
        best_f = wrapped(best_x)

        # Local refinement: coordinate-wise line search + Nelder-Mead style
        # Use remaining budget for local polish
        remaining = max_calls - call_count[0]
        if remaining > 10:
            # Simple local refinement: try small perturbations
            # Use a shrinking step size
            step = 0.5
            for _ in range(min(remaining, 200)):
                if time.time() > deadline:
                    break
                improved = False
                for d in range(self.dim):
                    for sign in [1, -1]:
                        trial = best_x.copy()
                        trial[d] = np.clip(trial[d] + sign * step, -5.0, 5.0)
                        ft = wrapped(trial)
                        if ft < best_f:
                            best_x = trial
                            best_f = ft
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
