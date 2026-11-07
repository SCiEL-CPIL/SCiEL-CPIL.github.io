"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


class Optimizer:
    """Differential Evolution optimizer with careful budget management."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim
        self._call_count = 0

    def __call__(self, func):
        self._call_count = 0
        dim = self.dim
        budget = self.budget

        # Counting wrapper
        def wrapped(x):
            if self._call_count >= budget:
                return 1e300
            self._call_count += 1
            try:
                return float(func(x))
            except Exception:
                return 1e300

        bounds = [(-5.0, 5.0)] * dim

        # Use differential evolution with conservative settings
        # popsize=10, maxiter large enough, but we cap via budget
        # tol=1e-8, mutation=(0.5, 1.0), recombination=0.9
        # seed fixed for reproducibility per instance (but instances have hidden shifts)
        try:
            result = differential_evolution(
                wrapped,
                bounds,
                maxiter=2000,
                popsize=10,
                tol=1e-10,
                mutation=(0.5, 1.0),
                recombination=0.9,
                seed=42,
                init='sobol',
                polish=False,
            )
        except Exception:
            pass

        return None
# EVOLVE-BLOCK-END
