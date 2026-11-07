"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """L-BFGS-B based optimizer with multi-start and adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim
        self._calls = 0
        self._max_calls = budget

    def __call__(self, func):
        self._calls = 0
        self._max_calls = self.budget
        rng = np.random.default_rng(42)

        # Wrapper to track calls and enforce budget
        def wrapped_func(x):
            if self._calls >= self._max_calls:
                return float('inf')
            self._calls += 1
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            if not np.isfinite(val):
                return float('inf')
            return float(val)

        best_x = None
        best_f = float('inf')
        bounds = [(-5.0, 5.0)] * self.dim

        def try_optimize(x0, maxiter, ftol, gtol):
            nonlocal best_x, best_f
            if self._calls >= self._max_calls:
                return
            try:
                res = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
                )
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 1: Broad exploration with diverse starting points
        # Use a combination of grid-like and random points
        n_broad = 8
        for i in range(n_broad):
            if self._calls >= self._max_calls:
                break
            
            if i < 3:
                # Structured points: center, corners-ish
                if i == 0:
                    x0 = np.zeros(self.dim)
                elif i == 1:
                    x0 = rng.uniform(-4, 4, self.dim)
                else:
                    x0 = rng.uniform(-3, 3, self.dim)
            else:
                x0 = rng.uniform(-5, 5, self.dim)
            
            # Use moderate iterations for broad phase
            try_optimize(x0, maxiter=40, ftol=1e-10, gtol=1e-6)

        # Phase 2: Local refinement around best point with varied perturbations
        if best_x is not None and self._calls < self._max_calls:
            perturbations = [0.3, 0.5, 0.7, 1.0, 0.2, 0.4]
            for scale in perturbations:
                if self._calls >= self._max_calls:
                    break
                x0 = np.clip(best_x + scale * rng.standard_normal(self.dim), -5.0, 5.0)
                try_optimize(x0, maxiter=30, ftol=1e-12, gtol=1e-8)

        # Phase 3: Fine-tuning with very small perturbations
        if best_x is not None and self._calls < self._max_calls:
            for scale in [0.05, 0.1, 0.15, 0.08]:
                if self._calls >= self._max_calls:
                    break
                x0 = np.clip(best_x + scale * rng.standard_normal(self.dim), -5.0, 5.0)
                try_optimize(x0, maxiter=20, ftol=1e-14, gtol=1e-10)

        # Phase 4: If still have budget, try additional random restarts
        if best_x is not None and self._calls < self._max_calls:
            n_extra = 5
            for _ in range(n_extra):
                if self._calls >= self._max_calls:
                    break
                # Mix of random and perturbed best
                if rng.random() < 0.5:
                    x0 = rng.uniform(-5, 5, self.dim)
                else:
                    x0 = np.clip(best_x + 0.8 * rng.standard_normal(self.dim), -5.0, 5.0)
                try_optimize(x0, maxiter=25, ftol=1e-12, gtol=1e-8)

        # Final fallback
        if best_x is None:
            best_x = rng.uniform(-5, 5, self.dim)
            if self._calls < self._max_calls:
                self._calls += 1
                best_f = wrapped_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
