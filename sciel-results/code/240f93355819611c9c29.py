"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Adaptive multi-start L-BFGS-B optimizer with global exploration."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made
            x = np.clip(np.asarray(x, dtype=float).reshape(-1), -5, 5)
            fx = float(func(x))
            calls_made += 1
            return fx

        # Phase 1: Global exploration with diverse starting points
        # Use a grid-like sampling plus random points to cover the space
        n_global = min(30, budget // 10)
        for i in range(n_global):
            if calls_made >= budget:
                break
            # Mix of grid and random points
            if i < 5:
                # Corner and center points
                if i == 0:
                    x = np.zeros(n)
                elif i == 1:
                    x = np.full(n, -3.0)
                elif i == 2:
                    x = np.full(n, 3.0)
                elif i == 3:
                    x = rng.uniform(-4, 4, n)
                else:
                    x = rng.uniform(-5, 5, n)
            else:
                x = rng.uniform(-5, 5, n)
            
            fx = evaluate(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Local refinement using L-BFGS-B from best points
        # Collect top diverse starting points for local optimization
        if best_x is not None and calls_made < budget:
            # Use scipy's L-BFGS-B for local optimization
            try:
                from scipy.optimize import minimize
            except ImportError:
                # Fallback: simple Nelder-Mead if scipy not available
                from scipy.optimize import minimize

            # Try local optimization from the best point
            if calls_made < budget:
                try:
                    result = minimize(
                        lambda x: evaluate(x),
                        best_x.copy(),
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * n,
                        options={'maxiter': 50, 'maxls': 10, 'maxfun': budget - calls_made}
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5, 5).copy()
                except Exception:
                    pass

        # Phase 3: If budget remains, do more local optimizations from diverse points
        if calls_made < budget and best_x is not None:
            try:
                from scipy.optimize import minimize
            except ImportError:
                minimize = None

            if minimize is not None:
                # Generate a few more diverse starting points near the best solution
                n_extra = min(5, (budget - calls_made) // 10)
                for i in range(n_extra):
                    if calls_made >= budget:
                        break
                    # Perturb the best solution
                    perturbation_scale = 1.0 * (1 - i / max(1, n_extra))
                    x_start = best_x + rng.normal(0, perturbation_scale, n)
                    x_start = np.clip(x_start, -5, 5)
                    
                    try:
                        result = minimize(
                            lambda x: evaluate(x),
                            x_start,
                            method='L-BFGS-B',
                            bounds=[(-5, 5)] * n,
                            options={'maxiter': 30, 'maxls': 5, 'maxfun': max(10, budget - calls_made)}
                        )
                        if result.fun < best_f:
                            best_f = result.fun
                            best_x = np.clip(result.x, -5, 5).copy()
                    except Exception:
                        pass

        # Phase 4: Final fine-tuning with small random perturbations
        if calls_made < budget and best_x is not None:
            remaining = budget - calls_made
            n_fine = min(remaining, 20)
            for i in range(n_fine):
                if calls_made >= budget:
                    break
                # Very small perturbations around best
                scale = 0.1 * (1 - i / max(1, n_fine)) + 0.01
                x_try = best_x + rng.normal(0, scale, n)
                x_try = np.clip(x_try, -5, 5)
                fx = evaluate(x_try)
                if fx < best_f:
                    best_f = fx
                    best_x = x_try.copy()

        if best_x is None:
            best_x = np.zeros(n)
            best_f = 0.0

        return best_x, best_f
# EVOLVE-BLOCK-END
