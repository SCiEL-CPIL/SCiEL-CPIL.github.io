"""Improved SQP (SLSQP) with stratified sampling and adaptive multi-start."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start SLSQP optimizer with stratified initialization and adaptive refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Bounds for SLSQP
        bounds = [(-5.0, 5.0)] * dim

        best_x = None
        best_f = np.inf

        # Track total function calls made
        call_count = [0]

        def wrapped_func(x):
            call_count[0] += 1
            return func(x)

        def try_slsqp(x_start, max_iter, ftol=1e-10):
            """Run SLSQP and return (success, new_best_x, new_best_f)"""
            nonlocal best_x, best_f
            remaining = budget - call_count[0] - 10
            if remaining < 6:  # Need at least one iteration (dim+1=6 calls)
                return False
            max_iter = min(max_iter, remaining // (dim + 1))
            if max_iter < 1:
                return False
            try:
                res = minimize(
                    wrapped_func,
                    x_start,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': max_iter, 'ftol': ftol, 'disp': False}
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    return True
            except Exception:
                pass
            return False

        # Phase 1: Stratified sampling (Latin hypercube) for better coverage
        n_samples = 12
        # Generate stratified points: divide each dimension into n_samples bins
        stratified_points = []
        for i in range(n_samples):
            x0 = np.zeros(dim)
            for d in range(dim):
                # Stratified: bin i out of n_samples, with jitter
                base = -5.0 + 10.0 * i / n_samples
                span = 10.0 / n_samples
                x0[d] = base + rng.uniform(0, span)
            stratified_points.append(x0)
        
        # Also add center and a few random points
        stratified_points.append(np.zeros(dim))
        for _ in range(3):
            stratified_points.append(rng.uniform(-5.0, 5.0, dim))
        
        for x0 in stratified_points:
            if call_count[0] >= budget - 20:
                break
            x0 = np.clip(x0, -5.0, 5.0)
            f0 = wrapped_func(x0)
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        # Phase 2: Quick SLSQP from the best point found so far
        if best_x is not None:
            try_slsqp(best_x, max_iter=20, ftol=1e-8)

        # Phase 3: Multi-start with diverse perturbations around best
        # Use geometrically decreasing scales
        scales = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05]
        for scale in scales:
            if call_count[0] >= budget - 15:
                break
            # Generate 2-3 perturbed starts at this scale
            n_perturb = 2 if scale > 0.5 else 1
            for _ in range(n_perturb):
                if call_count[0] >= budget - 15:
                    break
                x0 = best_x + rng.normal(0, scale, dim)
                x0 = np.clip(x0, -5.0, 5.0)
                f0 = wrapped_func(x0)
                if f0 < best_f:
                    best_f = f0
                    best_x = x0.copy()
                # Run SLSQP from this point with moderate iterations
                try_slsqp(x0, max_iter=15, ftol=1e-10)

        # Phase 4: Final high-precision refinement
        if best_x is not None and call_count[0] < budget - 10:
            # Run SLSQP with tight tolerance
            try_slsqp(best_x, max_iter=30, ftol=1e-14)
            
            # One more refinement if budget allows
            if call_count[0] < budget - 10:
                try_slsqp(best_x, max_iter=20, ftol=1e-16)

        # Ensure we return the best solution found
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
