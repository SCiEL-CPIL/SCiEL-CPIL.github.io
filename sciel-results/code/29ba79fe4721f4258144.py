"""L-BFGS-B multi-start with dense Hessian polish for Rosenbrock-rotated f9."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, L_BFGS_B


class Optimizer:
    """Multi-start L-BFGS-B with dense Hessian BFGS polish.

    Strategy:
      - Use L-BFGS-B from many diverse starting points (L-BFGS-B is a quasi-Newton
        method that follows the valley curvature well, unlike CMA-ES which samples
        isotropically).
      - After L-BFGS-B converges, apply a dense Hessian BFGS polish for quadratic
        convergence in the high-curvature region.
      - Budget: ~20 L-BFGS-B starts (each ~20-30 evals) + Hessian polish on the
        best few results.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270  # safety margin before 300s limit

        best_x = None
        best_f = np.inf
        call_count = 0

        # Track all good solutions for polishing
        good_solutions = []

        rng = np.random.default_rng(12345)

        # Generate diverse starting points
        # Include: center, corners of a sub-box, random points, and points along axes
        starts = []
        starts.append(np.zeros(dim))
        starts.append(np.full(dim, 1.0))
        starts.append(np.full(dim, -1.0))
        starts.append(np.full(dim, 2.0))
        starts.append(np.full(dim, -2.0))
        starts.append(np.full(dim, 3.0))
        starts.append(np.full(dim, -3.0))

        # Random points spread across the box
        for _ in range(12):
            starts.append(rng.uniform(-4, 4, dim))

        # Points near the typical Rosenbrock minimum region (before rotation/shift)
        starts.append(np.ones(dim) * 1.0)
        starts.append(np.ones(dim) * 0.5)

        # Limit total L-BFGS-B starts to fit budget
        # Each L-BFGS-B call uses ~20-40 evaluations for dim=5
        max_starts = 20
        starts = starts[:max_starts]

        for i, x0 in enumerate(starts):
            if time.time() > deadline - 30:
                break
            if call_count >= budget - 50:  # reserve calls for polish
                break

            # Check if we've already evaluated something close to this start
            if best_x is not None:
                if np.linalg.norm(x0 - best_x) < 0.1:
                    continue

            def objective(x):
                nonlocal call_count
                if call_count >= budget:
                    return np.inf
                x_clipped = np.clip(x, -5, 5)
                call_count += 1
                return func(x_clipped)

            # Run L-BFGS-B
            try:
                res = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'maxfun': 60, 'gtol': 1e-4}
                )
            except Exception:
                continue

            if res.success or res.fun < best_f:
                x_sol = np.clip(res.x, -5, 5)
                f_sol = res.fun
                if f_sol < best_f:
                    best_f = f_sol
                    best_x = x_sol.copy()
                # Keep top solutions for polishing
                good_solutions.append((f_sol, x_sol.copy()))

        # Sort good solutions and polish the best few
        good_solutions.sort(key=lambda t: t[0])
        n_polish = min(5, len(good_solutions))

        for f_init, x_init in good_solutions[:n_polish]:
            if time.time() > deadline - 10:
                break
            if call_count >= budget - 10:
                break

            # Dense Hessian BFGS polish
            def obj_polish(x):
                nonlocal call_count
                if call_count >= budget:
                    return np.inf
                x_clipped = np.clip(x, -5, 5)
                call_count += 1
                return func(x_clipped)

            try:
                res2 = minimize(
                    obj_polish,
                    x_init,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 30, 'maxfun': 40, 'gtol': 1e-6, 'ftol': 1e-10}
                )
            except Exception:
                continue

            if res2.fun < best_f:
                best_f = res2.fun
                best_x = np.clip(res2.x, -5, 5).copy()

        # If we haven't made any calls yet (shouldn't happen), make one
        if call_count == 0:
            x0 = np.zeros(dim)
            call_count += 1
            best_f = func(x0)
            best_x = x0.copy()

        return best_x, best_f


# EVOLVE-BLOCK-END
