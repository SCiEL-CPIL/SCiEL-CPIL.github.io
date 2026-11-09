"""Rewritten optimizer for BBOB-style f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead based optimizer with adaptive ridge-following for BBOB f13."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 270.0  # safety margin before 300s kill
        n = self.dim
        budget = self.budget

        def elapsed():
            return time.time() - t0

        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            f = func(np.asarray(x, dtype=np.float64))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = np.asarray(x, dtype=np.float64).copy()
            return f

        # Phase 1: Sobol-based exploration to find promising regions
        try:
            from scipy.stats import qmc
            sobol = qmc.Sobol(d=n, scramble=True, seed=42)
            sobol.random(1)  # skip origin
            n_init = min(32, budget // 8)
            points = sobol.random(n_init) * 10.0 - 5.0
        except ImportError:
            rng = np.random.default_rng(42)
            n_init = min(32, budget // 8)
            points = rng.random((n_init, n)) * 10.0 - 5.0

        for i in range(n_init):
            if calls >= budget or elapsed() > deadline:
                break
            f = evaluate(points[i])

        # Phase 2: Nelder-Mead from best point with ridge-adaptive parameters
        if best_x is not None and calls < budget and elapsed() < deadline:
            self._nelder_mead(func, best_x.copy(), best_f, n, budget, calls, deadline, t0, evaluate)

        # Phase 3: If we have budget left, do more local searches from diverse points
        remaining = budget - calls
        if remaining > 20 and elapsed() < deadline:
            rng = np.random.default_rng(123)
            n_extra = min(5, remaining // 40)
            for s in range(n_extra):
                if calls >= budget or elapsed() > deadline:
                    break
                # Perturb around current best
                scale = 1.5 / (s + 1)
                x0 = best_x + rng.standard_normal(n) * scale
                x0 = np.clip(x0, -5.0, 5.0)
                f0 = evaluate(x0)
                if f0 < best_f:
                    best_f = f0
                    best_x = x0.copy()
                if calls < budget - 10 and elapsed() < deadline:
                    self._nelder_mead(func, x0.copy(), best_f, n, budget, calls, deadline, t0, evaluate)

        # Phase 4: Final high-precision polish with tight tolerances
        if calls < budget - 5 and elapsed() < deadline:
            self._nelder_mead(func, best_x.copy(), best_f, n, budget, calls, deadline, t0, evaluate,
                              max_iter=100, xatol=1e-10, fatol=1e-12, initial_simplex_size=0.01)

        return best_x, best_f

    def _nelder_mead(self, func, x_start, f_start, n, budget, calls, deadline, t0, evaluate,
                     max_iter=200, xatol=1e-8, fatol=1e-10, initial_simplex_size=0.5):
        """Custom Nelder-Mead with ridge-adaptive simplex management."""
        nonlocal best_x, best_f

        # Create initial simplex
        simplex = [x_start.copy()]
        f_values = [evaluate(x_start)]

        for i in range(n):
            xi = x_start.copy()
            xi[i] += initial_simplex_size
            simplex.append(xi)
            f_values.append(evaluate(xi))

        # Nelder-Mead parameters
        alpha = 0.5  # reflection
        gamma = 1.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink

        # Ridge-adaptive: track direction of improvement
        ridge_dir = np.zeros(n)
        ridge_magnitude = 0.0

        for iteration in range(max_iter):
            if calls >= budget or elapsed() > deadline:
                break

            # Sort simplex by function value
            order = np.argsort(f_values)
            simplex = [simplex[i] for i in order]
            f_values = [f_values[i] for i in order]

            # Check convergence
            f_range = f_values[-1] - f_values[0]
            if f_range < fatol:
                break
            x_range = np.max([np.max(np.abs(simplex[i] - simplex[0])) for i in range(len(simplex))])
            if x_range < xatol:
                break

            # Compute centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, -5.0, 5.0)
            f_ref = evaluate(x_ref)

            if f_ref < f_values[0]:
                # Try expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5.0, 5.0)
                f_exp = evaluate(x_exp)

                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    f_values[-1] = f_exp
                    # Track ridge direction
                    delta = x_exp - centroid
                    mag = np.linalg.norm(delta)
                    if mag > 1e-10:
                        ridge_dir = 0.8 * ridge_dir + 0.2 * (delta / mag)
                        ridge_magnitude = 0.8 * ridge_magnitude + 0.2 * mag
                else:
                    simplex[-1] = x_ref
                    f_values[-1] = f_ref
                    delta = x_ref - centroid
                    mag = np.linalg.norm(delta)
                    if mag > 1e-10:
                        ridge_dir = 0.8 * ridge_dir + 0.2 * (delta / mag)
                        ridge_magnitude = 0.8 * ridge_magnitude + 0.2 * mag
            elif f_ref < f_values[-2]:
                simplex[-1] = x_ref
                f_values[-1] = f_ref
            else:
                # Contraction
                if f_ref < f_values[-1]:
                    # Outside contraction
                    x_con = centroid + 0.5 * (x_ref - centroid)
                else:
                    # Inside contraction
                    x_con = centroid - 0.5 * (x_ref - centroid)
                x_con = np.clip(x_con, -5.0, 5.0)
                f_con = evaluate(x_con)

                if f_con <= f_ref:
                    simplex[-1] = x_con
                    f_values[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        f_values[i] = evaluate(simplex[i])

            # Update best
            min_idx = np.argmin(f_values)
            if f_values[min_idx] < best_f:
                best_f = f_values[min_idx]
                best_x = simplex[min_idx].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
