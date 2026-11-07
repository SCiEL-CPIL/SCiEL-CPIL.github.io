"""Optimizer using Nelder-Mead with adaptive multi-start and CMA-ES-like restarts."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start Nelder-Mead with adaptive restarts for box-constrained minimization on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = [0]

        def wrapped_func(x):
            if time.time() > deadline:
                raise TimeoutError()
            if call_count[0] >= budget:
                raise TimeoutError()
            call_count[0] += 1
            val = float(func(x))
            return val

        def time_ok():
            return time.time() < deadline

        def budget_ok():
            return call_count[0] < budget - 5

        rng = np.random.default_rng(12345)

        # Phase 1: Structured coarse sampling to quickly find promising basins
        # For Rosenbrock, the basin is elongated, so we sample along the diagonal and random points
        n_coarse = 20
        points = []

        # Origin
        points.append(np.zeros(dim))
        # Points along the diagonal (common for Rosenbrock)
        for scale in [0.5, 1.0, 1.5, 2.0, 3.0]:
            points.append(np.full(dim, scale))
            points.append(np.full(dim, -scale))
        # Axis-aligned points
        for i in range(dim):
            for sign in [1, -1]:
                x = np.zeros(dim)
                x[i] = sign * 2.0
                points.append(x)
        # Random points
        while len(points) < n_coarse:
            points.append(rng.uniform(-4, 4, dim))

        for x0 in points:
            if not time_ok() or not budget_ok():
                break
            x0 = np.clip(x0, -5, 5)
            try:
                f0 = wrapped_func(x0)
                if f0 < best_f:
                    best_f = f0
                    best_x = x0.copy()
            except Exception:
                break

        # Phase 2: Nelder-Mead refinement with adaptive restarts
        # Nelder-Mead is derivative-free and works well for smooth functions like Rosenbrock
        # We use multiple restarts with varying initial simplex sizes

        def run_nelder_mead(x0, maxiter=200, initial_simplex_size=0.5):
            """Run Nelder-Mead from x0 with given parameters."""
            try:
                # Create initial simplex
                simplex = [x0.copy()]
                for i in range(dim):
                    xi = x0.copy()
                    # Perturb along each axis
                    if abs(x0[i]) < 1e-10:
                        xi[i] += initial_simplex_size
                    else:
                        xi[i] += initial_simplex_size * np.sign(x0[i])
                    simplex.append(xi)

                # Evaluate initial simplex
                simplex_f = []
                for xi in simplex:
                    xi = np.clip(xi, -5, 5)
                    fi = wrapped_func(xi)
                    simplex_f.append(fi)
                    if fi < best_f:
                        best_f = fi
                        best_x = xi.copy()

                # Sort by function value
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Nelder-Mead iterations
                alpha = 1.0  # reflection
                gamma = 2.0  # expansion
                rho = 0.5    # contraction
                sigma = 0.5  # shrink

                for iteration in range(maxiter):
                    if not time_ok() or not budget_ok():
                        break

                    # Compute centroid (excluding worst point)
                    centroid = np.mean(simplex[:-1], axis=0)

                    # Reflection
                    x_ref = centroid + alpha * (centroid - simplex[-1])
                    x_ref = np.clip(x_ref, -5, 5)
                    f_ref = wrapped_func(x_ref)

                    if f_ref < simplex_f[0]:
                        # Expansion
                        x_exp = centroid + gamma * (x_ref - centroid)
                        x_exp = np.clip(x_exp, -5, 5)
                        f_exp = wrapped_func(x_exp)

                        if f_exp < f_ref:
                            simplex[-1] = x_exp
                            simplex_f[-1] = f_exp
                        else:
                            simplex[-1] = x_ref
                            simplex_f[-1] = f_ref
                    elif f_ref < simplex_f[-2]:
                        # Accept reflection
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                    else:
                        # Contraction
                        if f_ref < simplex_f[-1]:
                            # Outer contraction
                            x_con = centroid + rho * (x_ref - centroid)
                        else:
                            # Inner contraction
                            x_con = centroid - rho * (simplex[-1] - centroid)
                        x_con = np.clip(x_con, -5, 5)
                        f_con = wrapped_func(x_con)

                        if f_con < simplex_f[-1]:
                            simplex[-1] = x_con
                            simplex_f[-1] = f_con
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                simplex_f[i] = wrapped_func(simplex[i])

                    # Update best
                    for i in range(dim + 1):
                        if simplex_f[i] < best_f:
                            best_f = simplex_f[i]
                            best_x = simplex[i].copy()

                    # Check convergence
                    f_range = max(simplex_f) - min(simplex_f)
                    if f_range < 1e-12:
                        break

                return best_f, best_x.copy()
            except Exception:
                return None, None

        # Run Nelder-Mead from the best point found so far
        if best_x is not None and time_ok() and budget_ok():
            f_new, x_new = run_nelder_mead(best_x, maxiter=300, initial_simplex_size=0.3)
            if x_new is not None:
                best_f = f_new
                best_x = x_new

        # Adaptive restarts: try multiple restarts with varying simplex sizes
        # For Rosenbrock, the basin is narrow, so we try both small and large simplex sizes
        restart_configs = [
            (0.1, 200),
            (0.2, 200),
            (0.5, 150),
            (1.0, 100),
            (2.0, 80),
        ]

        for simplex_size, maxiter in restart_configs:
            if not time_ok() or not budget_ok():
                break

            # Generate a few random restarts around the current best
            n_restarts = 3
            for r in range(n_restarts):
                if not time_ok() or not budget_ok():
                    break

                # Perturb the best point
                x0 = best_x + rng.normal(0, simplex_size, dim)
                x0 = np.clip(x0, -5, 5)

                f_new, x_new = run_nelder_mead(x0, maxiter=maxiter, initial_simplex_size=simplex_size * 0.5)
                if x_new is not None:
                    best_f = f_new
                    best_x = x_new

        # Phase 3: Final polish with very tight convergence
        if time_ok() and best_x is not None and budget_ok():
            f_new, x_new = run_nelder_mead(best_x, maxiter=100, initial_simplex_size=0.05)
            if x_new is not None:
                best_f = f_new
                best_x = x_new

        # Fallback if nothing was found
        if best_x is None:
            best_x = np.zeros(dim)
            if budget_ok():
                best_f = wrapped_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
