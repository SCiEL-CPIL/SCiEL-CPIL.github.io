"""Sobol quasi-random global search + adaptive Nelder-Mead local polish for BBOB f18."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Sobol-based global search with adaptive Nelder-Mead local optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls = 0

        def call_func(x):
            nonlocal calls, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.where(np.isfinite(x), x, 0.0)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Sobol sequence generation (using scipy if available, else fallback)
        try:
            from scipy.stats import qmc
            sobol = qmc.Sobol(d=dim, scramble=True)
            sobol_seed = 42
            sobol.seed(sobol_seed)
            has_sobol = True
        except ImportError:
            has_sobol = False
            rng = np.random.default_rng(42)

        def generate_point(idx):
            """Generate a point in [-4, 4]^d using Sobol or uniform."""
            if has_sobol:
                # Generate next Sobol point
                pt = sobol.random(1)[0]  # [0, 1)^d
                return pt * 8.0 - 4.0  # map to [-4, 4]
            else:
                return rng.uniform(-4, 4, dim)

        # Phase 1: Global search using Sobol quasi-random sequence
        n_global = min(200, budget)
        for i in range(n_global):
            if calls >= budget or time.time() > deadline:
                break
            x = generate_point(i)
            call_func(x)

        # Phase 2: Local optimization with adaptive Nelder-Mead from best point
        if best_x is not None and calls < budget and time.time() < deadline:
            x0 = best_x.copy()
            f0 = best_f

            # Nelder-Mead simplex initialization
            # Start with simplex around x0
            simplex = [x0.copy()]
            fvals = [f0]

            # Add vertices with adaptive initial step
            step = 0.5
            for j in range(dim):
                v_plus = x0.copy()
                v_plus[j] += step
                v_minus = x0.copy()
                v_minus[j] -= step

                # Choose the direction that's within bounds
                if v_plus[j] <= 5.0:
                    v = v_plus
                else:
                    v = v_minus

                fv = call_func(v)
                simplex.append(v)
                fvals.append(fv)

                if calls >= budget or time.time() > deadline:
                    break

            # Nelder-Mead iterations
            max_nm_iter = 300
            for it in range(max_nm_iter):
                if calls >= budget or time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Check convergence
                f_range = fvals[-1] - fvals[0]
                if f_range < 1e-12 * max(1.0, abs(fvals[0])):
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = call_func(x_ref)

                if calls >= budget or time.time() > deadline:
                    break

                if f_ref < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    f_exp = call_func(x_exp)

                    if calls >= budget or time.time() > deadline:
                        break

                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    # Contraction
                    rho = 0.5
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5.0, 5.0)
                    f_con = call_func(x_con)

                    if calls >= budget or time.time() > deadline:
                        break

                    if f_con < fvals[-1]:
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            if calls < budget and time.time() < deadline:
                                fvals[i] = call_func(simplex[i])
                            else:
                                break

        return best_x, best_f
# EVOLVE-BLOCK-END
