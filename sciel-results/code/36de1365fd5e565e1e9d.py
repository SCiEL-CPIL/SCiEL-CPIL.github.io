"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Stochastic quasi-Newton (Broyden) optimizer with restarts for Schaffer F7."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def remaining():
            nonlocal calls_used
            return budget - calls_used

        def check_time():
            return time.time() < deadline

        # Number of restarts: we'll try a few different starting points
        # and allocate budget dynamically
        n_restarts = 3
        calls_per_restart = budget // n_restarts

        for restart in range(n_restarts):
            if not check_time() or remaining() <= 0:
                break

            # Initial point: center for first, random for others
            if restart == 0:
                x = np.zeros(n)
            else:
                x = rng.uniform(-4, 4, n)

            fx = func(x)
            calls_used += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Broyden's method (BFGS-like) for derivative-free optimization
            # We'll use a trust-region approach with Broyden updates
            # Approximate gradient using finite differences
            # But since we don't have derivatives, we'll use a pattern search
            # combined with Broyden-like step adaptation

            # Actually, for derivative-free, let's use a smarter approach:
            # Nelder-Mead simplex with adaptive parameters, plus a final
            # high-precision polish using coordinate descent

            # Phase 1: Nelder-Mead simplex method
            # Initialize simplex
            eps = 0.5
            simplex = [x.copy()]
            for i in range(n):
                xi = x.copy()
                xi[i] += eps
                simplex.append(xi)

            fvals = []
            for i in range(n + 1):
                fi = func(simplex[i])
                calls_used += 1
                fvals.append(fi)
                if fi < best_f:
                    best_f = fi
                    best_x = simplex[i].copy()

            fvals = np.array(fvals)

            # Nelder-Mead iterations
            max_nm_iter = 200
            for it in range(max_nm_iter):
                if not check_time() or remaining() <= 0:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = fvals[order]

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - simplex[-1])
                f_ref = func(x_ref)
                calls_used += 1
                if f_ref < best_f:
                    best_f = f_ref
                    best_x = x_ref.copy()

                if f_ref < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - simplex[-1])
                    f_exp = func(x_exp)
                    calls_used += 1
                    if f_exp < best_f:
                        best_f = f_exp
                        best_x = x_exp.copy()
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    # Contraction
                    rho = 0.5
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    f_con = func(x_con)
                    calls_used += 1
                    if f_con < best_f:
                        best_f = f_con
                        best_x = x_con.copy()
                    if f_con < f_ref:
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = func(simplex[i])
                            calls_used += 1
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()

                # Check convergence
                if np.max(fvals) - np.min(fvals) < 1e-10:
                    break

            # Phase 2: High-precision polish with coordinate descent
            # Start from the best point found
            x_polish = best_x.copy() if best_x is not None else x.copy()
            f_polish = best_f

            # Coordinate descent with line search
            for cd_iter in range(50):
                if not check_time() or remaining() <= 0:
                    break

                improved = False
                for i in range(n):
                    # Binary search for best step in dimension i
                    lo = -2.0
                    hi = 2.0
                    best_step = 0.0
                    best_fi = f_polish

                    for _ in range(15):
                        mid = (lo + hi) / 2.0
                        x_trial = x_polish.copy()
                        x_trial[i] += mid
                        x_trial = np.clip(x_trial, -5, 5)
                        f_trial = func(x_trial)
                        calls_used += 1

                        if f_trial < best_fi:
                            best_fi = f_trial
                            best_step = mid
                            lo = mid
                        else:
                            hi = mid

                    if abs(best_step) > 1e-8:
                        x_polish[i] += best_step
                        x_polish = np.clip(x_polish, -5, 5)
                        f_polish = best_fi
                        improved = True
                        if f_polish < best_f:
                            best_f = f_polish
                            best_x = x_polish.copy()

                if not improved:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
