"""Rewritten optimizer for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-phase optimizer:
    Phase 1: Structured grid search for early AOCC progress
    Phase 2: Multiple Nelder-Mead restarts from diverse starting points
    Phase 3: CMA-ES refinement from the best point found
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls = 0

        def should_stop():
            return time.time() > deadline or calls >= budget

        def track(func_call, x):
            nonlocal calls, best_f, best_x
            fx = func_call(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(12345)

        # Phase 1: Structured grid search for early progress
        # Evaluate a coarse grid in the central region where the minimizer is likely
        # Use adaptive grid: start with center, then expand
        phase1_budget = int(budget * 0.15)

        # Center point
        center = np.zeros(dim)
        track(func, center)

        # Axis-aligned points at various distances
        for dist in [1.0, 2.0, 3.0, 4.0]:
            for d in range(dim):
                if should_stop():
                    break
                for sign in [1, -1]:
                    if should_stop():
                        break
                    x = center.copy()
                    x[d] = sign * dist
                    track(func, x)

        # Diagonal points
        for dist in [1.0, 2.0, 3.0]:
            if should_stop():
                break
            x = np.ones(dim) * dist
            track(func, x)
            x = -np.ones(dim) * dist
            track(func, x)

        # Random points in [-4, 4]^5
        n_random = min(phase1_budget - calls, 30)
        for _ in range(max(0, n_random)):
            if should_stop():
                break
            x = rng.uniform(-4, 4, dim)
            track(func, x)

        if should_stop() or best_x is None:
            return best_x, best_f

        # Phase 2: Multiple Nelder-Mead restarts
        # Start from best point and several diverse points
        phase2_budget = int(budget * 0.50)
        nm_budget_per_restart = max(20, phase2_budget // 6)

        # Starting points: best + 5 diverse points
        start_points = [best_x.copy()]

        # Add points near best
        for _ in range(3):
            x = best_x + 1.5 * rng.standard_normal(dim)
            x = np.clip(x, -5, 5)
            start_points.append(x)

        # Add random points
        for _ in range(2):
            x = rng.uniform(-4, 4, dim)
            start_points.append(x)

        for sp in start_points:
            if should_stop():
                break

            # Use Nelder-Mead with a wrapper that tracks calls
            nm_calls = [0]

            def nm_func(x):
                nonlocal calls, best_f, best_x
                nm_calls[0] += 1
                fx = func(x)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx

            # Initial simplex
            x0 = np.clip(sp, -5, 5)
            simplex = [x0.copy()]
            for d in range(dim):
                x = x0.copy()
                x[d] += 0.5
                x = np.clip(x, -5, 5)
                simplex.append(x)

            # Run Nelder-Mead manually to control budget
            # Simple implementation
            simplex = np.array(simplex)
            fvals = np.array([nm_func(s) for s in simplex])

            max_nm_iters = nm_budget_per_restart - (dim + 1)
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink

            for it in range(max_nm_iters):
                if should_stop():
                    break

                # Sort by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Centroid of all but worst
                centroid = simplex[:-1].mean(axis=0)

                # Reflect
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = nm_func(x_ref)

                if f_ref < fvals[0]:
                    # Expand
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = nm_func(x_exp)
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
                    # Contract
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = nm_func(x_con)
                    if f_con < fvals[-1]:
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = nm_func(simplex[i])

        if should_stop():
            return best_x, best_f

        # Phase 3: CMA-ES refinement from best point
        remaining = budget - calls
        if remaining < 20:
            return best_x, best_f

        # CMA-ES parameters
        pop_size = 4 + int(np.floor(3 * np.log(dim)))  # ~11 for dim=5
        mu = pop_size // 2
        weights = np.log(mu + 1) - np.arange(mu)
        weights = weights / weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 1) ** 2 + 2)
        cmu = min(1 - c1, (2 * (mu_eff - 2 + 2.0 / (mu_eff + 1.3))) / ((dim + 2) ** 2 + 2))
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1)
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # Initialize CMA-ES state
        x_mean = best_x.copy()
        sigma = 0.5
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        max_gens = remaining // pop_size

        for gen in range(max_gens):
            if should_stop():
                break

            # Generate population
            x_pop = np.zeros((pop_size, dim))
            f_pop = np.zeros(pop_size)

            for i in range(pop_size):
                if should_stop():
                    break
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                x_pop[i] = y
                f_pop[i] = func(y)
                calls += 1
                if f_pop[i] < best_f:
                    best_f = f_pop[i]
                    best_x = x_pop[i].copy()

            # Sort by fitness
            order = np.argsort(f_pop)
            f_sorted = f_pop[order]
            x_sorted = x_pop[order]

            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = x_sorted[0].copy()

            # Compute new mean
            x_new_mean = np.sum(weights[:, None] * x_sorted[:mu], axis=0)

            # Update step size
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (B @ (D * (x_new_mean - x_mean)))
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 20.0)

            # Update evolution path
            p_c = (1 - cs) * p_c + h_sigma * np.sqrt(cs * (2 - cs) * mu_eff) * (x_new_mean - x_mean) / sigma

            # Update covariance matrix
            C = (1 - c1 - cmu) * C
            C += c1 * (np.outer(p_c, p_c) + (1 - h_sigma) * cs * (2 - cs) * C)
            for i in range(mu):
                diff = (x_sorted[i] - x_mean) / sigma
                C += cmu * weights[i] * np.outer(diff, diff)

            # Ensure positive semi-definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update B and D
            B, D = eigvecs, np.sqrt(eigvals)

            # Update mean
            x_mean = x_new_mean

        return best_x, best_f
# EVOLVE-BLOCK-END
