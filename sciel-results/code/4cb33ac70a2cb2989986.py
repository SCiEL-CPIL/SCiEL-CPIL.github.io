"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """
    Multi-strategy optimizer combining:
    1. Latin hypercube sampling for initial exploration
    2. Nelder-Mead local refinement from multiple starting points
    3. CMA-ES for global optimization
    4. Final polishing with small random perturbations
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s kill
        rng = np.random.default_rng(42)

        best_x = None
        best_fx = np.inf
        calls_used = 0

        def check_time():
            return time.time() < deadline

        def check_budget():
            return calls_used < self.budget

        # Phase 1: Latin Hypercube Sampling for diverse initial points
        n_lhs = min(30, self.budget // 10)
        if n_lhs > 0 and check_time() and check_budget():
            # Generate LHS points
            indices = np.arange(n_lhs)
            lhs_points = np.zeros((n_lhs, self.dim))
            for d in range(self.dim):
                shuffled = rng.permutation(n_lhs)
                lhs_points[:, d] = (shuffled + 0.5) / n_lhs * 10 - 5  # Map to [-5, 5]

            # Evaluate all LHS points
            for i in range(n_lhs):
                if not check_time() or not check_budget():
                    break
                x = lhs_points[i]
                fx = func(x)
                calls_used += 1
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

        # Phase 2: Nelder-Mead refinement from top starting points
        if best_x is not None and check_time() and check_budget():
            # Collect candidate starting points (top from LHS + center)
            candidates = [best_x.copy()]
            # Add a few more diverse points
            for _ in range(min(5, self.budget // 30)):
                if not check_time() or not check_budget():
                    break
                x_try = rng.uniform(-4, 4, self.dim)
                fx_try = func(x_try)
                calls_used += 1
                if fx_try < best_fx:
                    best_fx = fx_try
                    best_x = x_try.copy()
                    candidates = [best_x.copy()]
                else:
                    candidates.append(x_try)

            # Run Nelder-Mead from the best candidate
            x0 = best_x.copy()
            # Set initial simplex
            xk = 0.5  # Initial step size
            simplex = np.vstack([x0] + [x0 + xk * rng.standard_normal(self.dim) for _ in range(self.dim)])
            simplex = np.clip(simplex, -5, 5)

            # Evaluate initial simplex
            simplex_fvals = np.empty(self.dim + 1)
            for i in range(self.dim + 1):
                if not check_time() or not check_budget():
                    break
                simplex_fvals[i] = func(simplex[i])
                calls_used += 1
                if simplex_fvals[i] < best_fx:
                    best_fx = simplex_fvals[i]
                    best_x = simplex[i].copy()

            # Nelder-Mead iterations
            nm_iters = 0
            max_nm_iters = 50
            while nm_iters < max_nm_iters and check_time() and check_budget():
                nm_iters += 1
                # Sort simplex by function value
                order = np.argsort(simplex_fvals)
                simplex = simplex[order]
                simplex_fvals = simplex_fvals[order]

                # Compute centroid (excluding worst point)
                centroid = simplex[:-1].mean(axis=0)

                # Reflection
                xr = centroid + (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fxr = func(xr)
                calls_used += 1

                if fxr < best_fx:
                    best_fx = fxr
                    best_x = xr.copy()

                if simplex_fvals[0] <= fxr < simplex_fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    simplex_fvals[-1] = fxr
                elif fxr < simplex_fvals[0]:
                    # Expansion
                    xe = centroid + 2 * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fxe = func(xe)
                    calls_used += 1
                    if fxe < best_fx:
                        best_fx = fxe
                        best_x = xe.copy()
                    if fxe < fxr:
                        simplex[-1] = xe
                        simplex_fvals[-1] = fxe
                    else:
                        simplex[-1] = xr
                        simplex_fvals[-1] = fxr
                else:
                    # Contraction
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fxc = func(xc)
                    calls_used += 1
                    if fxc < best_fx:
                        best_fx = fxc
                        best_x = xc.copy()
                    if fxc < simplex_fvals[-1]:
                        simplex[-1] = xc
                        simplex_fvals[-1] = fxc
                    else:
                        # Shrink
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if not check_time() or not check_budget():
                                break
                            simplex_fvals[i] = func(simplex[i])
                            calls_used += 1
                            if simplex_fvals[i] < best_fx:
                                best_fx = simplex_fvals[i]
                                best_x = simplex[i].copy()

                # Check for convergence
                if np.max(simplex_fvals) - np.min(simplex_fvals) < 1e-10:
                    break

        # Phase 3: CMA-ES for global optimization
        if best_x is not None and check_time() and check_budget():
            # Implement a simple CMA-ES
            dim = self.dim
            lam = max(4, int(4 + 3 * dim))  # Population size
            mu = lam // 2  # Number of parents

            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / (weights ** 2).sum()

            # Parameters
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c1 = (2 - (mu_eff + 2) / (dim ** 2 + mu_eff) + mu_eff / (dim + 1.3)) / ((dim + 1.3) ** 2 + mu_eff)
            cmu = min(1 - c1, (mu_eff - 2 + 2 / (mu_eff + 1)) / ((dim + 2) ** 2 + mu_eff))
            cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            dc = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1) - 1)) + cc

            # State
            x_mean = best_x.copy()
            sigma = 1.0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chisq = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            cma_iters = 0
            max_cma_iters = 30
            while cma_iters < max_cma_iters and check_time() and check_budget():
                cma_iters += 1

                # Generate population
                C_sqrt = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                samples = rng.standard_normal((lam, dim))
                pop = x_mean + sigma * (samples @ C_sqrt.T)
                pop = np.clip(pop, -5, 5)

                # Evaluate population
                fvals = np.empty(lam)
                for i in range(lam):
                    if not check_time() or not check_budget():
                        break
                    fvals[i] = func(pop[i])
                    calls_used += 1
                    if fvals[i] < best_fx:
                        best_fx = fvals[i]
                        best_x = pop[i].copy()

                # Sort
                order = np.argsort(fvals)
                pop_sorted = pop[order]
                fvals_sorted = fvals[order]

                # Update mean
                w = weights[:min(mu, lam)]
                w = w / w.sum()
                x_old = x_mean.copy()
                x_mean = (w[:, None] * pop_sorted[:len(w)]).sum(axis=0)

                # Update step size
                y = (x_mean - x_old) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt.T @ (y / C_sqrt.diagonal()))
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chisq - 1))
                sigma = np.clip(sigma, 1e-8, 1e6)

                # Update covariance
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu_eff) * y
                delta_sig = (np.linalg.norm(ps) / chisq - 1) * (1 + d_sigma)
                C = (1 + c1 * delta_sig + cmu - c1) * C + c1 * (np.outer(pc, pc) - C)
                C += cmu * sum(w[i] * np.outer(pop_sorted[i] - x_old, pop_sorted[i] - x_old) for i in range(len(w))) / sigma ** 2

                # Ensure positive definiteness
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Phase 4: Final polishing with small perturbations
        if best_x is not None and check_time() and check_budget():
            small_sigma = 0.1
            for _ in range(min(50, self.budget - calls_used)):
                if not check_time() or not check_budget():
                    break
                x_try = np.clip(best_x + small_sigma * rng.standard_normal(self.dim), -5, 5)
                fx_try = func(x_try)
                calls_used += 1
                if fx_try < best_fx:
                    best_fx = fx_try
                    best_x = x_try.copy()
                    small_sigma *= 0.8
                else:
                    small_sigma *= 0.7

        if best_x is None:
            best_x = np.zeros(self.dim)
            if check_time() and check_budget():
                best_fx = func(best_x)
                calls_used += 1

        return best_x, best_fx
# EVOLVE-BLOCK-END
