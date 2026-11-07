"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global-to-local strategy: random initialization + CMA-ES with adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Global exploration with 50 uniform random samples
        n_global = min(50, budget)
        candidates = []
        for _ in range(n_global):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            candidates.append((fx, x))

        candidates.sort(key=lambda t: t[0])
        best_x = candidates[0][1].copy()
        best_fx = candidates[0][0]
        calls_used = n_global

        # Phase 2: CMA-ES with adaptive restarts
        remaining_budget = budget - calls_used

        # Initialize CMA-ES around the best point found
        x = best_x.copy()
        sigma = 1.5  # initial step size

        # CMA-ES parameters
        lambda_pop = 10
        mu = lambda_pop // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        cs = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        ds = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + cs

        c1 = (2 / ((dim + 1.3) ** 2 + mu_eff)) * (1 - c_sigma)
        cmu = min(1 - c1, 2 * (weights[0] - 0.5)) * c_sigma * c_sigma

        # State variables
        ps = np.zeros(dim)
        pc = np.zeros((dim, dim))
        C = np.eye(dim)
        csqrt = np.eye(dim)

        def update_sqrt(C_new):
            try:
                w, V = np.linalg.eigh(C_new)
                w = np.maximum(w, 1e-14)
                return V @ np.diag(np.sqrt(w)) @ V.T
            except:
                return np.eye(dim)

        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        stall_count = 0
        best_fx_cma = best_fx
        best_x_cma = best_x.copy()
        no_improve_generations = 0

        while calls_used < budget:
            # Check deadline
            if time.time() > deadline - 3.0:
                break

            # Generate population
            pop = []
            for i in range(lambda_pop):
                if calls_used >= budget:
                    break
                if time.time() > deadline - 2.0:
                    break
                try:
                    zi = rng.standard_normal(dim)
                    xi = x + sigma * (csqrt @ zi)
                    xi = np.clip(xi, -5, 5)
                    fi = func(xi)
                    calls_used += 1
                    pop.append((fi, xi))
                except Exception:
                    break

            if len(pop) == 0:
                break

            # Sort by fitness
            pop.sort(key=lambda t: t[0])

            # Update best
            if pop[0][0] < best_fx_cma:
                best_fx_cma = pop[0][0]
                best_x_cma = pop[0][1].copy()
                no_improve_generations = 0
            else:
                no_improve_generations += 1

            # Check for stagnation: restart if no improvement for many generations
            if no_improve_generations > 30 and calls_used < budget - 50:
                # Try a local restart around the best point with smaller sigma
                x = best_x_cma + rng.normal(0, 0.3, dim)
                x = np.clip(x, -5, 5)
                sigma = 0.5
                ps = np.zeros(dim)
                pc = np.zeros((dim, dim))
                C = np.eye(dim)
                csqrt = np.eye(dim)
                no_improve_generations = 0
                stall_count = 0
                continue

            # Update mean
            x_new = np.sum([weights[i] * pop[i][1] for i in range(min(mu, len(pop)))], axis=0)

            # Update evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (csqrt.T @ (x_new - x))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_used - n_global + 1)))) < \
                   (1 + 2 / (dim + 1)) * np.sqrt(mu_eff)
            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (x_new - x)

            # Update covariance matrix
            rank1 = np.outer(pc, pc)
            diff = [pop[i][1] - x for i in range(min(mu, len(pop)))]
            rankmu = np.zeros((dim, dim))
            for i in range(min(mu, len(pop))):
                y = diff[i] / sigma
                rankmu += weights[i] * np.outer(y, y)

            C = (1 + c1 + cmu - c1 - c_sigma) * C + c1 * rank1 + cmu * rankmu

            # Ensure C is symmetric and PSD
            C = (C + C.T) / 2
            try:
                w, V = np.linalg.eigh(C)
                w = np.maximum(w, 1e-14)
                C = V @ np.diag(w) @ V.T
                csqrt = update_sqrt(C)
            except:
                csqrt = np.eye(dim)
                C = np.eye(dim)

            # Update step size
            norm_ps = np.linalg.norm(ps)
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 0.001, 20.0)

            x = x_new

        # Phase 3: Final local refinement
        remaining = budget - calls_used
        if remaining > 5 and time.time() < deadline - 1.0:
            local_sigma = max(0.05, sigma * 0.3)
            for _ in range(min(remaining, 30)):
                if time.time() > deadline - 0.5:
                    break
                try:
                    y = best_x_cma + local_sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls_used += 1
                    if fy < best_fx_cma:
                        best_fx_cma = fy
                        best_x_cma = y.copy()
                        local_sigma *= 1.1
                    else:
                        local_sigma *= 0.85
                except Exception:
                    break

        return best_x_cma, best_fx_cma
# EVOLVE-BLOCK-END
