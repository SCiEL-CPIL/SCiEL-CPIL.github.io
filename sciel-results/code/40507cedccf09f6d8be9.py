"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-restart portfolio and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 300s limit, leave 15s margin

        d = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)
        best_x = None
        best_fx = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made
            fx = func(x)
            calls_made += 1
            return fx

        def cma_es(x0, sigma0, max_evals):
            nonlocal calls_made, best_x, best_fx

            # CMA-ES parameters for dim=5
            lam = 9  # population size
            mu = lam // 2  # number of selected individuals
            weights = np.log(np.arange(mu, 0, -1)) - np.log(np.arange(mu, 0, -1)).sum()
            weights /= weights.sum()
            mueff = 1 / np.sum(weights ** 2)

            c_sigma = (mueff + 2) / (d + mueff + 5)
            c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
            c_1 = 2 / ((d + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
            cs = (c_sigma + 1) * (0.5 + 0.5 * min(1, mueff / d - 1))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + c_sigma
            chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

            x = np.array(x0, dtype=float)
            sigma = sigma0
            ps = np.zeros(d)
            pc = np.zeros(d)
            C = np.eye(d)
            invsqrtC = np.eye(d)

            local_best_x = x.copy()
            local_best_f = np.inf

            # Generate initial population
            pop = np.tile(x, (lam, 1)) + sigma * rng.standard_normal((lam, d))
            pop = np.clip(pop, -5, 5)
            fvals = np.array([call_func(xi) for xi in pop])

            idx_best = np.argmin(fvals)
            local_best_f = fvals[idx_best]
            local_best_x = pop[idx_best].copy()
            if local_best_f < best_fx:
                best_fx = local_best_f
                best_x = local_best_x.copy()

            while calls_made < max_evals and time.time() < deadline:
                # Check if we have enough budget for next generation
                if calls_made + lam > max_evals:
                    break

                # Sort by fitness
                order = np.argsort(fvals)
                selected = pop[order[:mu]]

                # Update mean
                x_new = weights @ selected

                # Update sigma control vector
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (invsqrtC @ (x_new - x))

                # Check hsig
                norm_ps = np.linalg.norm(ps)
                hsig = 1 if norm_ps / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (d + 1)) * chi_n else 0

                # Update path for C
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) / sigma * (x_new - x)

                # Check hsig for C update
                norm_pc = np.linalg.norm(pc)
                hsig_c = 1 if norm_pc / np.sqrt(1 - (1 - c_c) ** 2) < (1.4 + 2 / (d + 1)) * chi_n else 0

                # Update covariance matrix
                delta = hsig_c * c_1 * (1 - c_1) * np.outer(pc, pc)
                diff = selected - x
                rank_mu_term = np.sum([weights[i] * np.outer(diff[i], diff[i]) for i in range(mu)], axis=0)
                C = (1 - c_1 - c_mu) * C + delta + c_mu * rank_mu_term

                x = x_new

                # Update sigma
                sigma *= np.exp(c_sigma / damps * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 100.0)

                # Update inverse square root of C periodically
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-20, None)
                invsqrtC = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T

                # Generate new population
                pop = np.tile(x, (lam, 1)) + sigma * rng.standard_normal((lam, d))
                pop = np.clip(pop, -5, 5)
                fvals = np.array([call_func(xi) for xi in pop])

                # Track best
                idx_best = np.argmin(fvals)
                if fvals[idx_best] < local_best_f:
                    local_best_f = fvals[idx_best]
                    local_best_x = pop[idx_best].copy()
                    if local_best_f < best_fx:
                        best_fx = local_best_f
                        best_x = local_best_x.copy()

            return local_best_x, local_best_f

        # Phase 1: Random sampling to find good starting points
        n_initial = min(15, budget)
        candidates = []
        for i in range(n_initial):
            if time.time() > deadline or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, d)
            fx = call_func(x)
            candidates.append((fx, x.copy()))
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_fx = call_func(best_x)
            candidates.append((best_fx, best_x.copy()))

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Run CMA-ES from top candidates
        remaining_budget = budget - calls_made
        n_restarts = min(3, max(1, remaining_budget // 80))
        budget_per_restart = remaining_budget // n_restarts

        for restart in range(n_restarts):
            if calls_made >= budget or time.time() > deadline:
                break

            # Pick a starting point: best candidate for first, then diverse ones
            if restart == 0:
                x0 = candidates[0][1].copy()
            elif restart < len(candidates):
                x0 = candidates[min(restart, len(candidates) - 1)][1].copy()
            else:
                x0 = best_x + 1.0 * rng.standard_normal(d)
                x0 = np.clip(x0, -5, 5)

            sigma0 = 1.0 if restart == 0 else 0.5
            cma_es(x0, sigma0, calls_made + budget_per_restart)

        return best_x, best_fx
# EVOLVE-BLOCK-END
