"""CMA-ES with adaptive restarts and coordinate descent for Bent Cigar."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with adaptive population, restarts, and coordinate descent for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls = 1

        def compute_weights(mu):
            w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            return w

        def cma_es(x_start, sigma_start, lam, budget_remaining, rng, func, best_x, best_f, calls):
            dim = self.dim
            x = x_start.copy()
            sigma = sigma_start
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)

            mu = max(2, lam // 2)
            weights = compute_weights(mu)
            mu_eff = 1 / np.sum(weights ** 2)

            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            chi_norm = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (16 * dim ** 2))

            stagnation = 0
            last_best = best_f

            while calls < budget:
                if budget - calls < lam:
                    remaining = budget - calls
                    for _ in range(remaining):
                        y = np.clip(best_x + sigma * 0.05 * rng.standard_normal(dim), -5, 5)
                        fy = func(y)
                        calls += 1
                        if fy < best_f:
                            best_f = fy
                            best_x = y.copy()
                    break

                pop = np.zeros((lam, dim))
                L = np.linalg.cholesky(C)
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    pop[i] = np.clip(x + sigma * L.T @ z, -5, 5)

                fpop = np.array([func(p) for p in pop])
                calls += lam

                min_idx = np.argmin(fpop)
                if fpop[min_idx] < best_f:
                    best_f = fpop[min_idx]
                    best_x = pop[min_idx].copy()
                    stagnation = 0
                else:
                    stagnation += 1

                if stagnation > 4:
                    break

                order = np.argsort(fpop)
                pop = pop[order]
                fpop = fpop[order]

                x_w = np.sum(weights[:, None] * pop[:mu], axis=0)

                L_inv = np.linalg.inv(L)
                C_inv_sqrt = L_inv.T @ L_inv

                ps = (1 - c_sigma) * ps + c_sigma * np.sqrt(mu_eff) * C_inv_sqrt @ (x_w - x) / sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_norm - 1))
                sigma = min(sigma, 10.0)

                pc = (1 - c_c) * pc + c_c * np.sqrt(mu_eff) * C_inv_sqrt @ (x_w - x) / sigma

                delta_sigma = (c_c - 2 + c_c ** 2) * sigma ** 2 * (np.linalg.norm(pc) ** 2 / dim - 1)
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + delta_sigma * np.eye(dim))
                for i in range(mu):
                    diff = pop[i] - x
                    C += c_mu * weights[i] * np.outer(diff, diff)

                x = x_w
                C = (C + C.T) / 2
                try:
                    np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 1e-8 * np.eye(dim)

            return best_x, best_f, calls, x, sigma, C

        # Phase 1: Initial CMA-ES with moderate population
        lam1 = 10
        x1, best_x, best_f, calls, x_mean, sigma, C = cma_es(
            np.zeros(dim), 3.0, lam1, budget, rng, func, best_x, best_f, calls
        )

        # Phase 2: Coordinate descent refinement
        if calls < budget:
            coord_step = max(sigma * 0.1, 0.1)
            for _ in range(min(100, budget - calls)):
                improved = False
                for d in range(dim):
                    for sign in [1, -1]:
                        y = best_x.copy()
                        y[d] = np.clip(y[d] + sign * coord_step, -5, 5)
                        fy = func(y)
                        calls += 1
                        if fy < best_f:
                            best_f = fy
                            best_x = y.copy()
                            improved = True
                    if calls >= budget:
                        break
                if not improved:
                    coord_step *= 0.5
                    if coord_step < 1e-4:
                        break
                if calls >= budget:
                    break

        # Phase 3: Final CMA-ES with smaller population for fine-tuning
        if calls < budget and budget - calls > 8:
            lam3 = min(8, budget - calls)
            x3, best_x, best_f, calls, _, _, _ = cma_es(
                best_x.copy(), max(sigma * 0.1, 0.01), lam3, budget, rng, func, best_x, best_f, calls
            )

        return best_x, best_f
# EVOLVE-BLOCK-END
