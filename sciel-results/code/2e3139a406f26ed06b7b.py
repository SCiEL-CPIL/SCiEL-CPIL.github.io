"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts and a wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330 s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(12345)

        best_x = np.full(dim, np.nan)
        best_f = np.inf
        calls_used = 0

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        def still_ok():
            nonlocal calls_used
            return (calls_used < budget) and (time.time() < deadline)

        def eval_point(x):
            nonlocal calls_used, best_x, best_f
            x = clip(x)
            fx = float(func(x))
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return x, fx

        # CMA-ES single run
        def cma_run(x0, sigma0, max_evals):
            nonlocal calls_used, best_x, best_f
            d = dim
            x = x0.copy()
            sigma = sigma0
            C = np.eye(d)
            Cs = np.zeros((d, d))
            ps = np.zeros(d)
            pc = np.zeros(d)

            lam = 4 * (1 + int(3 * d))
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)

            c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (d + 1)) + c_sigma
            c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
            c_1 = (2 / ((d + 1.3) ** 2 + mu_eff)) * (
                (1 - 1 / d) + 2 * max(0, (mu_eff - 2) / (d + mu_eff - 2))
            )
            c_mu = (
                2 * (mu_eff - 2 + 2 / mu_eff) / ((d + 2) ** 2 + mu_eff)
            ) * min(1, c_1 / c_mu if c_mu > 0 else 1)
            if c_mu <= 0:
                c_mu = c_1 * 0.5

            chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

            for _ in range(max_evals):
                if not still_ok():
                    break
                # sample
                B = np.linalg.cholesky(C)
                directions = []
                for _ in range(lam):
                    z = rng.standard_normal(d)
                    directions.append(z @ B)
                directions = np.array(directions)
                pop = x + sigma * directions
                pop = clip(pop)

                fvals = np.empty(lam)
                for i in range(lam):
                    if not still_ok():
                        # fill remaining with worst
                        fvals[i:] = np.inf
                        break
                    _, fvals[i] = eval_point(pop[i])

                order = np.argsort(fvals)
                y = directions[order]
                fvals_s = fvals[order]

                # update mean
                x_new = x + sigma * np.sum(w[:, None] * y[:mu], axis=0)
                x_new = clip(x_new)

                # update ps
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (
                    np.linalg.inv(B) @ (y[:mu].T @ w).T
                )

                # update pc
                hsig = (
                    np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_used)))
                    <= (1 + 2 / (d + 1)) * chi_d
                )
                pc = (1 - c_c) * pc + hsig * c_c * (x_new - x)
                norm_pc = np.linalg.norm(pc)
                hsig = (
                    np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (2 * (calls_used)))
                    <= (1 + 2 / (d + 1)) * chi_d
                )

                # update C
                rank1 = np.outer(ps, ps)
                rankmu = np.sum(w[:, None, None] * y[:mu, :, None] * y[:mu, None, :], axis=0)
                C = (
                    (1 - c_sigma * d_sigma * hsig * (1 - hsig) - c_1 - c_mu) * C
                    + c_sigma * d_sigma * hsig * (1 - hsig) * (C + np.outer(pc, pc))
                    + c_1 * rank1
                    + c_mu * rankmu
                )
                # symmetrize
                C = 0.5 * (C + C.T)

                # sigma update (1/5 success rule)
                sigma *= np.exp(
                    c_sigma / d_sigma * (np.linalg.norm(ps) / chi_d - 1)
                )
                sigma = min(max(sigma, 1e-12), 1e6)

                x = x_new

            return x

        # Restart loop: several independent CMA-ES runs with different starts
        # First: a few random probes to get a rough best, then CMA from best
        n_probes = 10
        for _ in range(n_probes):
            if not still_ok():
                break
            xp = rng.uniform(-5, 5, dim)
            eval_point(xp)

        # Now run CMA-ES from the current best
        remaining = budget - calls_used
        if remaining > 0:
            cma_run(best_x.copy(), 2.0, remaining)

        # If time/budget remains, do a second CMA-ES restart from a different region
        # to help with multimodal functions
        if still_ok() and (budget - calls_used) > 30:
            # pick a random new start, biased toward best
            x_start = best_x + rng.standard_normal(dim) * 1.5
            x_start = clip(x_start)
            remaining2 = budget - calls_used
            cma_run(x_start, 1.5, remaining2)

        # If still budget left, another restart
        if still_ok() and (budget - calls_used) > 30:
            x_start = rng.uniform(-5, 5, dim)
            remaining3 = budget - calls_used
            cma_run(x_start, 2.0, remaining3)

        if np.isnan(best_f) or np.isinf(best_f):
            # fallback: return origin
            best_x = np.zeros(dim)
            best_f = float(func(best_x))
            calls_used += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
