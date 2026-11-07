"""Rewritten optimizer: multi-start CMA-ES with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES for black-box continuous optimization on [-5, 5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        deadline = time.time() + 280  # 300s limit, 20s safety margin

        def safe_call(x):
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            return float(func(x))

        best_x = None
        best_f = np.inf
        calls_used = 0

        def run_cmaes(x0, sigma0, max_calls):
            nonlocal best_x, best_f, calls_used

            if max_calls < 10:
                return

            # CMA-ES parameters
            n = d
            lam = max(4, int(4 + 3 * np.log(n)))  # population size
            w = np.log(lam / 2 + 0.5) - np.log(np.arange(1, lam + 1))
            w = w / w.sum()
            mu_eff = 1.0 / (w * w).sum()
            cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            cs = (mu_eff + 2) / (n + mu_eff + 5)
            c1 = (2 / ((n + 1.3) ** 2 + mu_eff)) * (1 - cc + 2 * min(1, (1 - mu_eff) / (n + 1)) / max(1, mu_eff))
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

            # Initialize
            m = np.array(x0, dtype=np.float64)
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Initial population
            pop = []
            fvals = []
            for i in range(lam):
                if calls_used >= max_calls or time.time() >= deadline:
                    break
                z = np.random.default_rng().standard_normal(n)
                xi = m + sigma * z
                xi = np.clip(xi, -5.0, 5.0)
                fi = safe_call(xi)
                calls_used += 1
                pop.append(xi)
                fvals.append(fi)

            if not pop:
                return

            # Sort by fitness
            order = np.argsort(fvals)
            pop = [pop[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = pop[0].copy()

            # Main loop
            while calls_used < max_calls and time.time() < deadline:
                # Update mean
                m_new = sum(w[i] * pop[i] for i in range(lam))

                # Update covariance
                diff = [pop[i] - m for i in range(lam)]
                C_new = C.copy()
                for i in range(lam):
                    C_new += w[i] * np.outer(diff[i], diff[i])
                # Rank-one and rank-mu updates
                pc = (1 - cc) * pc + cc * (m_new - m) / sigma
                hsig = (1 - (1 - cc) ** (1 + 0))  # simplified
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (m_new - m) / sigma
                c_sig = 1 + 2 * max(0, np.sqrt(mu_eff / n) - 1) / (cs * n + 1)
                C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc) / cc + cmu * sum(
                    w[i] * np.outer(diff[i], diff[i]) for i in range(lam)
                )

                # Symmetrize
                C = 0.5 * (C + C.T)

                # Update sigma
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 50.0)

                m = m_new

                # Generate new population
                pop = []
                fvals = []
                for i in range(lam):
                    if calls_used >= max_calls or time.time() >= deadline:
                        break
                    z = np.random.default_rng().standard_normal(n)
                    # Cholesky decomposition for C
                    try:
                        L = np.linalg.cholesky(C)
                        xi = m + sigma * (L @ z)
                    except np.linalg.LinAlgError:
                        xi = m + sigma * z
                    xi = np.clip(xi, -5.0, 5.0)
                    fi = safe_call(xi)
                    calls_used += 1
                    pop.append(xi)
                    fvals.append(fi)

                if not pop:
                    break

                # Sort
                order = np.argsort(fvals)
                pop = [pop[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Update best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = pop[0].copy()

                # Convergence check
                if sigma < 1e-8:
                    break

        # Multi-start strategy
        # Start 1: center
        run_cmaes(np.zeros(d), 2.0, int(budget * 0.4))

        # Start 2: random point
        rng2 = np.random.default_rng(123)
        x2 = rng2.uniform(-4, 4, d)
        run_cmaes(x2, 1.5, int(budget * 0.3))

        # Start 3: another random point
        rng3 = np.random.default_rng(456)
        x3 = rng3.uniform(-3, 3, d)
        run_cmaes(x3, 1.0, int(budget * 0.2))

        # Final polish: (1+1)-ES around best point
        if best_x is not None and calls_used < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            rng_polish = np.random.default_rng(789)
            while calls_used < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng_polish.standard_normal(d), -5.0, 5.0)
                fy = safe_call(y)
                calls_used += 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.8

        return best_x, best_f
# EVOLVE-BLOCK-END
