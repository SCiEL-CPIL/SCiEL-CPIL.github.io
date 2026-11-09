"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for BBOB-style continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(12345)

        dim = self.dim
        budget = self.budget
        calls = 0

        def time_ok():
            return (time.time() - start_time) < deadline

        # CMA-ES parameters
        def run_cmaes(x0, sigma0, remaining_budget):
            nonlocal calls
            d = dim
            # Population size
            lam = max(4, int(4 + 3 * np.log(d)))
            # Weights
            mu = lam // 2
            w = np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mueff = int(1 / (w.sum() / (w ** 2).sum()))
            mu_w = w[:mueff] / w[:mueff].sum()

            # CMA-ES parameters
            cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
            cs = (mueff + 2) / (d + mueff + 5)
            c1 = (2 - cc - cs) / ((d + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))

            # State
            x = x0.copy()
            sigma = sigma0
            C = np.eye(d)
            pc = np.zeros(d)
            ps = np.zeros(d)
            chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

            best_x = x.copy()
            best_f = func(x)
            calls += 1

            gen = 0
            while calls < budget and time_ok():
                # Generate population
                B = np.linalg.cholesky(C).T
                pops = []
                for i in range(lam):
                    z = rng.standard_normal(d)
                    xi = x + sigma * (B @ z)
                    xi = np.clip(xi, -5, 5)
                    pops.append((xi, None))

                # Evaluate
                for i in range(lam):
                    fi = func(pops[i][0])
                    pops[i] = (pops[i][0], fi)
                    calls += 1
                    if fi < best_f:
                        best_f = fi
                        best_x = pops[i][0].copy()

                # Sort by fitness
                pops.sort(key=lambda t: t[1])
                ys = [p[0] for p in pops[:mu]]
                ys = np.array(ys)

                # Update mean
                xold = x.copy()
                x = x + sum(mu_w[i] * (ys[i] - x) for i in range(mu))
                x = np.clip(x, -5, 5)

                # Update step size (CSA)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / chi_n * (B.T @ (x - xold) / sigma)
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))) < (1.4 + 2 / (d + 1)) * chi_n
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 20.0)

                # Update covariance
                delta = hsig * cc * (x - xold)
                pc = (1 - cc) * pc + cc * delta
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + cc * (1 - cc) * np.eye(d))
                rank_mu = np.zeros((d, d))
                for i in range(mu):
                    vi = (ys[i] - xold) / sigma
                    rank_mu += mu_w[i] * np.outer(vi, vi)
                C += cmu * rank_mu

                # Ensure C is positive definite
                eigvals = np.linalg.eigvalsh(C)
                if eigvals.min() < 1e-12:
                    C += 1e-10 * np.eye(d)

                gen += 1

                # Check for stagnation - if sigma is very small or no progress
                if gen % 50 == 0:
                    # Check if we're stuck
                    if sigma < 1e-6 or gen > 200:
                        break

            return best_x, best_f

        # Multi-restart strategy
        best_overall_x = None
        best_overall_f = np.inf

        # Initial diverse starting points
        start_points = []
        # Center
        start_points.append(np.zeros(dim))
        # Corners and midpoints
        for i in range(min(8, budget // 50)):
            sp = rng.uniform(-4, 4, dim)
            start_points.append(sp)

        # Run CMA-ES from each starting point with remaining budget
        for sp in start_points:
            if not time_ok() or calls >= budget:
                break
            remaining = budget - calls
            if remaining < 30:
                break

            # Use a portion of budget for each restart
            alloc = min(remaining, max(50, budget // len(start_points)))
            if alloc < 20:
                alloc = remaining

            try:
                bx, bf = run_cmaes(sp, 2.0, budget)
                if bf < best_overall_f:
                    best_overall_f = bf
                    best_overall_x = bx
            except Exception:
                break

            if not time_ok():
                break

        if best_overall_x is None:
            # Fallback: single point
            x = np.zeros(dim)
            best_overall_f = func(x)
            calls += 1
            best_overall_x = x

        return best_overall_x, best_overall_f
# EVOLVE-BLOCK-END
