"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        def cma_es(x0, sigma0, budget_cma, rng_cma):
            """Run CMA-ES for budget_cma evaluations."""
            n = dim
            # Population size
            lambda_ = 4 + int(3 * np.log(n))
            mu = lambda_ // 2
            # Weights
            w = np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1 / np.sum(w ** 2)
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = (2 / ((n + 1.3) ** 2 + mueff)) * (2 - cc)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
            # Initialization
            mean = np.array(x0, dtype=float)
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            # Precompute
            B, D = np.linalg.eigh(C)
            invsqrtC = B @ np.diag(1.0 / np.sqrt(D)) @ B.T
            # Evaluate initial point
            x_eval = np.clip(mean, -5, 5)
            fvals = [func(x_eval)]
            count = 1
            # History of best
            best_x = x_eval.copy()
            best_f = fvals[0]

            while count < budget_cma:
                if time.time() > deadline - 5:
                    break
                # Generate population
                xs = np.zeros((lambda_, n))
                for i in range(lambda_):
                    z = rng_cma.standard_normal(n)
                    xs[i] = mean + sigma * (invsqrtC @ z)
                    xs[i] = np.clip(xs[i], -5, 5)
                # Evaluate
                for i in range(lambda_):
                    if count >= budget_cma:
                        break
                    fvals.append(func(xs[i]))
                    count += 1
                # Sort by fitness
                order = np.argsort(fvals[-lambda_:])
                ys = xs[order]
                fs = np.array(fvals[-lambda_:])[order]
                # Update best
                if fs[0] < best_f:
                    best_f = fs[0]
                    best_x = ys[0].copy()
                # Weighted mean
                y_w = np.dot(w, ys[:mu, :])
                # Update step-size control
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ (y_w / sigma))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (count - budget_cma // lambda_ + 1)))) < (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1))
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_w / sigma
                # Update covariance
                delta_d = (2 * hsig - 1) * cc
                C = (1 + c1 * delta_d - c1 - cmu) * C
                C += c1 * np.outer(pc, pc)
                C += cmu * np.sum(w[i] * np.outer(ys[i] - mean, ys[i] - mean) for i in range(mu))
                # Ensure positive definiteness
                C = (C + C.T) / 2
                # Eigen decomposition for next iteration
                B, D = np.linalg.eigh(C)
                # Clip eigenvalues to prevent numerical issues
                D = np.clip(D, 1e-20, None)
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(n) - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)
                mean = mean + w[0] * (ys[0] - mean)  # Simplified: use top solution
                # Actually, proper weighted mean:
                mean = mean + sigma * (invsqrtC @ y_w)
                # Recompute invsqrtC
                invsqrtC = B @ np.diag(1.0 / np.sqrt(D)) @ B.T

            return best_x, best_f, count

        # Phase 1: Coarse global search with large sigma
        x0 = rng.uniform(-4, 4, dim)
        best_x, best_f, used = cma_es(x0, 3.0, min(100, budget), rng)
        budget_remaining = budget - used

        if budget_remaining <= 0 or time.time() > deadline - 10:
            return best_x, best_f

        # Phase 2: Multiple restarts around best found so far
        n_restarts = min(5, max(1, budget_remaining // 60))
        budget_per_restart = max(20, budget_remaining // n_restarts)

        for r in range(n_restarts):
            if time.time() > deadline - 10:
                break
            # Perturb around current best
            x0_r = np.clip(best_x + rng.standard_normal(dim) * 0.5, -5, 5)
            sigma_r = 0.5
            bx, bf, used_r = cma_es(x0_r, sigma_r, budget_per_restart, rng)
            used += used_r
            if bf < best_f:
                best_f = bf
                best_x = bx
            budget_remaining = budget - used
            if budget_remaining <= 0:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
