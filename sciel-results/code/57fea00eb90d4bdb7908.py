"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead optimizer with CMA-ES polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 330s limit

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        # Helper to count calls and check deadline
        def counted_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= self.budget or time.time() > deadline - 1:
                raise RuntimeError("Budget or time exceeded")

            calls_made += 1
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float)
            return f

        def cma_es_polish(x0, sigma0, max_evals):
            """Run CMA-ES from x0 with given budget for final polish."""
            nonlocal calls_made, best_f, best_x
            if calls_made >= self.budget or time.time() > deadline - 10:
                return

            dim = self.dim
            # CMA-ES parameters
            lam = 4 + int(3 * np.log(dim + 1))
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1.0 / np.sum(w ** 2)
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 1) * (dim + 1) + 2)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) * (dim + mueff)))

            sigma = sigma0
            xmean = np.array(x0, dtype=float)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            C = np.eye(dim)
            B = np.eye(dim)
            D = np.ones(dim)

            evals_remaining = max_evals

            # Evaluate initial point
            if calls_made < self.budget and time.time() < deadline - 5:
                try:
                    fx = counted_func(xmean)
                    evals_remaining -= 1
                except RuntimeError:
                    return

            while evals_remaining > 0:
                if time.time() > deadline - 5 or calls_made >= self.budget:
                    break

                # Generate samples
                samples = []
                for i in range(lam):
                    if calls_made + len(samples) >= self.budget or time.time() > deadline - 3:
                        break
                    z = rng.standard_normal(dim)
                    xi = xmean + sigma * (B @ (D * z))
                    xi = np.clip(xi, -5, 5)
                    samples.append(xi)

                if len(samples) < 2:
                    break

                # Evaluate samples
                fvals = []
                for xi in samples:
                    if calls_made >= self.budget or time.time() > deadline - 3:
                        break
                    try:
                        fi = counted_func(xi)
                        fvals.append(fi)
                    except RuntimeError:
                        break

                if len(fvals) < 2:
                    break

                # Sort by fitness
                idx = np.argsort(fvals)
                x_sorted = [samples[i] for i in idx[:mu]]
                f_sorted = [fvals[i] for i in idx[:mu]]

                # Update mean
                xnew = np.sum([w[i] * x_sorted[i] for i in range(mu)], axis=0)

                # Update evolution paths
                diff = (xnew - xmean) / (sigma + 1e-15)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ (D * diff))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made // lam + 1)))) < \
                       (1 + 2 / (dim + 1)) * np.sqrt(dim)
                if hsig:
                    pc = (1 - c1) * pc + c1 * diff
                else:
                    pc = (1 - c1) * pc

                # Update covariance
                delta_hsig = (1 - hsig) * c1 * (2 - c1)
                C = (1 + c1 * delta_hsig - c1 - cmu) * C + \
                    c1 * (np.outer(pc, pc) + (delta_hsig - c1 + c1 * c1) * np.eye(dim)) + \
                    cmu * np.sum([w[i] * np.outer(x_sorted[i] - xmean, x_sorted[i] - xmean)
                                  for i in range(mu)], axis=0)

                # Regularization and re-decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-20, None)
                C = (eigvecs * eigvals) @ eigvecs.T
                B = eigvecs
                D = np.sqrt(eigvals)

                xmean = xnew
                sigma *= np.exp(cs / (dim + 3) * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
                sigma = np.clip(sigma, 1e-10, 1e10)

                evals_remaining = self.budget - calls_made

        # Phase 1: Multi-start exploration
        # 5 restarts, each with maxiter=8
        # Estimated calls: 5 * (6 * (1+8)) = 270

        n_restarts = 5
        maxiter_restart = 8

        for i in range(n_restarts):
            if time.time() > deadline - 30:
                break

            x0 = rng.uniform(-5, 5, self.dim)

            try:
                res = minimize(counted_func, x0, method='Nelder-Mead',
                               options={'maxiter': maxiter_restart, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except RuntimeError:
                break

        # Phase 2: Exploit the best found with Nelder-Mead
        if best_x is not None and calls_made < self.budget and time.time() < deadline - 20:
            remaining_budget = self.budget - calls_made
            # Estimated calls per maxiter: 6
            # Total calls = 6 * (1 + maxiter)
            # So maxiter = (remaining_budget // 6) - 1
            maxiter_final = (remaining_budget // 6) - 1

            if maxiter_final > 0:
                try:
                    res = minimize(counted_func, best_x, method='Nelder-Mead',
                                   options={'maxiter': maxiter_final, 'xatol': 1e-10, 'fatol': 1e-12})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except RuntimeError:
                    pass

        # Phase 3: CMA-ES polish on the best found
        if best_x is not None and calls_made < self.budget and time.time() < deadline - 15:
            remaining_budget = self.budget - calls_made
            # Use a fraction of remaining budget for CMA-ES polish
            polish_budget = min(remaining_budget, 100)
            if polish_budget > 10:
                cma_es_polish(best_x, 0.5, polish_budget)

        # If no calls were made (should not happen), do a single random call
        if calls_made == 0:
            x0 = rng.uniform(-5, 5, self.dim)
            best_f = func(x0)
            best_x = x0
            calls_made = 1

        return best_x, best_f
# EVOLVE-BLOCK-END
