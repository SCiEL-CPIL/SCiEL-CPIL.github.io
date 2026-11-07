"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive covariance, population, and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        # Initial exploration: sample a few points to get a rough idea
        x0 = rng.uniform(-4, 4, self.dim)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0

        # Try a few random starts to find a good initial region
        for i in range(min(10, self.budget // 50)):
            if time.time() > deadline:
                break
            xi = rng.uniform(-4, 4, self.dim)
            fi = func(xi)
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()

        # CMA-ES parameters
        dim = self.dim
        lam = max(4, int(4 + 3 * np.log(dim)))  # population size
        c1 = (2 + 2 / (dim + 1.3)) / ((dim + 2) ** 2 + 2)
        cmu = min(1 - c1, 2 * (2 - (dim + 1) / (dim + 4)) / ((dim + 2) ** 2 + 2))
        cs = (dim + 1.7) / (dim + 9 + 2 * dim / (dim + 1.7))
        ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1)
        cc = (4 + dim / dim) / (4 + dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        def cma_es(x_init, sigma_init, budget_left):
            nonlocal best_x, best_f
            x = x_init.copy()
            sigma = sigma_init
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            step_size = 1.0
            n_calls = 0

            # Precompute
            inv_sqrt_C = np.eye(dim)
            D = np.ones(dim)
            B = np.eye(dim)
            eigvals = np.ones(dim)

            for _ in range(budget_left):
                if time.time() > deadline:
                    break

                # Generate population
                ys = np.zeros((lam, dim))
                fs = np.zeros(lam)
                for i in range(lam):
                    if time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    y = B @ (D * z)
                    ys[i] = y
                    xi = np.clip(x + sigma * y, -5, 5)
                    fs[i] = func(xi)
                    n_calls += 1
                    if fs[i] < best_f:
                        best_f = fs[i]
                        best_x = xi.copy()

                # Sort by fitness
                order = np.argsort(fs)
                ys_sorted = ys[order]
                fs_sorted = fs[order]

                # Update mean
                weights = np.zeros(lam)
                weights[:min(lam, 2 * lam // 3)] = 1.0 / (2 * lam)
                # Use top mu individuals
                mu = lam // 2
                w_sum = np.sum(weights[:mu])
                weights[:mu] /= w_sum
                x = x + sigma * np.sum(weights[:mu, None] * ys_sorted[:mu], axis=0)

                # Update path evolution strategies
                # Compute eigen decomposition of C periodically
                if n_calls % (lam * 5) == 0 or n_calls < lam:
                    eigvals, B = np.linalg.eigh(C)
                    D = 1.0 / np.sqrt(np.maximum(eigvals, 1e-12))
                    inv_sqrt_C = (B @ np.diag(D)) @ B.T

                # ps update
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (
                    inv_sqrt_C @ (np.sum(weights[:mu, None] * ys_sorted[:mu], axis=0))
                )
                mu_eff = 1 / np.sum(weights[:mu] ** 2)

                # pc update
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (n_calls + 1) / lam)) <
                        (1.4 + 2 / (dim + 1)) * chi_n)
                pc = (1 - cc) * pc + hsig * cc * (np.sum(weights[:mu, None] * ys_sorted[:mu], axis=0))

                # Update covariance
                delta_sig = (np.linalg.norm(ps) / chi_n - 1) * ds
                sigma *= np.exp(cs / (dim + 2) * (np.linalg.norm(ps) ** 2 / dim - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Rank-mu update
                delta = (1 + 2 * max(0, np.sqrt(hsig - 1) - 1)) * cmu
                C = ((1 + c1 + delta - c1 - cmu) * C +
                     c1 * np.outer(pc, pc) +
                     delta * np.sum(weights[:mu] ** 2 * np.outer(ys_sorted[:mu, :][:, None], ys_sorted[:mu, :][:, None]), axis=0))

                # Ensure positive semi-definite
                eigvals_new, B_new = np.linalg.eigh(C)
                C = B_new @ np.diag(np.maximum(eigvals_new, 1e-12)) @ B_new.T

            return n_calls

        # Run CMA-ES with restarts
        remaining = self.budget - n_calls
        if remaining > lam * 10:
            calls_used = cma_es(best_x, 1.5, remaining)
        else:
            calls_used = 0

        # If we have budget left, do local refinement
        remaining2 = self.budget - (n_calls + calls_used)
        if remaining2 > 0 and time.time() < deadline:
            sigma_local = 0.5
            for _ in range(remaining2):
                if time.time() > deadline:
                    break
                y = np.clip(best_x + sigma_local * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma_local *= 1.2
                else:
                    sigma_local *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
