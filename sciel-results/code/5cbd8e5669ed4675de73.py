"""Optimizer for BBOB f13 (Sharp Ridge) using CMA-ES with restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for BBOB f13 Sharp Ridge."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # safety margin before 300s limit

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_used = 0

        def safe_call(x):
            nonlocal calls_used
            if calls_used >= budget:
                return None
            calls_used += 1
            return func(x)

        def cma_es(x0, sigma0, max_evals, rng, seed_offset):
            """Run CMA-ES. Returns (best_x, best_f, calls_made_in_this_run)."""
            n = dim
            # Population size
            lam = max(4, int(np.ceil(4 + 3 * np.log(n))))
            mu = lam // 2

            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)

            # Step size control parameters
            cs = (n + 1.3) / (n + n + 1)  # simplified damping
            c1 = (2 + 2 / (n + 1.3)) / ((n + 2) ** 2 + 10)
            cmu = min(0.5 - c1, 2 * ((2 - c1) ** 2 - 1) / ((n + 2) ** 2 + 12))
            chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

            # State
            x = x0.copy()
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            C = np.eye(n)

            local_best_x = x0.copy()
            local_best_f = float('inf')
            calls_in_run = 0
            no_improve = 0
            stagnation_limit = 15 * lam  # restart if no improvement for this many generations

            for gen in range(max_evals // lam + 1):
                if time.time() > deadline:
                    break
                if calls_used + lam > budget:
                    # Only evaluate remaining calls
                    remaining = budget - calls_used
                    if remaining <= 0:
                        break
                    # Generate only 'remaining' offspring
                    offspring = []
                    for i in range(remaining):
                        z = rng.standard_normal(n)
                        y = x + sigma * (B @ (D * z))
                        y = np.clip(y, -5, 5)
                        offspring.append(y)
                    off_f = []
                    for y in offspring:
                        fy = safe_call(y)
                        if fy is None:
                            break
                        off_f.append(fy)
                        calls_in_run += 1
                        if fy < local_best_f:
                            local_best_f, local_best_x = fy, y.copy()
                        if fy < best_f:
                            best_f, best_x = fy, y.copy()
                    break
                else:
                    # Generate full population
                    offspring = []
                    for i in range(lam):
                        z = rng.standard_normal(n)
                        y = x + sigma * (B @ (D * z))
                        y = np.clip(y, -5, 5)
                        offspring.append(y)

                    off_f = []
                    for y in offspring:
                        fy = safe_call(y)
                        if fy is None:
                            off_f = None
                            break
                        off_f.append(fy)
                        calls_in_run += 1
                        if fy < local_best_f:
                            local_best_f, local_best_x = fy, y.copy()
                        if fy < best_f:
                            best_f, best_x = fy, y.copy()

                    if off_f is None:
                        break

                    # Sort by fitness
                    indices = np.argsort(off_f)
                    sorted_offspring = [offspring[i] for i in indices]
                    sorted_off_f = [off_f[i] for i in indices]

                    # Check stagnation
                    if sorted_off_f[0] >= local_best_f + 1e-10:
                        no_improve += lam
                    else:
                        no_improve = 0

                    if no_improve >= stagnation_limit:
                        break

                    # Update mean
                    y_w = np.zeros(n)
                    for i in range(mu):
                        y_w += w[i] * sorted_offspring[i]
                    x_new = y_w

                    # Update sigma (CSA)
                    z_sigma = (B.T @ (D * (B.T @ (x_new - x)))) / sigma if sigma > 1e-15 else np.zeros(n)
                    p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * z_sigma
                    hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (2 * (lam + 1)))) < (1.4 + 2 / (n + 1)) * chi_n
                    if hsig:
                        sigma *= np.exp(cs / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))

                    # Update C (covariance matrix)
                    y_w_diff = x_new - x
                    x = x_new
                    p_c = (1 - c1) * p_c + hsig * np.sqrt(c1 * (2 - c1) * mu_eff) * y_w_diff

                    # C update
                    C = (1 - c1 * (1 + 2 * cmu / mu_eff) - c1) * C
                    C += c1 * (np.outer(p_c, p_c) - (1 - c1) * C)
                    for i in range(mu):
                        y_i = (B.T @ (D * (B.T @ (sorted_offspring[i] - x)))) / sigma
                        C += w[i] * np.outer(y_i, y_i)

                    # Ensure C is symmetric and positive semi-definite
                    C = (C + C.T) / 2

                    # Eigen-decompose C = B * D^2 * B^T
                    try:
                        eigvals, eigvecs = np.linalg.eigh(C)
                        # Clip small negative eigenvalues
                        eigvals = np.maximum(eigvals, 1e-20)
                        B = eigvecs
                        D = np.sqrt(eigvals)
                    except np.linalg.LinAlgError:
                        break

                    # Safety: if sigma becomes too small or too large, stop
                    if sigma < 1e-10 or sigma > 1e10:
                        break

            return local_best_x, local_best_f, calls_in_run

        # Phase 1: Initial exploration with a few random points
        rng_main = np.random.default_rng(42)
        n_init = min(10, budget)
        for i in range(n_init):
            if calls_used >= budget or time.time() > deadline:
                break
            x = rng_main.uniform(-5, 5, dim)
            fx = safe_call(x)
            if fx is None:
                break
            if fx < best_f:
                best_f, best_x = fx, x.copy()

        # Phase 2: CMA-ES from the best point found
        if calls_used < budget and time.time() < deadline:
            remaining = budget - calls_used
            # Use a reasonable initial sigma
            sigma0 = 1.0
            rng_cma = np.random.default_rng(123)
            x_best, f_best, _ = cma_es(best_x, sigma0, remaining, rng_cma, 0)
            if f_best < best_f:
                best_f, best_x = f_best, x_best

        # Phase 3: If budget remains, try a second CMA-ES restart from a different point
        if calls_used < budget and time.time() < deadline:
            remaining = budget - calls_used
            if remaining > 20:
                # Start from a random point near the best
                x_start = best_x + 0.5 * rng_main.standard_normal(dim)
                x_start = np.clip(x_start, -5, 5)
                sigma0 = 0.5
                rng_cma2 = np.random.default_rng(456)
                x_best2, f_best2, _ = cma_es(x_start, sigma0, remaining, rng_cma2, 1)
                if f_best2 < best_f:
                    best_f, best_x = f_best2, x_best2

        return best_x, best_f
# EVOLVE-BLOCK-END
