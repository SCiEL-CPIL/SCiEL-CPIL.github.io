"""Multi-start CMA-ES with adaptive restarts for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive restarts for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 275.0
        calls_made = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x = np.clip(x, -5, 5)
            f = float(func(x))
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Diverse initialization with Sobol-like sequence
        n_init = 20
        init_results = []
        
        # Use a quasi-random sequence for good coverage
        def sobol_like(i, n_dims):
            """Simple quasi-random sequence generator."""
            point = np.zeros(n_dims)
            for d in range(n_dims):
                # Use different bases for each dimension
                base = d + 2
                val = 0.0
                frac = 1.0
                idx = i
                while idx > 0:
                    val += frac * (idx % base)
                    idx //= base
                    frac /= base
                point[d] = val
            return point * 10.0 - 5.0

        for i in range(1, n_init + 1):
            if time.time() > deadline or calls_made >= budget:
                break
            x = sobol_like(i, dim)
            x = np.clip(x, -5, 5)
            f = evaluate(x)
            init_results.append((x, f))

        if not init_results:
            return best_x, best_f

        # Sort by fitness
        init_results.sort(key=lambda t: t[1])

        # Phase 2: CMA-ES from best start
        x0 = init_results[0][0].copy()
        f0 = init_results[0][1]

        # CMA-ES parameters
        pop_size = dim + 3
        sigma = 1.0
        cc = (4 + dim / (dim + 4)) / ((dim + 4) + 4 + dim / (dim + 4))
        cmu = min(1, (2 + 2.0 / (dim + 1.3)) / (dim + 4 + 2 + 2.0 / (dim + 1.3)))
        cs = (dim + 1.7) / (dim + 9 + cmu)
        ds = 1 + 2 * max(0, np.sqrt((cs - 1) / (cs + 1)) - 1)
        cc = (4 + dim / (dim + 4)) / ((dim + 4) + 4 + dim / (dim + 4))

        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        mean = x0.copy()
        sigma = 1.0
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        es_calls = 0
        max_es_calls = budget - calls_made - 20  # Reserve 20 for final polish

        while es_calls < max_es_calls:
            if time.time() > deadline or calls_made >= budget:
                break

            # Generate population
            offspring = []
            offspring_f = []
            for i in range(pop_size):
                if time.time() > deadline or calls_made >= budget:
                    break
                z = rng.standard_normal(dim)
                # Approximate C^{1/2} z using current C
                y = mean + sigma * (C_sqrt_inv @ z)
                y = np.clip(y, -5, 5)
                f = evaluate(y)
                offspring.append(y)
                offspring_f.append(f)
                es_calls += 1

            if len(offspring) < pop_size:
                break

            # Sort by fitness
            indices = np.argsort(offspring_f)
            offspring = [offspring[i] for i in indices]
            offspring_f = [offspring_f[i] for i in indices]

            # Update mean
            lambda_mu = max(2, pop_size // 2)
            w = np.zeros(lambda_mu)
            for i in range(lambda_mu):
                w[i] = np.log(i + 1) - np.mean([np.log(j + 1) for j in range(lambda_mu)])
            w /= np.sum(w)
            mu_eff = 1 / np.sum(w ** 2)

            mean_old = mean.copy()
            for i in range(lambda_mu):
                mean += w[i] * (offspring[i] - mean)

            # Update sigma path
            C_sqrt = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            C_sqrt_inv = np.linalg.inv(C_sqrt)

            z_bar = np.zeros(dim)
            for i in range(lambda_mu):
                z_bar += w[i] * (C_sqrt_inv @ (offspring[i] - mean_old)) / sigma

            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * z_bar
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs) ** (es_calls + 1))) / chi_n < 1.4 + 2 / (dim + 1)

            # Update C path
            p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mu_eff) * (C_sqrt_inv @ (mean - mean_old)) / sigma

            # Update covariance matrix
            delta_h = (2 * h_sigma - 1) * cc * (2 - cc)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(lambda_mu):
                y_i = (offspring[i] - mean_old) / sigma
                rank_mu += w[i] * np.outer(y_i, y_i)

            C = (1 + cs * delta_h - cs - cmu) * C + cs * rank_one + cmu * rank_mu

            # Ensure C is positive definite
            C = (C + C.T) / 2
            try:
                C_sqrt = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
                C_sqrt_inv = np.linalg.inv(C_sqrt)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                C_sqrt_inv = np.eye(dim)

            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Check for convergence
            if np.max(np.abs(C)) < 1e-10 or sigma < 1e-8:
                break

        # Phase 3: Final polish with local search
        if time.time() < deadline and calls_made < budget and best_x is not None:
            x_polish = best_x.copy()
            f_polish = best_f
            sigma_p = 0.05
            max_polish = min(30, budget - calls_made)

            for _ in range(max_polish):
                if time.time() > deadline or calls_made >= budget:
                    break
                y = np.clip(x_polish + sigma_p * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                if fy < f_polish:
                    x_polish = y
                    f_polish = fy
                    sigma_p *= 1.1
                else:
                    sigma_p *= 0.8
                sigma_p = np.clip(sigma_p, 1e-5, 0.5)

            if f_polish < best_f:
                best_x = x_polish.copy()
                best_f = f_polish

        return best_x, best_f
# EVOLVE-BLOCK-END
