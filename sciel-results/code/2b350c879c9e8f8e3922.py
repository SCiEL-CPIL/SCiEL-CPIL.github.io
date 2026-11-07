"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-restart CMA-ES with Powell polish and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s

        # Track best solution
        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = x.reshape(dim)
            x = np.clip(x, -5.0, 5.0)
            call_count += 1
            return float(func(x))

        # CMA-ES implementation
        def cma_es(func, x0, sigma0, max_iters, rng):
            nonlocal call_count, best_x, best_f, deadline
            
            n = dim
            # Population size
            lambda_ = 4 + int(3 * np.log(n))
            # Step size adaptation
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)
            # Covariance matrix adaptation
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            chi_n = n * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # State
            x_mean = np.array(x0, dtype=float)
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            sqrtC_inv = np.eye(n)
            sqrtC = np.eye(n)

            # Precompute
            inv_chi_n = 1.0 / chi_n

            for it in range(max_iters):
                if time.time() > deadline - 5.0:
                    break
                if call_count >= budget:
                    break

                # Generate population
                # Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reset
                    C = np.eye(n)
                    L = np.eye(n)
                
                B = np.linalg.inv(L)  # This is actually L^{-1}
                D = np.diag(1.0 / np.sqrt(np.diag(C)))
                B = L @ D  # Actually we need B = L^{-1} D^{-1}... let me fix this

                # Standard CMA-ES: B = L^{-1}, D = diag(1/sqrt(diag(C)))
                # Then y = B @ (D @ z) where z is standard normal
                # Actually the standard is: x = x_mean + sigma * B @ D @ z
                # where B D = C^{1/2}

                # Let's use a simpler approach: use eigendecomposition
                try:
                    evals, evecs = np.linalg.eigh(C)
                    evals = np.maximum(evals, 1e-12)
                    C_sqrt = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
                except:
                    C_sqrt = np.eye(n)

                # Generate lambda samples
                zs = rng.standard_normal((lambda_, n))
                # Transform to C^{1/2} * z
                ys = (zs @ C_sqrt.T).T  # shape (lambda_, n)
                ps_pop = x_mean[None, :] + sigma * ys

                # Evaluate
                fs = []
                for i in range(lambda_):
                    if call_count >= budget or time.time() > deadline - 5.0:
                        break
                    xi = np.clip(ps_pop[i], -5.0, 5.0)
                    fi = safe_func(xi)
                    fs.append(fi)
                    if fi < best_f:
                        best_f = fi
                        best_x = xi.copy()

                if len(fs) < lambda_:
                    break

                fs = np.array(fs)
                # Sort
                order = np.argsort(fs)
                ps_sorted = ps_pop[order]
                fs_sorted = fs[order]

                # Update mean
                x_new_mean = sum(weights[i] * ps_sorted[i] for i in range(mu))
                
                # Update step size
                d = x_new_mean - x_mean
                if sigma > 1e-12:
                    ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * d
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                    sigma = np.clip(sigma, 1e-12, 100.0)

                # Update covariance
                if sigma > 1e-12:
                    pc = (1 - c_c) * pc + c_c * d / sigma
                    delta_h = 1 if np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c) ** (it + 1)) < (1.4 + 1 / c_c) * chi_n else 0
                    C = (1 + c_1 * delta_h - c_1 - c_mu) * C \
                        + c_1 * (pc @ pc.T - (1 - delta_h) * C) \
                        + c_mu * sum(weights[i] * np.outer(ps_sorted[i] - x_mean, ps_sorted[i] - x_mean) for i in range(mu))

                x_mean = x_new_mean

            return x_mean, fs_sorted[0] if len(fs_sorted) > 0 else np.inf

        # Powell polish
        def powell_polish(func, x0, maxiter=50):
            nonlocal call_count, best_x, best_f, deadline
            try:
                res = minimize(func, x0, method='Powell', options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-12})
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.clip(res.x, -5.0, 5.0).copy()
            except:
                pass

        # Multi-restart strategy
        rng = np.random.default_rng(123)

        # Phase 1: Global search with CMA-ES from multiple starting points
        n_restarts = 5
        budget_per_restart = budget // (n_restarts + 2)  # reserve some for polish
        budget_per_restart = max(budget_per_restart, 30)

        for r in range(n_restarts):
            if call_count >= budget or time.time() > deadline - 30.0:
                break

            # Starting point: mix of random and around best
            if r == 0:
                x0 = rng.uniform(-5, 5, dim)
            elif best_x is not None and r < n_restarts // 2:
                # Local restart around best
                x0 = best_x + rng.standard_normal(dim) * 1.0
                x0 = np.clip(x0, -5, 5)
            else:
                x0 = rng.uniform(-5, 5, dim)

            sigma0 = 2.0 if r == 0 else 1.0
            max_iters = budget_per_restart // (dim + 1) + 5
            max_iters = min(max_iters, 200)

            cma_es(safe_func, x0, sigma0, max_iters, rng)

        # Phase 2: Powell polish from best point
        if best_x is not None and call_count < budget - 10 and time.time() < deadline - 15.0:
            powell_polish(safe_func, best_x, maxiter=100)

        # Phase 3: Fine-tune with small steps if budget remains
        if best_x is not None and call_count < budget - 5 and time.time() < deadline - 10.0:
            sigma = 0.1
            for _ in range(min(50, budget - call_count)):
                if time.time() > deadline - 5.0:
                    break
                y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)
                if fy < best_f:
                    best_x, best_f = y.copy(), fy
                    sigma *= 1.3
                else:
                    sigma *= 0.8

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = safe_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
