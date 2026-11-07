"""Portfolio CMA-ES with L-BFGS-B polish for Bent Cigar (f12)."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Portfolio CMA-ES with L-BFGS-B polish for Bent Cigar (f12)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 270.0
        n = self.dim
        budget = self.budget

        # Global best tracking
        best_x = np.zeros(n)
        best_fx = func(best_x)
        calls_used = 1

        def cma_es_instance(x0, sigma0, max_calls, deadline, rng_local):
            """Run a single CMA-ES instance from x0 with initial sigma0."""
            nonlocal best_x, best_fx, calls_used

            lam = max(4, int(4 + 3 * np.log(n)))
            mu = lam // 2
            w = np.log(np.arange(mu, 0, -1))
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_cov_pos = (2 + 2 * mu_eff / n) / (n + 3 + 2 * mu_eff / n)
            c_cov_neg = (2 + mu_eff / n) / (n + 3 + 2 * mu_eff / n)

            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            D, B = np.linalg.eigh(C)
            D = np.maximum(D, 1e-12)
            C_inv_sqrt = B @ np.diag(1.0 / np.sqrt(D)) @ B.T

            local_best_x = x0.copy()
            local_best_fx = func(x0)
            calls_used += 1
            if local_best_fx < best_fx:
                best_fx = local_best_fx
                best_x = local_best_x.copy()

            for gen in range(1, 500):
                if time.time() > deadline:
                    break
                if calls_used + lam > max_calls:
                    rem = max_calls - calls_used
                    if rem < 1:
                        break
                    for _ in range(rem):
                        if time.time() > deadline:
                            break
                        z = rng_local.standard_normal(n)
                        y = x_mean + sigma * (C_inv_sqrt @ z)
                        y = np.clip(y, -5, 5)
                        fy = func(y)
                        calls_used += 1
                        if fy < local_best_fx:
                            local_best_fx = fy
                            local_best_x = y.copy()
                        if fy < best_fx:
                            best_fx = fy
                            best_x = y.copy()
                    break

                # Sample population
                ys = np.zeros((lam, n))
                fs = np.zeros(lam)
                for i in range(lam):
                    z = rng_local.standard_normal(n)
                    ys[i] = x_mean + sigma * (C_inv_sqrt @ z)
                    ys[i] = np.clip(ys[i], -5, 5)
                    fs[i] = func(ys[i])
                    calls_used += 1
                    if fs[i] < local_best_fx:
                        local_best_fx = fs[i]
                        local_best_x = ys[i].copy()
                    if fs[i] < best_fx:
                        best_fx = fs[i]
                        best_x = ys[i].copy()

                # Sort by fitness
                idx = np.argsort(fs)
                ys_sorted = ys[idx]

                # Update mean
                x_mean_new = np.sum(w[:, None] * ys_sorted[:mu], axis=0)

                # Update evolution paths
                y_vec = (x_mean_new - x_mean) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_vec
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < (1.4 + 2 / (n + 1)) * chi_n
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_vec

                # Update covariance
                delta_h = (1 - hsig) * c_c * (2 - c_c)
                rank_mu = np.outer(pc, pc) + delta_h * np.outer(pc, pc)
                rank_1 = np.outer(y_vec, y_vec)
                C = (1 + c_sigma * (c_cov_pos - c_sigma) + c_cov_neg * (mu_eff - 2) - c_sigma * c_cov_neg) * C \
                    + c_cov_pos * rank_1 + c_cov_neg * (np.sum(w) - 1) * rank_mu

                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                # Update sigma
                sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 100.0)

                # Update C^{-1/2}
                D, B = np.linalg.eigh(C)
                D = np.maximum(D, 1e-12)
                C_inv_sqrt = B @ np.diag(1.0 / np.sqrt(D)) @ B.T

                x_mean = x_mean_new

            return local_best_x, local_best_fx

        def lbfgsb_polish(x0, max_calls, deadline):
            """Run L-BFGS-B polish from x0."""
            nonlocal best_x, best_fx, calls_used

            if max_calls < 3:
                return

            nm_func_calls = [0]

            def lbfgsb_func(x):
                nm_func_calls[0] += 1
                if nm_func_calls[0] > max_calls:
                    raise RuntimeError("Budget exceeded")
                x_clipped = np.clip(x, -5, 5)
                return func(x_clipped)

            try:
                result = minimize(lbfgsb_func, x0, method='L-BFGS-B',
                                  bounds=[(-5, 5)] * n,
                                  options={'maxiter': max_calls, 'maxfun': max_calls,
                                           'gtol': 1e-8, 'ftol': 1e-10})
                if result.fun < best_fx:
                    best_fx = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

            calls_used += nm_func_calls[0]

        # Portfolio: run multiple CMA-ES instances with diverse starts
        # 4 instances, each gets ~25% of the main budget
        num_instances = 4
        per_instance_budget = int((budget - 50) / num_instances)  # reserve 50 for polish
        rng_starts = np.random.default_rng(123)

        for inst in range(num_instances):
            if time.time() > deadline - 10:
                break
            if calls_used >= budget - 50:
                break

            # Diverse initializations
            if inst == 0:
                x0 = np.zeros(n)
                sigma0 = 2.0
            elif inst == 1:
                x0 = rng_starts.uniform(-3, 3, size=n)
                sigma0 = 1.5
            elif inst == 2:
                x0 = rng_starts.uniform(-3, 3, size=n)
                sigma0 = 3.0
            else:
                x0 = rng_starts.uniform(-4, 4, size=n)
                sigma0 = 2.5

            rng_local = np.random.default_rng(1000 + inst)
            remaining_budget = budget - 50 - calls_used
            cma_es_instance(x0, sigma0, remaining_budget, deadline, rng_local)

        # Phase 2: L-BFGS-B polish with remaining budget
        remaining = budget - calls_used
        if remaining > 5 and time.time() < deadline - 5:
            lbfgsb_polish(best_x.copy(), remaining, deadline)

        return best_x, best_fx
