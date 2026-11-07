import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 290  # Safety margin before 300s limit
        d = self.dim
        max_calls = self.budget

        best_x = None
        best_f = float('inf')
        calls_used = 0

        def try_eval(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= max_calls:
                return None
            x = np.asarray(x, dtype=np.float64).flatten()
            if x.shape[0] != d:
                return None
            try:
                fx = float(func(x))
            except Exception:
                return None
            calls_used += 1
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def cma_es(func, budget, dim, start_x=None, sigma0=None, max_time=None):
            nonlocal calls_used, best_x, best_f
            d = dim
            budget = budget
            rng = np.random.default_rng()

            # CMA-ES parameters
            lam = 4 + int(3 * np.log(d))
            mu = lam // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1 / np.sum(weights**2)

            # Step size control
            c_sigma = (mueff + 2) / (d + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + c_sigma
            cs = (mueff + 1.3) / (d + mueff + 2)
            ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + cs

            # Covariance matrix adaptation
            c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
            c_1 = 2 / ((d + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff))

            # Initial state
            if start_x is not None:
                m = start_x.copy()
            else:
                m = np.zeros(d)
            C = np.eye(d)
            sigma = sigma0 if sigma0 is not None else 2.0
            p_sigma = np.zeros(d)
            p_c = np.zeros(d)

            Csqrt = np.linalg.cholesky(C)
            Cinv_sqrt = np.linalg.inv(Csqrt)

            evals_in_this_run = 0

            while evals_in_this_run < budget:
                # Check time
                if max_time is not None and time.time() > max_time:
                    break

                remaining = budget - evals_in_this_run
                if remaining < lam:
                    # Do a simple local search with remaining budget
                    x = best_x if best_x is not None else m
                    fx = best_f if best_x is not None else float('inf')
                    sigma_local = sigma * 0.5
                    while evals_in_this_run < budget:
                        if max_time is not None and time.time() > max_time:
                            break
                        y = np.clip(x + sigma_local * rng.standard_normal(d), -5, 5)
                        fy = try_eval(y)
                        if fy is None:
                            break
                        evals_in_this_run += 1
                        if fy < fx:
                            x, fx = y, fy
                            sigma_local *= 1.5
                        else:
                            sigma_local *= 0.85
                    break

                # Sample population
                Y = np.zeros((lam, d))
                F = np.zeros(lam)
                for i in range(lam):
                    z = rng.standard_normal(d)
                    Y[i] = np.clip(m + sigma * Csqrt @ z, -5, 5)
                    fy = try_eval(Y[i])
                    if fy is None:
                        break
                    evals_in_this_run += 1
                    F[i] = fy

                if evals_in_this_run < budget - lam:
                    # Not enough evaluations completed for full generation, break
                    if evals_in_this_run < lam:
                        break

                # Sort
                order = np.argsort(F)
                Y_sel = Y[order[:mu]]
                F_sel = F[order[:mu]]

                # Update mean
                m_old = m.copy()
                m = np.sum(weights[:, None] * Y_sel, axis=0)

                # Update step size
                try:
                    y_sigma = Cinv_sqrt @ ((m - m_old) / sigma)
                except Exception:
                    y_sigma = (m - m_old) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_sigma
                norm_p_sigma = np.linalg.norm(p_sigma)
                chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
                sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))

                # Update covariance matrix
                y_c = (m - m_old) / sigma
                p_c = (1 - cs) * p_c + np.sqrt(cs * (2 - cs) * mueff) * y_c

                delta_heavy = (1 - cs)**2 * (np.linalg.norm(p_c)**2 / d - 1)
                C = (1 + c_c * (c_c + 2 * c_1) * delta_heavy - c_1 - c_mu) * C \
                    + c_1 * (np.outer(p_c, p_c) + (1 - cs)**2 * (np.linalg.norm(p_c)**2 / d - 1) * C) \
                    + c_mu * np.sum(weights[:, None, None] * np.outer(Y_sel - m_old, Y_sel - m_old), axis=0)

                # Ensure symmetry and positive semi-definiteness
                C = (C + C.T) / 2
                try:
                    Csqrt = np.linalg.cholesky(C + 1e-12 * np.eye(d))
                except np.linalg.LinAlgError:
                    C = C + 1e-6 * np.eye(d)
                    try:
                        Csqrt = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        C = np.eye(d)
                        Csqrt = np.eye(d)

                Cinv_sqrt = np.linalg.inv(Csqrt)

                # Check for convergence
                if sigma < 1e-15 or np.all(np.abs(m) > 10):
                    break

            return evals_in_this_run

        # IPOP-CMA-ES with restarts
        # Phase 1: Initial exploration with larger sigma
        sigma_start = 2.0
        budget_per_restart = max(50, max_calls // 10)
        num_restarts = min(10, max_calls // budget_per_restart)

        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            if calls_used >= max_calls:
                break

            remaining_budget = max_calls - calls_used
            if remaining_budget < 20:
                break

            # For first restart, use larger sigma for exploration
            if restart == 0:
                start_x = None
                sigma0 = sigma_start
            else:
                # Restart from current best with reduced sigma
                if best_x is not None:
                    start_x = best_x.copy()
                else:
                    start_x = np.zeros(d)
                sigma0 = sigma_start * (0.5 ** restart)

            # Use a fraction of remaining budget for this restart
            b = min(budget_per_restart, remaining_budget)
            cma_es(func, b, d, start_x=start_x, sigma0=sigma0, max_time=deadline)

            # If we've used most of our budget, do a final local search
            if calls_used >= max_calls - 20:
                break

        # Final local search with remaining budget
        if calls_used < max_calls:
            if best_x is not None:
                x = best_x.copy()
                fx = best_f
            else:
                x = np.zeros(d)
                fx = try_eval(x)
                if fx is None:
                    return best_x, best_f
                x = x.copy()
                fx = best_f

            sigma_local = 0.5
            while calls_used < max_calls:
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma_local * np.random.randn(d), -5, 5)
                fy = try_eval(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma_local *= 1.5
                else:
                    sigma_local *= 0.85

        return best_x, best_f
