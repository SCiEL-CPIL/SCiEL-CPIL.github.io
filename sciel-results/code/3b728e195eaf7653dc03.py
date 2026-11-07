import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for BBOB f22 (Gallagher 21 peaks) with multi-start strategy."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 295.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def safe_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f if np.isfinite(best_f) else 1e10
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                fx = 1e10
            if not np.isfinite(fx):
                fx = 1e10
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse global exploration with Latin Hypercube Sampling
        n_lhs = 50
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            strata = np.linspace(0, 1, n_lhs + 1)
            offsets = rng.uniform(0, 1.0 / n_lhs, n_lhs)
            lhs_points[:, d] = (strata[:n_lhs] + offsets) * 10.0 - 5.0

        for pt in lhs_points:
            if time.time() > deadline or calls >= budget:
                break
            safe_call(pt)

        # Collect evaluated points for ranking
        # We'll track a few good points from LHS
        # Since we only track best, let's also store some points
        # Actually, let's just use best_x and generate diverse starts

        # Phase 2: Multi-start CMA-ES
        # Proper CMA-ES implementation
        def cma_es_run(x0, max_evals):
            nonlocal calls, best_x, best_f
            if time.time() > deadline or calls >= budget - 5:
                return

            n = dim
            x = np.array(x0, dtype=float).copy()
            x = np.clip(x, -5, 5)
            f_x = safe_call(x)

            # CMA-ES parameters
            pop = max(4, int(4 + 3 * np.log(n)))
            if pop > max_evals // 2:
                pop = max(4, max_evals // 2)
            mu = pop // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            sigma = 1.0
            C = np.eye(n)
            C_inv = np.eye(n)
            C_sqrt = np.eye(n)  # We'll use a simplified approach

            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            c_sigma = (n + 2) / (n + 5)
            c_c = (4 + n / 4) / (n + 4 + 2 * n / (n + 4))
            d_sigma = 1 + 2 * max(0, np.sqrt((n - 1) / (n + 3)) - 1) + c_sigma
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

            # For sampling, we need to generate N(0, C) samples
            # Use eigen decomposition: C = Q diag(lambda) Q^T
            # sample = Q @ (sqrt(lambda) * z)

            local_evals = 0

            for gen in range(500):
                if time.time() > deadline or calls >= budget - 2 or local_evals >= max_evals:
                    break

                # Compute C_sqrt via eigen decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))

                # Sample offspring
                offspring = []
                for i in range(pop):
                    if time.time() > deadline or calls >= budget - 2 or local_evals >= max_evals:
                        break
                    z = rng.standard_normal(n)
                    y = x + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    fy = safe_call(y)
                    local_evals += 1
                    offspring.append((fy, y))

                if len(offspring) < mu:
                    break

                # Sort and select
                offspring.sort(key=lambda t: t[0])
                selected = offspring[:mu]
                y_mean = np.average([s[1] for s in selected], axis=0, weights=weights)

                # Update
                d = y_mean - x

                # Update p_sigma
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * (C @ d) / sigma

                # Update p_c
                h_sigma = 1.0 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (local_evals + 1))) < (1 + c_sigma) * chi_n * 1.14 else 0.0
                p_c = (1 - c_c) * p_c + h_sigma * c_c * d / sigma

                # Update C
                rank_mu = np.outer(d, d)
                rank_1 = np.outer(p_c, p_c)
                C = (1 + c_1 * (mu_eff - 2) - c_1 - c_mu) * C + c_mu * rank_mu + c_1 * (1 / (1 - c_c) - 1 + 2 * h_sigma * c_c * (1 - c_c)) * rank_1

                # Symmetrize
                C = (C + C.T) / 2

                # Update sigma
                sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-6, 10.0)

                x = y_mean
                x = np.clip(x, -5, 5)

            # Final check
            if np.isfinite(f_x):
                fx_final = safe_call(x)

        # Generate diverse starts
        starts = []
        if best_x is not None:
            starts.append(best_x.copy())
            # Perturbations
            for scale in [0.5, 1.0, 2.0, 3.0]:
                s = best_x + rng.normal(0, scale, dim)
                s = np.clip(s, -5, 5)
                starts.append(s)
            # Some random
            for _ in range(3):
                starts.append(rng.uniform(-4, 4, dim))
        else:
            for _ in range(8):
                starts.append(rng.uniform(-4, 4, dim))

        # Allocate budget to CMA-ES runs
        remaining = budget - calls
        if remaining > 10:
            n_runs = len(starts)
            # Give more budget to first (best) start
            base_budget = max(10, remaining // (n_runs + 1))
            first_budget = min(remaining - base_budget * (n_runs - 1), int(remaining * 0.4))
            if first_budget < base_budget:
                first_budget = base_budget

            cma_budgets = [first_budget] + [base_budget] * (n_runs - 1)
            # Adjust to fit
            total = sum(cma_budgets)
            if total > remaining - 15:
                scale = (remaining - 15) / total
                cma_budgets = [max(5, int(b * scale)) for b in cma_budgets]

            for i, (s, b) in enumerate(zip(starts, cma_budgets)):
                if time.time() > deadline or calls >= budget - 10:
                    break
                if b < 5:
                    break
                cma_es_run(s, b)

        # Phase 3: Final L-BFGS-B refinement
        if calls < budget - 15 and best_x is not None:
            from scipy.optimize import minimize
            bounds = [(-5.0, 5.0)] * dim

            def local_func(x):
                if calls >= budget or time.time() > deadline:
                    return best_f + 1e10 if np.isfinite(best_f) else 1e10
                return safe_call(x)

            try:
                res = minimize(
                    local_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 30, 'maxfun': 50, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5).copy()
            except Exception:
                pass

        # Phase 4: (1+1)-ES for remaining budget
        if calls < budget - 3 and best_x is not None:
            sigma = 0.2
            x = best_x.copy()
            fx = best_f

            for _ in range(budget - calls - 1):
                if time.time() > deadline:
                    break
                if calls >= budget - 1:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = safe_call(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.06
                else:
                    sigma *= 0.94
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
