import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin for 330s limit
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        calls = 0
        fbest = np.inf
        x_best = np.zeros(n)

        def time_ok():
            return time.time() < deadline

        def safe_call(x):
            nonlocal calls, fbest, x_best
            if calls >= budget or not time_ok():
                return fbest
            x = np.clip(x, -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                return fbest
            calls += 1
            if fx < fbest:
                fbest = fx
                x_best = x.copy()
            return fx

        # Phase 1: CMA-ES with adaptive restarts
        cma_budget = int(budget * 0.55)
        
        def run_cma_es(x_start, sigma_start, max_calls):
            nonlocal calls, fbest, x_best
            if not time_ok() or calls >= max_calls:
                return
            
            lam = 8
            mu = lam // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            x = x_start.copy()
            sigma = sigma_start
            pc = np.zeros(n)
            ps = np.zeros(n)
            C = np.eye(n)
            C_sqrt = np.eye(n)
            inv_C_sqrt = np.eye(n)
            no_improve_gens = 0
            gen = 0
            fbest_local = fbest

            while calls < max_calls and time_ok() and (deadline - time.time()) > 10:
                offspring = []
                for i in range(lam):
                    if calls >= max_calls or not time_ok():
                        break
                    z = rng.standard_normal(n)
                    y = C_sqrt @ z
                    xi = x + sigma * y
                    fi = safe_call(xi)
                    offspring.append((fi, xi, z))

                if len(offspring) < 2:
                    break

                offspring.sort(key=lambda t: t[0])

                if offspring[0][0] < fbest_local:
                    fbest_local = offspring[0][0]
                    no_improve_gens = 0
                else:
                    no_improve_gens += 1

                # Restart if stuck
                if no_improve_gens > 15 and sigma < 0.2:
                    break

                # Update mean
                x_old = x.copy()
                x = np.zeros(n)
                for i in range(mu):
                    x += weights[i] * offspring[i][1]

                if sigma > 1e-12:
                    y_mean = (x - x_old) / sigma
                else:
                    y_mean = np.zeros(n)

                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_C_sqrt @ y_mean)
                ps_norm = np.linalg.norm(ps)
                h_sigma = 1.0 if ps_norm / np.sqrt(1 - (1 - c_sigma) ** (gen + 1)) < (1.4 + 2 / (n + 1)) * chi_n else 0.0
                if h_sigma:
                    sigma *= np.exp(c_sigma / d_sigma * (ps_norm / chi_n - 1))

                pc = (1 - c_c) * pc + c_c * y_mean
                delta = (1 + 2 * max(0, mu_eff / n - 1)) * c_c * (1 - c_c)
                rank_one = np.outer(pc, pc)
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    rank_mu += weights[i] * np.outer(offspring[i][2], offspring[i][2])
                C = (1 + c_1 * delta - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

                C = (C + C.T) / 2
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                    inv_C_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    C = np.eye(n)
                    C_sqrt = np.eye(n)
                    inv_C_sqrt = np.eye(n)

                sigma = np.clip(sigma, 1e-4, 10.0)
                gen += 1

        # Initial evaluation
        safe_call(np.zeros(n))
        
        # Run CMA-ES
        run_cma_es(np.zeros(n), 2.0, cma_budget)

        # Phase 2: L-BFGS-B from multiple starts
        if calls < budget and time_ok() and (deadline - time.time()) > 20:
            # Generate diverse starting points
            n_starts = 6
            starts = [x_best.copy()]
            
            for i in range(1, n_starts):
                if calls >= budget or not time_ok():
                    break
                # Mix of local and global perturbations
                scale = 0.5 + i * 0.3
                x_try = x_best + scale * rng.standard_normal(n)
                x_try = np.clip(x_try, -5, 5)
                starts.append(x_try)
            
            # Also add some random points
            for i in range(3):
                if calls >= budget or not time_ok():
                    break
                x_try = rng.uniform(-4, 4, n)
                starts.append(x_try)
            
            # Run L-BFGS-B from each start
            for start_x in starts:
                if calls >= budget or not time_ok() or (deadline - time.time()) < 10:
                    break
                
                try:
                    def wrapped_func(x):
                        nonlocal calls
                        if calls >= budget or not time_ok():
                            return fbest
                        x = np.clip(x, -5.0, 5.0)
                        try:
                            fx = float(func(x))
                        except Exception:
                            return fbest
                        calls += 1
                        if fx < fbest:
                            fbest = fx
                            x_best = x.copy()
                        return fx

                    result = scipy.optimize.minimize(
                        wrapped_func, start_x, method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * n,
                        options={'maxiter': 50, 'gtol': 1e-6, 'ftol': 1e-8}
                    )
                    if result.fun < fbest:
                        fbest = result.fun
                        x_best = result.x.copy()
                except Exception:
                    pass

        # Phase 3: Final fine-tuning with small perturbations
        if calls < budget and time_ok() and (deadline - time.time()) > 5:
            remaining = min(20, budget - calls)
            for _ in range(remaining):
                if not time_ok() or calls >= budget:
                    break
                for scale in [0.01, 0.005, 0.001]:
                    if calls >= budget or not time_ok():
                        break
                    perturbation = rng.standard_normal(n) * scale
                    x_try = x_best + perturbation
                    x_try = np.clip(x_try, -5, 5)
                    f_try = safe_call(x_try)

        return x_best, fbest
