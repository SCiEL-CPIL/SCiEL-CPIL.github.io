import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 290.0
        n = self.dim
        B = self.budget

        best_x = np.zeros(n)
        best_f = float('inf')
        calls = 0

        def time_ok():
            return (time.time() - start) < deadline

        def ev(x):
            nonlocal calls, best_x, best_f
            if calls >= B:
                return float('inf')
            calls += 1
            x = np.clip(x, -5, 5)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Initial sampling
        n_init = min(20, B)
        for i in range(n_init):
            if not time_ok() or calls >= B:
                break
            x = rng.uniform(-5, 5, size=n)
            ev(x)

        # Phase 2: CMA-ES-like search
        # Population size
        lam = max(10, int(4 + 3 * np.log(n)))
        # Initial sigma
        sigma = 2.0
        # Initial mean
        mean = best_x.copy()
        # Covariance matrix
        C = np.eye(n)
        # Step size
        sigma = 2.0

        # CMA-ES parameters
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

        # State
        pc = np.zeros(n)
        ps = np.zeros(n)

        # Loop
        while calls < B and time_ok():
            # Sample
            B_mat = np.linalg.cholesky(C)
            z = rng.standard_normal((lam, n))
            z_norm = np.linalg.norm(z, axis=1, keepdims=True)
            z = z / z_norm
            offspring = mean[None, :] + sigma * (z @ B_mat.T)

            # Evaluate
            offspring_f = np.array([ev(x) for x in offspring])

            # Sort
            order = np.argsort(offspring_f)
            offspring = offspring[order]
            offspring_f = offspring_f[order]

            # Update
            y_w = np.zeros(n)
            for i in range(mu):
                y_w += weights[i] * offspring[i]

            # Update mean
            mean = mean + c_sigma * (y_w / sigma - mean)

            # Update covariance
            ys = np.zeros((mu, n))
            for i in range(mu):
                ys[i] = (offspring[i] - mean) / sigma

            # Update pc
            pc = (1 - c_sigma) * pc + c_sigma * d_sigma * y_w

            # Update ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * B_mat.T @ y_w

            # Rank-one and rank-mu update
            C = (1 + c_1 * c_sigma * (2 - c_sigma) - c_1) * C \
                + c_1 * (np.outer(pc, pc) - C) \
                + c_mu * (np.outer(ys.T @ weights, ys) - C)

            # Symmetrize
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Restart if sigma is too large or too small
            if sigma > 1e5 or sigma < 1e-10:
                # Reset CMA-ES
                sigma = 2.0
                mean = best_x.copy()
                C = np.eye(n)
                pc = np.zeros(n)
                ps = np.zeros(n)

            # If no improvement for a while, try a different strategy
            if np.all(offspring_f[:mu] >= best_f - 1e-8):
                # Try a local search around best_x
                x0 = best_x.copy()
                for _ in range(5):
                    if not time_ok() or calls >= B:
                        break
                    x = x0 + rng.normal(0, 0.1, size=n)
                    ev(x)

        # Phase 3: Local polish
        if calls < B and time_ok():
            # Use Nelder-Mead
            from scipy.optimize import minimize

            def obj(x):
                if not time_ok() or calls >= B:
                    return float('inf')
                return ev(x)

            try:
                res = minimize(obj, best_x, method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            except Exception:
                pass

        return best_x, best_f
