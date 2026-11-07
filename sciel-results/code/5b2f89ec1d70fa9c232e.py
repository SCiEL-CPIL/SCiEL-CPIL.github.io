"""CMA-ES optimizer for BBOB-style f19 (Griewank-Rosenbrock)."""
import numpy as np
import time


class Optimizer:
    """CMA-ES with covariance matrix adaptation for rotated/shifted landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # 10s safety margin before 300s limit

        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Population size
        lam = max(10, int(4 + 3 * np.log(max(n, 2))))

        # CMA-ES parameters
        cc = (4 + n / (n + 4)) / ((n + 1.3) ** 2 + 4)
        cs = (n + 1.7) / (n + n + 5.5)
        cmu = min(1 - 2 * cc, 0.5 * (n - 2) / (n + 2))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Initialize
        m = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)

        best_x = m.copy()
        best_f = float('inf')
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = float(func(x_clipped))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # Initial evaluation
        evaluate(m)

        while calls < budget:
            if time.time() > deadline:
                break

            remaining = budget - calls
            if remaining < lam:
                # Use remaining budget for single steps
                while calls < budget and time.time() < deadline:
                    step = sigma * rng.multivariate_normal(np.zeros(n), C)
                    x_new = np.clip(m + step, -5.0, 5.0)
                    f_new = evaluate(x_new)
                    if f_new < best_f:
                        m = x_new.copy()
                    sigma *= 0.9
                break

            # Eigendecomposition of C
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            D = np.sqrt(eigvals)
            B = eigvecs

            # Generate population
            Y = []
            X = []
            for i in range(lam):
                z = rng.standard_normal(n)
                y = B @ (D * z)
                x = np.clip(m + sigma * y, -5.0, 5.0)
                Y.append(y)
                X.append(x)

            # Evaluate population
            F = [evaluate(x) for x in X]

            # Sort by fitness
            order = np.argsort(F)
            X_sorted = [X[i] for i in order]
            Y_sorted = [Y[i] for i in order]

            # Update mean
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1 / np.sum(weights ** 2)

            m_old = m.copy()
            m = sum(w * X_sorted[i] for i, w in enumerate(weights))

            # Transform mean shift to eigen-space
            delta = (m - m_old) / sigma
            delta_eig = B.T @ (delta / D)

            # Update ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * delta_eig
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (calls + 1))) < (1.4 + 2 / (n + 1)) * chi_n

            # Update pc
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * delta_eig

            # Update C
            rank_one = np.outer(pc, pc)
            rank_mu = sum(
                weights[i] * np.outer(Y_sorted[i] / D, Y_sorted[i] / D)
                for i in range(mu)
            )
            # Standard CMA-ES: C = (1 + 2*mu_eff*(mu_eff - n)) * C + cc * rank_one + cmu * rank_mu
            decay = 1 + 2 * mu_eff * (mu_eff - n)
            C = decay * C + cc * rank_one + cmu * rank_mu

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            C += 1e-10 * np.eye(n)

            # Update sigma
            sigma *= np.exp(cs / (5 * n) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)

            if sigma < 1e-12:
                break

        # Final polish: Nelder-Mead on best point if budget remains
        if calls < budget and time.time() < deadline:
            try:
                from scipy.optimize import minimize
                best_x_copy = best_x.copy()
                result = minimize(
                    lambda x: float(func(np.clip(x, -5.0, 5.0))),
                    best_x_copy,
                    method='Nelder-Mead',
                    options={'maxiter': budget - calls, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if result.fun < best_f:
                    best_x = np.clip(result.x, -5.0, 5.0)
                    best_f = result.fun
            except Exception:
                pass

        return best_x, best_f
