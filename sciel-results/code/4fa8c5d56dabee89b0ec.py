"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic global sampling + CMA-ES local refinement for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s kill
        n = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = float('inf')

        def safe_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return float('inf')
            calls += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = np.array(x)
            return fx

        # Phase 1: Deterministic global sampling for early progress
        # 1. Center
        # 2. All 2^5 = 32 vertices of [-5, 5]^5
        # 3. 10 axis midpoints (±2.5 on each axis)
        points_to_eval = []

        # Center
        points_to_eval.append(np.zeros(n))

        # Vertices: all combinations of -5 and 5
        for mask in range(2 ** n):
            v = np.array([5.0 if (mask >> i) & 1 else -5.0 for i in range(n)])
            points_to_eval.append(v)

        # Axis midpoints
        for i in range(n):
            for sign in [-1, 1]:
                mid = np.zeros(n)
                mid[i] = sign * 2.5
                points_to_eval.append(mid)

        # Evaluate these points deterministically
        for p in points_to_eval:
            if calls >= budget or time.time() > deadline - 30:
                break
            safe_call(p)

        # Fallback
        if best_x is None:
            safe_call(np.zeros(n))

        # Phase 2: CMA-ES refinement from the best point found
        if best_x is not None and calls < budget and time.time() < deadline - 10:
            # CMA-ES parameters
            lam = 12
            mu = lam // 2

            # Initialize mean at the best point found
            m = best_x.copy()
            # Initial step size - moderate to explore around the best point
            sigma = 1.5

            # Covariance matrix
            C = np.eye(n)

            # Eigen decomposition for efficient sampling
            D = np.ones(n)
            B = np.eye(n)

            # Paths for sigma adaptation
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            # Weights for the mean update
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)

            # Effective population size
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Learning rates
            c_sigma = (mu_eff + 2) / (n + 4)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 2 / mu_eff) / ((n + 2) ** 2 + mu_eff)))

            # Damping for sigma update
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # Main CMA-ES loop
            while calls < budget and time.time() < deadline - 5:
                m_old = m.copy()

                # Sample population
                X = np.zeros((lam, n))
                for i in range(lam):
                    z = np.random.randn(n)
                    y = B @ (D * z)
                    x = m + sigma * y
                    X[i] = x

                # Evaluate population
                F = np.full(lam, float('inf'))
                for i in range(lam):
                    if calls >= budget or time.time() > deadline - 5:
                        break
                    x_eval = np.clip(X[i], -5, 5)
                    F[i] = safe_call(x_eval)

                if calls >= budget:
                    break

                # Sort by fitness
                idx = np.argsort(F)
                X = X[idx]
                F = F[idx]

                # Update mean
                m_new = np.zeros(n)
                for i in range(mu):
                    m_new += weights[i] * X[i]
                m = m_new

                # Update covariance matrix and paths
                Y = (X[:mu] - m_old) / sigma

                # Update p_c
                try:
                    C_inv_m = np.linalg.solve(C, (m - m_old))
                except np.linalg.LinAlgError:
                    C_inv_m = np.linalg.pinv(C) @ (m - m_old)
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_m / sigma

                # Hessian test for sigma update
                gen = (calls / lam)
                h_sig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * gen))) < \
                        (1.4 + 2 / (n + 1)) * chi_n
                h_c = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (2 * gen))) < \
                      (1.4 + 2 / (n + 1)) * chi_n

                # Update C
                delta_h = (1 - h_c) * c_c * (2 - c_c) * np.linalg.norm(p_c) ** 2
                RankMu = np.zeros((n, n))
                for i in range(mu):
                    RankMu += weights[i] * np.outer(Y[i], Y[i])

                RankOne = np.outer(p_c, p_c) + delta_h

                C = (1 - c_1 - c_mu) * C + c_1 * RankOne + c_mu * RankMu

                # Update sigma
                p_sigma = (1 - c_sigma) * p_sigma + h_sig * np.sqrt(
                    c_sigma * (2 - c_sigma) * mu_eff) * (m - m_old) / sigma
                sigma = sigma * np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))

                # Update eigen decomposition
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    D = np.sqrt(eigvals)
                    B = eigvecs
                except np.linalg.LinAlgError:
                    D = np.ones(n)
                    B = np.eye(n)
                    C = np.eye(n)

        return best_x, best_f
# EVOLVE-BLOCK-END
