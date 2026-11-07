"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic Halton exploration + CMA-ES local refinement for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 270  # safety margin before 300s kill
        budget = self.budget
        dim = self.dim

        def time_left():
            return deadline - time.time()

        calls = 0
        best_x = None
        best_f = np.inf

        def eval_point(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            calls += 1
            x = np.clip(x, -5, 5)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # --- Phase 1: Halton-based global exploration ---
        # Use first 80 Halton points for good space-filling coverage
        n_halton = 80
        bases = [2, 3, 5, 7, 11]  # first dim primes

        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_seq(n, dim, bases):
            """Generate n points in [0,1)^dim using Halton sequence."""
            pts = []
            for i in range(1, n + 1):
                pt = np.array([halton(i, bases[d]) for d in range(dim)])
                pts.append(pt)
            return np.array(pts)

        # Map [0,1)^dim to [-5,5]^dim
        def to_box(pt):
            return pt * 10.0 - 5.0

        halton_pts_unit = halton_seq(n_halton, dim, bases)
        halton_pts = np.array([to_box(p) for p in halton_pts_unit])

        # Also include origin and a few strategic points
        extra_pts = [np.zeros(dim)]
        # Add midpoints of axes
        for d in range(dim):
            for s in [-1, 1]:
                pt = np.zeros(dim)
                pt[d] = s * 2.5
                extra_pts.append(pt)

        all_explore_pts = list(halton_pts) + extra_pts

        for pt in all_explore_pts:
            if calls >= budget or time_left() < 1.0:
                break
            pt = np.clip(pt, -5, 5)
            f = eval_point(pt)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = eval_point(best_x)

        # --- Phase 2: CMA-ES refinement from the best point found ---
        if best_x is not None and calls < budget and time_left() > 10:
            # CMA-ES parameters
            lam = 12
            mu = lam // 2

            # Initialize mean at the best point found
            m = best_x.copy()
            # Initial step size - moderate to explore around the best point
            sigma = 1.5

            # Covariance matrix
            C = np.eye(dim)

            # Eigen decomposition for efficient sampling
            D = np.ones(dim)
            B = np.eye(dim)

            # Paths for sigma adaptation
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            # Weights for the mean update
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)

            # Effective population size
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Learning rates
            c_sigma = (mu_eff + 2) / (dim + 4)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * ((mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff)))

            # Damping for sigma update
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # Main CMA-ES loop
            while calls < budget and time_left() > 5:
                m_old = m.copy()

                # Sample population
                X = np.zeros((lam, dim))
                for i in range(lam):
                    z = np.random.randn(dim)
                    y = B @ (D * z)
                    x = m + sigma * y
                    X[i] = x

                # Evaluate population
                F = np.full(lam, np.inf)
                for i in range(lam):
                    if calls >= budget or time_left() < 5:
                        break
                    x_eval = np.clip(X[i], -5, 5)
                    F[i] = eval_point(x_eval)

                if calls >= budget:
                    break

                # Sort by fitness
                idx = np.argsort(F)
                X = X[idx]
                F = F[idx]

                # Update mean
                m_new = np.zeros(dim)
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
                        (1.4 + 2 / (dim + 1)) * chi_n
                h_c = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (2 * gen))) < \
                      (1.4 + 2 / (dim + 1)) * chi_n

                # Update C
                delta_h = (1 - h_c) * c_c * (2 - c_c) * np.linalg.norm(p_c) ** 2
                RankMu = np.zeros((dim, dim))
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
                    D = np.ones(dim)
                    B = np.eye(dim)
                    C = np.eye(dim)

        # --- Phase 3: Final refinement with coordinate descent on best_x ---
        if calls < budget and time_left() > 2.0:
            # Fine coordinate descent from best_x
            for step_size in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]:
                if calls >= budget or time_left() < 1.0:
                    break
                improved = False
                for _pass in range(2):
                    if calls >= budget or time_left() < 1.0:
                        break
                    for d in range(dim):
                        if calls >= budget or time_left() < 0.5:
                            break
                        for s in [-1, 1]:
                            if calls >= budget or time_left() < 0.5:
                                break
                            pt = best_x.copy()
                            pt[d] += s * step_size
                            pt = np.clip(pt, -5, 5)
                            f = eval_point(pt)
                            if f < best_f:
                                improved = True
                    if not improved:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
