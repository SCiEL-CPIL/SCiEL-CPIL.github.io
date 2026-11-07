"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-restart for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 325.0  # safety margin before 330s kill

        best_x = None
        best_f = np.inf
        call_count = 0

        def eval_point(x):
            nonlocal call_count
            if time.time() > deadline:
                return None
            call_count += 1
            return float(func(np.asarray(x, dtype=np.float64)))

        # Multi-restart CMA-ES
        rng = np.random.default_rng(42)
        n_restarts = 5
        budget_per_restart = max(1, (budget - 5) // n_restarts)
        remaining = budget - 5

        for r in range(n_restarts):
            if call_count >= budget or time.time() > deadline:
                break

            # Initialize near best found, or random for first restart
            if best_x is not None and r > 0:
                x0 = np.clip(best_x + rng.normal(0, 1.5, dim), -5, 5)
            else:
                x0 = rng.uniform(-5, 5, dim)

            # CMA-ES parameters
            lam = 4 + int(3 * np.log(dim))  # population size
            sigma = 2.0
            c1 = (4 - lam / (dim + 1)) / ((dim + 2) ** 2 + 2)
            cmu = min(1 - c1 - 0.5, 2 * ((2 - (dim + 1) / (dim + 3)) ** 2 - 1) / ((dim + 2) ** 2 + 4))
            cs = (lam / 4 + 2) / (dim + lam / 2 + 5)
            c_sigma = cs * (1 + 2 * max(0, np.sqrt((lam / dim - 1) / (dim + 1)) - 1) + cs)
            d_sigma = 1 + 2 * max(0, np.sqrt((lam / dim - 1) / (dim + 1)) - 1) + cs
            chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            steps = 0

            # Initial evaluation
            fx = eval_point(x0)
            if fx is None:
                break
            if fx < best_f:
                best_f, best_x = fx, x0.copy()

            # Budget for this restart
            restart_budget = min(budget_per_restart, budget - call_count)
            if r == n_restarts - 1:
                restart_budget = budget - call_count

            while call_count < budget - 2 and time.time() < deadline and steps < restart_budget:
                # Generate population
                B = rng.standard_normal((lam, dim))
                # Transform with C^(1/2) via eigenvalue decomposition (approx)
                # For speed, use Cholesky of C
                try:
                    L = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
                except np.linalg.LinAlgError:
                    L = np.eye(dim)

                Y = (x0 + sigma * (B @ L.T))
                # Clip to bounds
                Y = np.clip(Y, -5, 5)

                # Evaluate
                F = []
                for i in range(lam):
                    fi = eval_point(Y[i])
                    if fi is None:
                        break
                    F.append(fi)

                if len(F) < lam:
                    break

                F = np.array(F)
                idx = np.argsort(F)
                Y_sorted = Y[idx]
                F_sorted = F[idx]

                # Update best
                if F_sorted[0] < best_f:
                    best_f, best_x = F_sorted[0], Y_sorted[0].copy()

                # Weighted mean
                mu = lam // 2
                w = np.log(np.arange(1, mu + 1)) - np.log(mu)
                w = w / w.sum()
                mueff = 1 / np.sum(w ** 2)

                m = np.sum(w[:, None] * Y_sorted[:mu], axis=0)

                # Update evolution paths
                pc = (1 - cs) * pc + cs * np.sqrt(mueff) * np.linalg.inv(L) @ (m - x0) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * np.linalg.inv(L) @ (m - x0) / sigma

                # Update covariance
                delta_sig = (np.linalg.norm(ps) / chi - 1) * c_sigma
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (delta_sig * (1 + delta_sig) - 1) * C)
                C += cmu * np.sum(w[:, None] * np.outer(Y_sorted[:mu] - m, Y_sorted[:mu] - m), axis=0)

                # Ensure symmetry and positive semi-definite
                C = (C + C.T) / 2
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                except np.linalg.LinAlgError:
                    C = np.eye(dim)

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

                # Update mean
                x0 = m

                # Check for convergence / restart
                if sigma < 1e-6 or np.all(np.abs(np.diag(C)) < 1e-10):
                    break

                steps += 1

        # Final polish: fine local search around best
        if best_x is not None and call_count < budget - 3 and time.time() < deadline:
            sigma_polish = 0.05
            for _ in range(3):
                y = np.clip(best_x + sigma_polish * rng.standard_normal(dim), -5, 5)
                fy = eval_point(y)
                if fy is None:
                    break
                if fy < best_f:
                    best_f, best_x = fy, y.copy()
                else:
                    sigma_polish *= 0.5

        return best_x, best_f
# EVOLVE-BLOCK-END
