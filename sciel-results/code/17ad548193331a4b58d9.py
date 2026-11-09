"""CMA-ES optimizer for BBOB-style f11 (Discus)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES with covariance adaptation and a wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 295.0   # safety margin under the 330s kill
        n = self.dim
        rng = np.random.default_rng(12345)

        # --- Initialization ---
        x = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        Cs = np.zeros((n, 1))
        ps = np.zeros(n)
        pc = np.zeros(n)
        best_x = x.copy()
        best_f = np.inf

        # --- Hyperparameters ---
        mu = max(1, (n + 1) // 2)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / w.sum()
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        def evaluate(xk):
            fx = func(np.clip(xk, -5, 5))
            return fx

        def update_state(x, fx):
            nonlocal C, Cs, ps, pc, best_x, best_f
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # First evaluation
        f0 = evaluate(x)
        update_state(x, f0)
        calls = 1

        while calls < self.budget:
            if time.time() >= deadline:
                break

            # --- Sample offspring ---
            B = np.linalg.cholesky(C)
            offspring = []
            for _ in range(mu):
                z = rng.standard_normal(n)
                y = x + sigma * (B @ z)
                offspring.append(y)

            # --- Evaluate ---
            vals = []
            for y in offspring:
                fy = evaluate(y)
                vals.append(fy)
                calls += 1
                if time.time() >= deadline:
                    break
            if calls >= self.budget:
                break

            # Sort by fitness
            order = np.argsort(vals)
            xs = [offspring[i] for i in order]
            fs = [vals[i] for i in order]
            xnew = xs[0]
            fnew = fs[0]
            update_state(xnew, fnew)

            # --- Update step-size control variable ps ---
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) \
                 * (B.T @ (xnew - x) / sigma)

            # --- Update evolution path pc and covariance C ---
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - 1))) < \
               (1.4 + 2 / (n + 1)) * chi_n:
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (xnew - x) / sigma
                delta_sig = c_c * (np.linalg.norm(pc) ** 2 - n) / (2 * n)
                C = (1 + c_1 + c_mu - c_1 - c_mu) * C
                C = (1 - c_1 - c_mu) * C
                C += c_1 * (np.outer(pc, pc) + (np.linalg.norm(pc) ** 2 - n) * np.eye(n))
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    rank_mu += w[i] * np.outer(xs[i] - x, xs[i] - x)
                C += c_mu * rank_mu
                # Ensure symmetry & positive-definiteness
                C = (C + C.T) / 2.0
                eigval, eigvec = np.linalg.eigh(C)
                eigval = np.maximum(eigval, 1e-12)
                C = eigvec @ np.diag(eigval) @ eigvec.T
                C = (C + C.T) / 2.0
                sigma *= np.exp((delta_sig / (5 * n)) * min(1, 10 / (calls - 1)))
            else:
                # Hessian damping: avoid rank-1 update when ps is large
                C = (1 + c_1 + c_mu - c_1 - c_mu) * C
                C = (1 - c_1 - c_mu) * C
                C += c_1 * (np.outer(pc, pc) + (np.linalg.norm(pc) ** 2 - n) * np.eye(n))
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    rank_mu += w[i] * np.outer(xs[i] - x, xs[i] - x)
                C += c_mu * rank_mu
                C = (C + C.T) / 2.0
                eigval, eigvec = np.linalg.eigh(C)
                eigval = np.maximum(eigval, 1e-12)
                C = eigvec @ np.diag(eigval) @ eigvec.T
                C = (C + C.T) / 2.0

            # --- Move mean ---
            x = xnew

            # --- Stopping: sigma too large ---
            if sigma > 1e6:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
