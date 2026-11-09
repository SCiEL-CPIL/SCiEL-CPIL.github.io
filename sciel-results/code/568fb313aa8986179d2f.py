"""Rewritten optimizer for BBOB f11 (Discus) with fresh design."""
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        rng = np.random.default_rng(42)
        n = self.dim
        lam = max(4, int(4 + 3 * np.log(n)))  # population size

        # Initial point: center of box
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Initial sigma: explore a good chunk of the box
        sigma = 3.0

        # CMA-ES matrices
        C = np.eye(n)
        C_inv = np.eye(n)
        pc = np.zeros(n)
        cs = 1.0 / (n + 3)
        cc = (n + 2) / (n / 2 + 2)
        cmu = min(0.2, (lam - 2 + 1) / (n + 2))
        damps = (1 + 2 * max(0, np.sqrt((lam - 1) / (n + 1)) - 1) + cmu) / \
                (1 + 2 * max(0, np.sqrt((lam - 1) / (n + 1)) - 1) + 2 * cmu)
        damps = min(1.0, damps)

        calls_made = 1
        remaining = self.budget - 1

        # Precompute Cholesky factors
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            L = np.eye(n)

        while remaining > 0:
            # Deadline check
            if time.time() > deadline:
                break

            # Generate population
            ys = []
            fs = []
            for i in range(lam):
                if time.time() > deadline:
                    break
                z = rng.standard_normal(n)
                y = x + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                ys.append(y)
                fs.append(fy)
                calls_made += 1
                remaining -= 1

            if len(ys) == 0:
                break

            # Sort by fitness
            order = np.argsort(fs)
            ys = np.array(ys)
            fs = np.array(fs)
            ys = ys[order]
            fs = fs[order]

            # Update best
            if fs[0] < best_fx:
                best_fx = fs[0]
                best_x = ys[0].copy()

            # Update mean
            w = np.log(np.arange(1, lam // 2 + 1)) - np.log(lam // 2)
            w = w / np.sum(w)
            mu_eff = 1.0 / np.sum(w ** 2)

            x_old = x.copy()
            x = np.sum(w[:, None] * ys[:int(np.sum(w > 0))], axis=0)

            # Update covariance
            diff = (x - x_old) / sigma
            C = (1 - cmu * damps) * C + cmu * damps * np.outer(diff, diff)
            # Add population covariance
            for i in range(int(np.sum(w > 0))):
                d = (ys[i] - x_old) / sigma
                C += cmu * damps * w[i] * np.outer(d, d)

            # Update path
            pc = (1 - cs) * pc + cs * damps * diff
            hsig = (np.sqrt((1 - (1 - cs) ** (calls_made / (n + 1))) /
                           (1 - (1 - cs) ** (calls_made / (n + 1)))) /
                    np.sqrt(np.trace(C))) if calls_made > 0 else 1.0
            hsig = min(1.0, hsig)

            # Update sigma
            sigma *= np.exp(cs / (n + 1) * (np.linalg.norm(pc) /
                                            np.sqrt(np.trace(C)) - 1))
            sigma = np.clip(sigma, 0.001, 10.0)

            # Ensure C is PSD
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Regularize
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C = (eigvecs * eigvals) @ eigvecs.T
                L = np.linalg.cholesky(C)

            # Adaptive step size: if no improvement in last lam calls, reduce sigma
            if len(fs) > 0 and fs[0] >= best_fx:
                sigma *= 0.95

        return best_x, best_fx
