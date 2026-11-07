"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive step size for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initial point
        x0 = np.zeros(dim)
        fx0 = func(x0)
        best_x = x0.copy()
        best_fx = fx0
        calls = 1

        # CMA-ES setup
        lam = 4 + int(3 * np.log(dim))  # ~14 for dim=5
        sigma = 1.0
        m = x0.copy()
        C = np.eye(dim)
        D, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(D, 1e-20))

        cc = (4 + lam / dim) / ((dim + 4) * (2 + lam / dim))
        cmu = min(1 - cc, 2 * (dim - 2 + 1 / dim) / ((dim + 2) * (lam + 2)))
        ctho = 1.0
        damps = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (dim + 1)) - 1) + cmu
        damps_c = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (dim + 1)) - 1) + cc

        ps = np.zeros(dim)
        pc = np.zeros(dim)

        gen = 0
        while calls < budget:
            if time.time() > deadline:
                break

            # Sample offspring
            offspring = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                xi = m + sigma * (B @ (D * z))
                xi = np.clip(xi, -5, 5)
                offspring.append(xi)

            # Evaluate
            results = []
            for xi in offspring:
                if calls >= budget or time.time() > deadline:
                    break
                fi = func(xi)
                results.append((fi, xi))
                calls += 1
                if fi < best_fx:
                    best_fx = fi
                    best_x = xi.copy()

            if len(results) < lam:
                break

            results.sort(key=lambda t: t[0])
            mus = min(lam // 2, len(results))
            w = np.log(mus + 0.5) - np.log(np.arange(1, mus + 1))
            w /= w.sum()
            mueff = 1.0 / np.sum(w ** 2)

            # Weighted mean in x-space
            m_new = np.zeros(dim)
            for i in range(mus):
                m_new += w[i] * results[i][1]

            # Update evolution path ps
            # Transform weighted mean to z-space
            diff = m_new - m
            z_wm = np.diag(1.0 / D) @ (B.T @ diff) / sigma
            ps = (1 - cc) * ps + np.sqrt(cc * (2 - cc) * mueff) * z_wm

            # Update pc
            norm_ps = np.linalg.norm(ps)
            if norm_ps / ctho < np.sqrt(1 - (1 - cc) ** (2 * (gen + 1))):
                pc = (1 - cmu) * pc + np.sqrt(cmu * (2 - cmu) * mueff) * (
                    (1 - cc) * (B @ (D * ps)) / sigma
                )
            else:
                pc = (1 - cmu) * pc

            # Update mean
            m = m_new

            # Update covariance
            delta_var = (1 - (1 - cmu) ** 2) * np.outer(pc, pc)
            for i in range(mus):
                diff_i = results[i][1] - m
                z_i = np.diag(1.0 / D) @ (B.T @ diff_i) / sigma
                delta_var += w[i] * np.outer(z_i, z_i)
            C = (1 - cmu - 2 * w[0] * mueff) * C + cmu * (
                B @ (np.diag(D) @ delta_var @ np.diag(D)) @ B.T
            )

            # Update sigma
            sigma *= np.exp(1.0 / damps * (norm_ps / ctho - 1))

            # Re-decompose
            C = (C + C.T) / 2
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-20))

            gen += 1

            # Safety: if sigma too small, break
            if sigma < 1e-12:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
