"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with anisotropic initialization for Bent Cigar (f12)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Anisotropic initial sigma: smaller for cigar axis, larger for wide dims
        sigma = np.array([0.3, 1.5, 1.5, 1.5, 1.5])

        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        lam = 20
        mu = lam // 2

        cc = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
        cs = (lam / 2 + 1) / (dim + lam / 2 + 3)
        c1 = 2 / ((dim + 1.3) ** 2 + lam)
        cmu = min(1 - c1, 2 * ((mu + 0.5) / (mu + 1) - 2) / ((dim + 2) ** 2 + lam))

        ps = np.zeros(dim)
        pc = np.zeros(dim)
        chi_lam = np.sqrt(lam) * (1 + 1 / (2 * dim) - 1 / (4 * (lam - 1)))

        weights = np.log(lam + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights ** 2)

        # C = diag(sigma^2), so C^{1/2} = diag(sigma)
        C_half = np.diag(sigma)
        C = C_half @ C_half

        for gen in range(budget // lam + 1):
            if time.time() > deadline:
                break

            # Generate population
            pop = []
            pop_f = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x + C_half @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                pop.append(y)
                pop_f.append(fy)

            # Sort
            order = np.argsort(pop_f)
            pop = [pop[i] for i in order]
            pop_f = [pop_f[i] for i in order]

            if pop_f[0] < best_fx:
                best_fx = pop_f[0]
                best_x = pop[0].copy()

            # Weighted mean
            mean_new = np.zeros(dim)
            for i in range(mu):
                mean_new += weights[i] * pop[i]

            # Update evolution paths
            diff = mean_new - x
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_half.T @ (C_half @ (diff / sigma)))
            # Simpler: ps = (1-cs)*ps + sqrt(cs*(2-cs)*mueff) * (diff / sigma) element-wise
            # Actually for diagonal C, the transformation is just scaling
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (diff / sigma)

            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))) < (1.4 + 2 / (dim + 1)) * chi_lam

            if hsig:
                pc = (1 - cc) * pc + cc * (diff / sigma)

            # Update C (diagonal approximation for efficiency, but let's do full)
            # For dim=5, full update is fine
            C = (1 + c1 * (hsig - 1) + cmu) * C
            if hsig:
                C += c1 * (np.outer(pc, pc) - C)
            for i in range(mu):
                zi = (pop[i] - x) / sigma
                C += cmu * weights[i] * np.outer(zi, zi)

            # Update C_half via Cholesky
            try:
                L = np.linalg.cholesky(C)
                C_half = L
            except np.linalg.LinAlgError:
                # Fallback: reinitialize
                C_half = np.diag(sigma)
                C = C_half @ C_half

            # Adapt sigma
            sigma_norm = np.sqrt(np.trace(C) / dim)
            sigma_old = np.sqrt(np.trace(C_half @ C_half) / dim)
            # Standard CMA-ES: sigma *= exp(cs/chi_lam * (||ps||/chi_lam - 1))
            sigma_scale = np.exp(cs / chi_lam * (np.linalg.norm(ps) / chi_lam - 1))
            # Apply to C_half
            C_half = C_half * sigma_scale
            C = C_half @ C_half

            x = mean_new

        return best_x, best_fx
# EVOLVE-BLOCK-END
