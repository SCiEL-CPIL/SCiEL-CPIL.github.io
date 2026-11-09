# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer for BBOB f12 (Bent cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270  # safety margin before 300s kill

        # Initial population size
        lam = 4 + int(3 * np.log(dim))  # ~10 for dim=5
        mu = lam // 2

        # Initial mean and sigma
        rng = np.random.default_rng(42)
        x_mean = np.zeros(dim)
        sigma = 2.0

        # CMA-ES parameters
        alpha_mu = mu / (2 * lam + 1)
        c_sigma = (mu + 2) / (dim + 4 + mu)
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * (alpha_mu - 2 + 1 / alpha_mu) / ((dim + 2) ** 2 + mu))
        d_sigma = 1 + 2 * max(0, np.sqrt((alpha_mu - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # State variables
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        step_size = 1.0

        # Initial evaluation
        x0 = rng.uniform(-5, 5, dim)
        f0 = func(x0)
        x_mean = np.clip(x0, -5, 5)

        # Generate initial population
        def generate_population(x_mean, sigma, B, D):
            z = rng.standard_normal((lam, dim))
            # Transform to CMA-ES step distribution
            steps = (z @ (B * D)).T * sigma  # shape (lam, dim)
            pops = x_mean[None, :] + steps
            # Clip to box
            pops = np.clip(pops, -5, 5)
            return pops

        # Evaluate initial population
        pops = generate_population(x_mean, sigma, B, D)
        fvals = np.array([func(p) for p in pops])
        best_idx = np.argmin(fvals)
        x_best = pops[best_idx].copy()
        f_best = fvals[best_idx]
        x_mean = x_best.copy()

        call_count = lam + 1  # initial point + population

        for gen in range(1, budget):
            if time.time() > deadline:
                break
            if call_count >= budget:
                break

            # Update evolution paths
            w = np.zeros(lam)
            w[:mu] = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()

            # Sort by fitness
            order = np.argsort(fvals)
            x_sorted = pops[order]
            f_sorted = fvals[order]

            # Weighted recombination
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += w[i] * x_sorted[i]

            # Update mean
            x_mean = np.clip(x_new, -5, 5)

            # Update evolution path for sigma adaptation
            y = (x_new - pops[0]) / sigma if sigma > 0 else np.zeros(dim)
            # Actually, we need the step from the old mean
            y = (x_new - x_best) / sigma if sigma > 0 else np.zeros(dim)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * alpha_mu * mu) * y

            # Update evolution path for covariance matrix
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu) * y

            # Check for sigma adaptation
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) > (1.4 + 2 / (dim + 1)) * chi_n:
                sigma *= np.exp((1.4 + 2 / (dim + 1)) * chi_n - np.linalg.norm(ps) / d_sigma)

            # Update covariance matrix
            # Compute rank-1 and rank-mu updates
            delta_sig = (np.linalg.norm(ps) ** 2 / chi_n ** 2 - 1) * d_sigma * c_sigma
            h_sigma = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (gen + 1)) <
                       (1.4 + 2 / (dim + 1)) * chi_n)
            h_sigma = 1.0 if h_sigma else 0.0

            # Rank-1 update
            outer = np.outer(pc, pc)
            h = (1 - h_sigma) * c_c * (2 - c_c)
            C_update = (1 + c_1 * (2 - c_1) - c_1) * np.eye(dim) + c_1 * (outer - np.eye(dim)) + c_1 * h * np.eye(dim)

            # Rank-mu update
            # This is expensive; for small dim it's fine
            diff = (x_sorted[:mu] - x_best) / sigma if sigma > 0 else np.zeros((mu, dim))
            C_mu = np.zeros((dim, dim))
            for i in range(mu):
                C_mu += w[i] * np.outer(diff[i], diff[i])
            C_update += c_mu * C_mu

            # Ensure positive definite and update B, D
            # Use eigendecomposition
            try:
                eigvals, eigvecs = np.linalg.eigh(C_update)
                eigvals = np.maximum(eigvals, 1e-12)
                B = eigvecs
                D = np.sqrt(eigvals)
            except:
                pass

            # Generate next population
            pops = generate_population(x_mean, sigma, B, D)
            fvals = np.array([func(p) for p in pops])
            call_count += lam

            best_idx = np.argmin(fvals)
            if fvals[best_idx] < f_best:
                f_best = fvals[best_idx]
                x_best = pops[best_idx].copy()

        return x_best, f_best
# EVOLVE-BLOCK-END
