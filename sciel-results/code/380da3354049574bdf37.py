"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for the bent cigar function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Population size
        lambda_pop = 4 + int(3 * np.log(dim))  # for dim=5: 4 + int(3*1.609) = 4+4 = 8
        # But let's use a slightly larger pop for better exploration
        lambda_pop = max(8, int(4 + 3 * np.log(dim)))

        # CMA-ES parameters
        mu = lambda_pop // 2  # number of recombination weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights ** 2)

        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        # Initial state
        rng = np.random.default_rng(42)
        # Start near origin since minimizer is in [-4,4]^5
        mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        Csqrt = np.eye(dim)

        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Initial population
        def sample_population():
            # Generate lambda samples
            samples = []
            for i in range(lambda_pop):
                z = rng.standard_normal(dim)
                x = mean + sigma * (Csqrt @ z)
                x = np.clip(x, -5, 5)
                samples.append(x)
            return samples

        def update_covariance_matrix(C, Csqrt, z, ps, pc, sigma, cc, cs, c1, cmu, weights, mueff, dim):
            """Update C and Csqrt using rank-one and rank-mu updates."""
            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (Csqrt @ z)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** 2)) < (1.4 + 2 / (dim + 1)) * np.sqrt(mueff + 1)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) / sigma * z

            # Rank-one update
            C = (1 + c1 * hsig * (cc * (2 - cc) - 1)) * C + c1 * np.outer(pc, pc)

            # Rank-mu update
            # z is the best sample's standard normal vector
            # For simplicity, we approximate using the mean shift
            # Actually, proper rank-mu uses the weighted sum of top-mu samples' z-vectors
            # Let's do a simplified version: use the best z
            if cmu > 0:
                C += cmu * np.outer(z, z)

            # Ensure symmetry
            C = (C + C.T) / 2

            # Compute new Csqrt via eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            Csqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

            return C, Csqrt, ps, pc

        best_x = None
        best_f = np.inf

        # Initial evaluation
        samples = sample_population()
        fvals = np.array([func(s) for s in samples])

        # Track best
        idx_best = np.argmin(fvals)
        best_x = samples[idx_best]
        best_f = fvals[idx_best]

        calls_made = lambda_pop
        calls_remaining = budget - calls_made

        while calls_remaining > 0:
            # Select top mu
            order = np.argsort(fvals)
            selected_idx = order[:mu]
            selected_samples = [samples[i] for i in selected_idx]
            selected_fvals = fvals[selected_idx]

            # Compute weighted mean
            new_mean = np.sum(np.array(selected_samples) * weights[:, np.newaxis], axis=0)

            # Standard normal vectors for selected samples
            # We need to invert the transformation: z = Csqrt^{-1} @ (x - mean) / sigma
            # Use pseudo-inverse for stability
            try:
                Csqrt_inv = np.linalg.pinv(Csqrt)
            except np.linalg.LinAlgError:
                Csqrt_inv = np.eye(dim)

            z_vectors = []
            for s in selected_samples:
                z = Csqrt_inv @ (s - mean) / sigma
                z_vectors.append(z)
            z_vectors = np.array(z_vectors)

            # Update paths and covariance
            # Use the best z for the path update (approximation)
            z_best = z_vectors[0]
            C, Csqrt, ps, pc = update_covariance_matrix(
                C, Csqrt, z_best, ps, pc, sigma, cc, cs, c1, cmu, weights, mueff, dim
            )

            # Adapt sigma
            sigma *= np.exp(cs / (dim + 3) * (np.linalg.norm(ps) / np.sqrt(mueff) - 1))
            sigma = np.clip(sigma, 1e-10, 100)

            # Update mean
            mean = new_mean
            mean = np.clip(mean, -5, 5)

            # Generate new population
            samples = sample_population()
            fvals = np.array([func(s) for s in samples])
            calls_made += lambda_pop
            calls_remaining = budget - calls_made

            # Track best
            idx_best = np.argmin(fvals)
            if fvals[idx_best] < best_f:
                best_x = samples[idx_best]
                best_f = fvals[idx_best]

        return best_x, best_f
# EVOLVE-BLOCK-END
