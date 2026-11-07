"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES based black-box continuous optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        n = self.dim
        budget = self.budget

        # Population size
        lam = max(4, int(4 + 3 * np.log(n)))
        mu = lam // 2

        # Initial point: center of the box
        x = np.zeros(n)
        sigma = 2.0

        # Covariance matrix: identity
        C = np.eye(n)

        # Eigen decomposition cache
        evals, evec = np.linalg.eigh(C)

        # CMA-ES parameters
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        cs = (mu + 2) / (n + mu + 5)
        ds = 1 + 2 * max(0, np.sqrt((mu + 1) / (n + 1) - 1)) + cs
        cc = (4 + n/mu) / (n + 4 + 2*mu/n)
        c1 = (2 + 2*min(mu/n - 1, 1/(n+1))) / ((n+1.3)**2 + mu)
        cmu = min(1 - c1, 2 * (mu - 2 + 1/mu) / ((n+2)**2 + mu))

        ps = np.zeros(n)
        pc = np.zeros(n)

        # Initial population
        def sample_population():
            """Sample lambda individuals from N(x, sigma^2 C)."""
            # Update eigen decomposition if needed
            nonlocal C, evals, evec
            # Reuse eigen decomposition
            D = np.sqrt(np.maximum(evals, 1e-20))
            B = evec
            # Standard normal samples
            N = rng.standard_normal((lam, n))
            # Transform: B * D * N^T
            Y = (B @ (D * N.T)).T
            return x + sigma * Y

        # Track best
        best_x = x.copy()
        best_f = func(best_x)
        calls = 1

        # Main loop
        while calls < budget:
            # Sample population
            pop = sample_population()
            # Clip to box
            pop = np.clip(pop, -5, 5)

            # Evaluate
            fitnesses = np.zeros(lam)
            for i in range(lam):
                fitnesses[i] = func(pop[i])
                calls += 1
                if fitnesses[i] < best_f:
                    best_f = fitnesses[i]
                    best_x = pop[i].copy()

            # Sort by fitness
            order = np.argsort(fitnesses)
            pop_sorted = pop[order]
            fit_sorted = fitnesses[order]

            # Weights (log-weights)
            weights = np.zeros(mu)
            for i in range(mu):
                weights[i] = np.log(mu + 0.5) - np.log(i + 1)
            weights /= weights.sum()
            mueff = 1 / np.sum(weights**2)

            # Mean update
            m_old = x.copy()
            x_new = np.sum(pop_sorted[:mu] * weights[:, None], axis=0)
            x = x_new

            # Step-size control
            z = (x - m_old) / (sigma * chi_n)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * z
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2)) < (1.4 + 2/(n+1)) * chi_n
            sigma *= np.exp(cs/ds * (np.linalg.norm(ps)/chi_n - 1))

            # Covariance matrix update
            y = (x - m_old) / sigma
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + cc*(1-cc)*C) + cmu * np.sum(
                [np.outer(pop_sorted[i] - m_old, pop_sorted[i] - m_old) / (sigma**2) * weights[i] for i in range(mu)], axis=0
            )

            # Symmetrize
            C = 0.5 * (C + C.T)

            # Eigen decomposition
            evals, evec = np.linalg.eigh(C)
            evals = np.maximum(evals, 1e-20)

            # Clip x to box
            x = np.clip(x, -5, 5)

            # If budget exhausted, break
            if calls >= budget:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
