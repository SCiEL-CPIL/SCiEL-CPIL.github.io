"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with random restarts, adapted for hidden rotation/shift."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _cma_es(self, func, x0, sigma0, budget, dim, rng, deadline, calls_ref):
        """Run CMA-ES from x0. Returns (best_x, best_f, calls_used)."""
        n = dim
        # Population size
        lam = max(4, int(4 + 3 * np.log(n)))
        # Selection
        mu = lam // 2
        # Weights
        weights = np.log(mu + 1) - np.arange(1, mu + 1)
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        # Step size control
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        # Covariance adaptation
        c1 = (2 / ((n + 1.3) ** 2 + mueff)) * (2 - cc)
        cmu = min(1 - c1, 2 * (weights[0] - 1 + weights[1]) / ((n + 2) ** 2 + mueff))
        
        # Initial state
        x = x0.copy()
        sigma = sigma0
        C = np.eye(n)
        B, D = np.linalg.eigh(C) # Initial eigendecomp
        pc = np.zeros(n)
        ps = np.zeros(n)
        best_x = x.copy()
        best_f = np.inf
        calls = 0
        gen = 0
        max_gen = budget // lam + 5

        chi_n = np.sqrt(n) * (1 - 1. / (4 * n) + 1. / (21 * n ** 2))

        for _ in range(max_gen):
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            
            # Generate population
            offspring = []
            # Pre-compute random normal matrix for speed
            Z = rng.standard_normal((lam, n))
            # Transform to x + sigma * B @ (D * Z.T).T
            # y = x + sigma * (B @ (D * Z.T)).T
            # Let's do it row by row or vectorized
            # y = x + sigma * (Z * D[None, :]) @ B.T
            # Wait, B is orthogonal, so B.T is inverse.
            # Standard CMA: y = x + sigma * B @ (D * z)
            # Vectorized: Y = X + sigma * (Z * D[None, :]) @ B.T
            # Since Z is (lam, n), D is (n,), B is (n, n)
            # Z * D[None, :] scales rows of Z by D? No, scales columns?
            # D is diagonal, so D @ Z.T = (Z * D[None, :]).T ?
            # Let's stick to loop for clarity and correctness in 5D, it's fast enough.
            
            for i in range(lam):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break
                zi = Z[i]
                y = x + sigma * (B @ (D * zi))
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                offspring.append((fy, y))
            
            if len(offspring) < mu:
                break
            
            offspring.sort(key=lambda t: t[0])
            
            # Update best
            if offspring[0][0] < best_f:
                best_f = offspring[0][0]
                best_x = offspring[0][1].copy()
            
            # Mean update
            old_x = x.copy()
            x_new = np.zeros(n)
            for i in range(mu):
                x_new += weights[i] * offspring[i][1]
            x = x_new
            
            # Step size control
            y_mean = (x - old_x) / sigma
            y_mean_std = np.linalg.solve(D, np.linalg.solve(B.T, y_mean))
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y_mean_std
            
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1.4 + 1 / n) * chi_n:
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            
            # Covariance matrix adaptation
            hs = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (gen + 1))) < (1.4 + 1 / n) * chi_n
            pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * mueff) * y_mean_std
            
            # Rank-1 update
            delta_hs = (1.0 if hs else 0.0) - (1 - cc)
            outer_pc = np.outer(pc, pc)
            H = np.eye(n) + delta_hs * outer_pc
            
            # Rank-mu update
            Y = np.array([np.linalg.solve(D, np.linalg.solve(B.T, (offspring[i][1] - old_x) / sigma)) for i in range(mu)])
            H_mu = np.sum([weights[i] * np.outer(Y[i], Y[i]) for i in range(mu)], axis=0)
            
            C = (1 - c1 - cmu) * C + c1 * H + cmu * H_mu
            
            # Re-orthogonalize C and update B, D
            # Symmetrize
            C = (C + C.T) / 2
            try:
                B, D = np.linalg.eigh(C)
                # Ensure D is positive
                D = np.clip(D, 1e-10, None)
                # If eigenvalues are too small, reset
                if np.min(D) < 1e-8:
                    C = np.eye(n)
                    B, D = np.linalg.eigh(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                B, D = np.linalg.eigh(C)
            
            gen += 1
            
            # Convergence check
            if sigma < 1e-10 or np.max(np.abs(x - best_x)) < 1e-10:
                break
        
        return best_x, best_f, calls

    def __call__(self, func):
        rng = np.random.default_rng(0)
        deadline = time.time() + 285.0
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0

        # Phase 1: Initial Exploration
        # Use a small number of random points to establish a baseline
        n_explore = 10
        for _ in range(n_explore):
            if time.time() > deadline or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES with restarts
        remaining = budget - calls_made
        if remaining > 20:
            # Run CMA-ES from best point
            bx, bf, c = self._cma_es(func, best_x, 2.0, remaining, dim, rng, deadline, None)
            calls_made += c
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()
            
            # If budget remains, do a few random restarts
            while calls_made < budget and time.time() < deadline:
                x = rng.uniform(-5, 5, dim)
                fx = func(x)
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    # Try CMA-ES from this new best
                    remaining = budget - calls_made
                    if remaining > 20:
                        bx, bf, c = self._cma_es(func, best_x, 1.0, remaining, dim, rng, deadline, None)
                        calls_made += c
                        if bf < best_f:
                            best_f = bf
                            best_x = bx.copy()
                    if calls_made >= budget:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
