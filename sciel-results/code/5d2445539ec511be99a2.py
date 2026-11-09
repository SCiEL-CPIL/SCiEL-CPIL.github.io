"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase CMA-ES: coarse basin search then intensive refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        # Track best value and count calls
        best_x = None
        best_f = np.inf
        n_calls = 0

        def evaluate(x):
            nonlocal best_x, best_f, n_calls
            fx = float(func(x))
            n_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: CMA-ES with large sigma for basin exploration
        x = np.zeros(dim)
        f = evaluate(x)
        sigma = 2.5
        pop = max(4, int(np.ceil(4 + 3 * np.log(dim))))
        # Covariance matrix
        C = np.eye(dim)
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        # Step sizes
        cs = (pop + 2) / (dim + pop + 5)
        cmu = min(0.5, (pop - 2 + 0.5) / (dim + pop + 2))
        c1 = (2 + 2 / (dim + 1)) / ((dim + 2) ** 2 + pop)
        cc = (4 + dim / pop) / (dim + 4 + 2 * pop / (dim + 1))
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        phase1_budget = 150
        phase1_calls = 1

        for gen in range(1, 100):
            if n_calls >= phase1_budget:
                break
            if time.time() > deadline - 50:
                break
            # Generate population
            W = np.zeros(pop)
            for i in range(pop):
                if n_calls >= phase1_budget:
                    break
                if time.time() > deadline - 50:
                    break
                # Sample direction
                z = rng.standard_normal(dim)
                # Apply C
                d = C @ z
                # Normalize by sigma
                y = x + sigma * d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-10 else x
                y = np.clip(y, -5, 5)
                fy = evaluate(y)
                W[i] = fy

            if n_calls >= phase1_budget:
                break

            # Sort by fitness
            order = np.argsort(W)
            # Update mean (weighted)
            w = np.exp(-np.arange(1, pop + 1) / (0.35 * dim + 2.5))
            w = w / w.sum()
            x_new = np.zeros(dim)
            for i in range(pop):
                x_new += w[i] * W[order[i]]
            x = x_new

            # Update evolution paths
            # ps
            d_best = C @ (W[order[0]] - x) / sigma if np.linalg.norm(W[order[0]] - x) > 1e-10 else np.zeros(dim)
            ps = (1 - cs) * ps + np.sqrt(2 * cs * (2 - cs)) * d_best
            # pc
            hsig = (1 + 3 / dim) / (1 - cs)
            if np.linalg.norm(ps) / chi < (1.4 + 2 / (dim + 1)) * hsig:
                pc = (1 - cc) * pc + cc * d_best

            # Update covariance
            delta = 1 + 2 * max(0, hsig - 1) * cs * (2 - cs)
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + delta * C)
            for i in range(pop):
                d_i = C @ (W[order[i]] - x) / sigma if np.linalg.norm(W[order[i]] - x) > 1e-10 else np.zeros(dim)
                C += cmu * w[i] * np.outer(d_i, d_i)

            # Adapt sigma
            sigma *= np.exp(cs / chi * (np.linalg.norm(ps) / chi - 1))
            sigma = max(sigma, 1e-6)
            sigma = min(sigma, 5.0)

            # Symmetrize C
            C = (C + C.T) / 2

        # Phase 2: Refined CMA-ES with smaller sigma
        # Reset for refinement
        x = best_x.copy()
        f = best_f
        sigma = max(0.3, 0.1 * np.linalg.norm(x))
        sigma = min(sigma, 1.0)
        C = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        cs = (pop + 2) / (dim + pop + 5)
        cmu = min(0.5, (pop - 2 + 0.5) / (dim + pop + 2))
        c1 = (2 + 2 / (dim + 1)) / ((dim + 2) ** 2 + pop)
        cc = (4 + dim / pop) / (dim + 4 + 2 * pop / (dim + 1))
        w = np.exp(-np.arange(1, pop + 1) / (0.35 * dim + 2.5))
        w = w / w.sum()

        for gen in range(1, 200):
            if n_calls >= budget:
                break
            if time.time() > deadline:
                break
            W = np.zeros(pop)
            for i in range(pop):
                if n_calls >= budget:
                    break
                if time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                d = C @ z
                norm_d = np.linalg.norm(d)
                y = x + sigma * d / norm_d if norm_d > 1e-10 else x
                y = np.clip(y, -5, 5)
                W[i] = evaluate(y)

            if n_calls >= budget:
                break

            order = np.argsort(W)
            x_new = np.zeros(dim)
            for i in range(pop):
                x_new += w[i] * W[order[i]]
            x = x_new

            d_best = C @ (W[order[0]] - x) / sigma if np.linalg.norm(W[order[0]] - x) > 1e-10 else np.zeros(dim)
            ps = (1 - cs) * ps + np.sqrt(2 * cs * (2 - cs)) * d_best
            hsig = (1 + 3 / dim) / (1 - cs)
            if np.linalg.norm(ps) / chi < (1.4 + 2 / (dim + 1)) * hsig:
                pc = (1 - cc) * pc + cc * d_best

            delta = 1 + 2 * max(0, hsig - 1) * cs * (2 - cs)
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + delta * C)
            for i in range(pop):
                d_i = C @ (W[order[i]] - x) / sigma if np.linalg.norm(W[order[i]] - x) > 1e-10 else np.zeros(dim)
                C += cmu * w[i] * np.outer(d_i, d_i)

            sigma *= np.exp(cs / chi * (np.linalg.norm(ps) / chi - 1))
            sigma = max(sigma, 1e-8)

            C = (C + C.T) / 2

        return best_x, best_f
# EVOLVE-BLOCK-END
