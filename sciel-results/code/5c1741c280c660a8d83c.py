import numpy as np
import time

class Optimizer:
    """
    Optimizer using a deterministic grid-based global search followed by 
    CMA-ES local refinement.
    
    Strategy:
    1. Evaluate a sparse 3^5 grid (243 points) to identify the best basin.
    2. Run CMA-ES from the best grid point to refine the solution.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, func, x0, sigma0, budget, rng, deadline):
        """Run CMA-ES with the given initial state and budget."""
        dim = self.dim
        lam = 4 + int(3 * np.log(dim))  # ~12 for dim=5
        mu = lam // 2
        weights = np.log(np.arange(mu, 0, -1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        damps = 1 + 2 * max(0, (mu_eff - 2) / (dim + 2) - 1) + cs

        x = x0.copy()
        sigma = sigma0
        C = np.eye(dim)
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-20)
        B = eigvecs
        D = np.sqrt(eigvals)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        best_x = x.copy()
        best_fx = func(x)
        calls = 1
        no_improve = 0

        for gen in range(budget // lam):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            pop = []
            fvals = []
            for i in range(lam):
                if calls >= budget:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (B * D) @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                pop.append(y)
                fvals.append(fy)
                calls += 1

            if len(pop) == 0:
                break

            idx = np.argsort(fvals)
            pop = [pop[i] for i in idx]
            fvals = np.array(fvals)[idx]

            if fvals[0] < best_fx:
                best_fx = fvals[0]
                best_x = pop[0].copy()
                no_improve = 0
            else:
                no_improve += 1

            if no_improve > 20 and sigma < 1e-5:
                break

            mu_use = min(mu, len(pop))
            m = np.sum(weights[:mu_use, None] * np.array(pop[:mu_use]), axis=0)

            # Update step size
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (B.T @ (m - x))
            hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 100.0)

            # Update evolution path
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) / sigma * (B.T @ (m - x))

            # Update covariance
            delta_hs = (1 - hsig) * cc * (2 - cc)
            outer = np.outer(pc, pc)
            C = (1 + c1 * delta_hs - c1 - cmu) * C + c1 * outer
            for i in range(mu_use):
                zi = (np.array(pop[i]) - x) / sigma
                C += cmu * weights[i] * np.outer(zi, zi)

            # Ensure symmetry and positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            B = eigvecs
            D = np.sqrt(eigvals)

            x = m

        return best_x, best_fx, calls

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)

        # Phase 1: Deterministic Grid Search
        # Create a 3^5 grid over [-5, 5]
        # Points: -5, 0, 5
        grid_coords = [-5.0, 0.0, 5.0]
        grid_points = []
        for x1 in grid_coords:
            for x2 in grid_coords:
                for x3 in grid_coords:
                    for x4 in grid_coords:
                        for x5 in grid_coords:
                            grid_points.append(np.array([x1, x2, x3, x4, x5]))
        
        # Evaluate grid points
        best_x = grid_points[0].copy()
        best_fx = np.inf
        calls = 0

        for gp in grid_points:
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            fx = func(gp)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = gp.copy()

        # Phase 2: CMA-ES Refinement
        # Use the remaining budget for CMA-ES starting from the best grid point
        remaining_budget = budget - calls
        if remaining_budget > 10 and time.time() < deadline:
            # Initialize sigma based on the grid spacing (which is 5.0)
            # A smaller sigma is better for refinement, but we want to ensure we stay in the basin.
            # The grid point is likely close to the minimum.
            sigma0 = 1.0 
            cma_x, cma_fx, cma_calls = self._cmaes(func, best_x, sigma0, remaining_budget, rng, deadline)
            calls += cma_calls
            if cma_fx < best_fx:
                best_fx = cma_fx
                best_x = cma_x.copy()

        # Safety: Ensure we don't exceed budget or time
        if time.time() > deadline - 5:
             pass # Just return current best

        return best_x, best_fx
