# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Restart CMA-ES (iRNG) optimizer for BBOB f7 (Step ellipsoidal) on [-5,5]^d.
    Uses multiple short, independent CMA-ES runs from diverse starting points.
    This balances global exploration (for early AOCC progress) with local refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        self._call_count = 0
        self._best_f = np.inf
        self._best_x = None

        def wrapped(x):
            self._call_count += 1
            fx = float(func(x))
            if fx < self._best_f:
                self._best_f = fx
                self._best_x = np.array(x, dtype=float)
            return fx

        rng = np.random.default_rng(42)

        # Helper: Run a single CMA-ES instance for a specific budget
        def run_cmaes(x0, sigma0, max_calls, rng_local):
            # Parameters for dim=5
            lam = 6
            mu = lam // 2
            
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            cs = (mu_eff + 2) / (dim + mu_eff + 5)
            c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1) - 1) + cs)
            chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            mean = np.array(x0, dtype=float)
            sigma = sigma0
            C = np.eye(dim)
            
            # Eigendecomposition helpers
            def _update_eigen(C):
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-20, None)
                D = np.sqrt(eigvals)
                B = eigvecs
                return B, D

            B, D = _update_eigen(C)
            C_sqrt = B @ np.diag(D)
            C_sqrt_inv = B @ np.diag(1.0 / D)

            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            local_best_f = np.inf
            local_best_x = mean.copy()
            no_improve_count = 0
            stagnation_limit = 15 # Generations without improvement before suggesting restart

            while self._call_count < max_calls:
                if time.time() > deadline:
                    break

                # Generate population
                population = []
                for i in range(lam):
                    if self._call_count >= max_calls:
                        break
                    z = rng_local.standard_normal(dim)
                    y = C_sqrt @ z
                    x_new = mean + sigma * y
                    x_new = np.clip(x_new, -5.0, 5.0)
                    population.append(x_new)

                if not population:
                    break

                # Evaluate
                fitness = []
                for x_new in population:
                    if self._call_count >= max_calls:
                        break
                    if time.time() > deadline:
                        break
                    fx = wrapped(x_new)
                    fitness.append((fx, x_new))
                    
                    if fx < local_best_f:
                        local_best_f = fx
                        local_best_x = x_new.copy()

                if len(fitness) < mu:
                    break

                fitness.sort(key=lambda t: t[0])
                selected = [f[1] for f in fitness[:mu]]

                # Check for improvement
                if fitness[0][0] < local_best_f - 1e-10:
                    local_best_f = fitness[0][0]
                    local_best_x = selected[0].copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # Stagnation check
                if no_improve_count > stagnation_limit:
                    break

                # Update mean
                mean_old = mean.copy()
                mean = np.zeros(dim)
                for i in range(mu):
                    mean += weights[i] * selected[i]

                # Update paths
                y_mean = C_sqrt_inv @ (mean - mean_old)
                p_c = (1 - cs) * p_c + cs * np.sqrt(mu_eff) * y_mean
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * y_mean

                # Update sigma
                h_sigma = (
                    np.linalg.norm(p_sigma) / (1 - (1 - cs) ** (no_improve_count * cs + 1))
                    < (1.4 + 2 / (dim + 1)) * chiN
                )
                if h_sigma:
                    sigma *= np.exp(cs / damps * (np.linalg.norm(p_sigma) / chiN - 1))

                # Update C
                delta_h = (1 - h_sigma) * cs * (2 - cs)
                C = (1 - c1 - cmu) * C + c1 * (np.outer(p_c, p_c) + delta_h * C)
                for i in range(mu):
                    yi = C_sqrt_inv @ (selected[i] - mean_old)
                    C += cmu * weights[i] * np.outer(yi, yi)

                C = 0.5 * (C + C.T)
                B, D = _update_eigen(C)
                C_sqrt = B @ np.diag(D)
                C_sqrt_inv = B @ np.diag(1.0 / D)

                # Convergence check
                if sigma < 1e-12:
                    break

            return local_best_x, local_best_f

        # Strategy: Multiple restarts with increasing budget
        # Allocate budget into chunks. Early chunks are smaller to allow restarts.
        # For 500 budget, let's try 4-5 restarts.
        # Chunk sizes: e.g., 50, 70, 100, 150, rest
        # Or simply: while budget remains, take a chunk of size ~50-100
        
        remaining_budget = budget
        restarts = 0
        max_restarts = 5
        
        # Initial guess for starting points
        # 1. Center
        # 2. Random
        # 3. Best found so far (perturbed)
        
        start_points = [
            np.zeros(dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-4, 4, dim)
        ]

        for i in range(max_restarts):
            if remaining_budget <= 10:
                break
            if time.time() > deadline:
                break
            
            # Determine chunk size
            # Early restarts get smaller budgets to explore
            # Later restarts get larger budgets to exploit
            if i == 0:
                chunk_size = 40
            elif i == 1:
                chunk_size = 60
            elif i == 2:
                chunk_size = 80
            elif i == 3:
                chunk_size = 100
            else:
                chunk_size = remaining_budget - 10 # Keep 10 for final polish

            if chunk_size < 10:
                break
            
            # Select starting point
            if i == 0:
                x0 = start_points[0]
            elif i == 1:
                x0 = start_points[1]
            elif i == 2:
                x0 = start_points[2]
            else:
                # For later restarts, start near the current best if it's good,
                # or a new random point if we suspect we are stuck
                if self._best_x is not None and rng.random() < 0.5:
                    # Perturb best
                    x0 = self._best_x + rng.normal(0, 0.5, dim)
                    x0 = np.clip(x0, -5, 5)
                else:
                    # New random
                    x0 = rng.uniform(-4, 4, dim)
            
            sigma0 = 1.5 # Moderate initial step
            
            local_rng = np.random.default_rng(rng.integers(0, 2**31))
            
            # Run CMA-ES for this chunk
            # We pass the global call count limit as max_calls
            # But we want to stop after chunk_size calls *if* we don't find a good improvement
            # Actually, run_cmaes handles the global count. We just give it the remaining budget.
            # But we want to FORCE a restart if it doesn't improve much.
            # The stagnation check inside run_cmaes handles this.
            
            # To ensure we actually restart, we cap the calls for this specific run
            # by passing a local limit? No, run_cmaes uses self._call_count.
            # We can modify run_cmaes to accept a local_stop_call_count.
            
            # Let's just run it. The stagnation limit of 15 gens * 6 pop = 90 calls
            # might exceed our chunk size. 
            # If chunk_size is 40, and stagnation limit is 15 gens, it will run until budget is hit or stagnation.
            # This is fine.
            
            run_cmaes(x0, sigma0, self._call_count + chunk_size, local_rng)
            
            remaining_budget -= chunk_size
            restarts += 1

        # Final polish: Use remaining budget for a very local search around the best found
        if self._best_x is not None and remaining_budget > 5:
            # Simple local search
            sigma_local = 0.1
            for _ in range(remaining_budget - 5):
                if self._call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                x_try = self._best_x + rng.normal(0, sigma_local, dim)
                x_try = np.clip(x_try, -5, 5)
                fx = wrapped(x_try)
                # Adaptive local step
                if fx < self._best_f:
                    sigma_local *= 0.9
                else:
                    sigma_local *= 1.1
                    sigma_local = min(sigma_local, 0.5)

        if self._best_x is None:
            self._best_x = np.zeros(dim)
            self._best_f = func(self._best_x)

        return self._best_x, self._best_f


# EVOLVE-BLOCK-END
