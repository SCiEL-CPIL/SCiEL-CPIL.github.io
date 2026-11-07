"""Two-phase Sobol + CMA-ES optimizer for Rastrigin rotated (f15)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.stats.qmc

class Optimizer:
    """Two-phase Sobol + CMA-ES optimizer for Rastrigin rotated (f15)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def cmaes_run(x0, sigma0, max_evals, rng, lambda_pop=None):
            """Run CMA-ES instance."""
            nonlocal calls, best_x, best_f
            n = dim
            if lambda_pop is None:
                lambda_pop = max(4, int(4 + 3 * np.log(n)))
            lam = lambda_pop
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
            cs = (mueff + 2.0) / (n + mueff + 5.0)
            c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
            chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

            x = x0.copy()
            sigma = sigma0
            pc = np.zeros(n)
            ps = np.zeros(n)
            C = np.eye(n)
            
            # Efficient square root calculation
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            local_best_x = x.copy()
            local_best_f = np.inf
            evals_used = 0

            while evals_used < max_evals and calls < budget and time.time() < deadline:
                population = []
                for i in range(lam):
                    if calls >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(n)
                    x_new = x + sigma * (C_sqrt @ z)
                    x_new = np.clip(x_new, -5.0, 5.0)
                    f_new = evaluate(x_new)
                    evals_used += 1
                    population.append((f_new, x_new))

                if len(population) < lam:
                    break

                population.sort(key=lambda p: p[0])

                if population[0][0] < local_best_f:
                    local_best_f = population[0][0]
                    local_best_x = population[0][1].copy()

                parents = [p[1] for p in population[:mu]]

                x_old = x.copy()
                x = sum(weights[i] * parents[i] for i in range(mu))

                # Update step size
                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * (C_inv_sqrt @ ((x - x_old) / sigma))
                hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2 * (evals_used + 1) // lam))) / chi_n < 1.4 + 2.0 / (n + 1.0)
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1.0))
                sigma = np.clip(sigma, 1e-12, 10.0)

                # Update paths and covariance
                if hsig:
                    pc = (1.0 - cc) * pc + np.sqrt(cc * (2.0 - cc) * mueff) * (C_inv_sqrt @ ((x - x_old) / sigma))

                rank_mu = np.outer(parents, weights) @ parents.T
                rank_c = np.outer(pc, pc)
                C = (1.0 + c1 * (2.0 - c1) - cmu) * C + c1 * rank_c + cmu * rank_mu

                # Ensure PSD
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

                if local_best_f < best_f:
                    best_f = local_best_f
                    best_x = local_best_x.copy()

            return local_best_x, local_best_f

        # Phase 1: Sobol sampling + Local Refinement
        n_init = 30
        sobol = scipy.stats.qmc.Sobol(d=dim, scramble=True, seed=42)
        # Generate points in [0, 1]
        sobol_points = sobol.random(n_init)
        # Map to [-5, 5]
        sobol_points = sobol_points * 10.0 - 5.0

        local_candidates = []
        
        for i in range(n_init):
            if calls >= budget or time.time() > deadline:
                break
            x_start = sobol_points[i]
            # Run a small CMA-ES to snap to local minimum
            # Small sigma to stay in basin, small population for speed
            x_local, f_local = cmaes_run(x_start, sigma0=0.5, max_evals=8, rng=rng, lambda_pop=4)
            local_candidates.append((f_local, x_local))

        # Phase 2: Final Polish on the best candidate
        if local_candidates:
            # Sort by fitness
            local_candidates.sort(key=lambda x: x[0])
            best_local_f, best_local_x = local_candidates[0]
            
            # Use remaining budget for a robust CMA-ES
            remaining = budget - calls
            if remaining > 0:
                # Use a slightly larger sigma to allow for fine-tuning if the local snap wasn't perfect
                x_final, f_final = cmaes_run(best_local_x, sigma0=1.0, max_evals=remaining, rng=rng)
                if f_final < best_f:
                    best_f = f_final
                    best_x = x_final

        return best_x, best_f
# EVOLVE-BLOCK-END
