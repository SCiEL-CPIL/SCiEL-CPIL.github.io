"""Rewritten optimizer for BBOB f14 (Different powers) using Multi-Start Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-Start Nelder-Mead with a small local CMA-ES polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # We'll track the best point and value globally
        best_x = None
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal calls_used, best_x, best_f
            if calls_used >= budget:
                return np.inf
            # Clip to box
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initial evaluations: a few random points to get a sense of the landscape
        n_init = min(10, budget)
        for _ in range(n_init):
            x = rng.uniform(-5, 5, dim)
            evaluate(x)

        if best_x is None:
            # Should not happen, but safety
            x = np.zeros(dim)
            evaluate(x)

        # Multi-start Nelder-Mead
        n_starts = 8
        # We have budget - calls_used left
        remaining = budget - calls_used
        if remaining <= 0:
            return best_x, best_f

        # Allocate remaining budget among starts
        # Use Nelder-Mead with maxiter based on remaining budget
        # Each NM iteration uses up to dim+1 evaluations in the worst case, but typically 1-2
        # Let's allocate a fraction of remaining to each start
        
        # Strategy: use a few strong starts near the best point found, and some random
        # For f14 (Different powers), the landscape is not too multimodal but has different scales
        # Nelder-Mead should work well if started near the optimum
        
        # First, do a coarse grid or random sampling to find a good starting region
        # Then use NM to refine
        
        # Let's use a hybrid: 
        # 1. CMA-ES for global exploration (small population)
        # 2. Nelder-Mead for local exploitation
        
        # Actually, let's stick with Multi-Start NM as suggested by inspiration 1
        # which scored 1.0882
        
        # Allocate budget: use 60% for NM starts, 40% for a final CMA-ES polish
        nm_budget = int(0.6 * remaining)
        cma_budget = int(0.4 * remaining)
        
        # Nelder-Mead implementation
        def nelder_mead(start_x, max_iter, f_eval):
            nonlocal calls_used
            dim = len(start_x)
            # Initialize simplex
            simplex = [start_x.copy()]
            f0 = f_eval(start_x)
            simplex_f = [f0]
            
            # Create initial simplex
            for i in range(dim):
                x_new = start_x.copy()
                if x_new[i] == 0:
                    x_new[i] = 0.00025
                else:
                    x_new[i] *= 1 + 0.00025
                x_new = np.clip(x_new, -5, 5)
                fx = f_eval(x_new)
                simplex.append(x_new)
                simplex_f.append(fx)
            
            # Nelder-Mead parameters
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink
            
            for _ in range(max_iter):
                if calls_used >= budget or time.time() > deadline - 5:
                    break
                    
                # Sort simplex
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = simplex_f[-1]
                
                # Reflection
                x_r = centroid + alpha * (centroid - worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = f_eval(x_r)
                
                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - worst)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = f_eval(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    if f_r < worst_f:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - worst)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = f_eval(x_c)
                    
                    if f_c < min(f_r, worst_f):
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = f_eval(simplex[i])
            
            # Return best
            best_idx = np.argmin(simplex_f)
            return simplex[best_idx], simplex_f[best_idx]
        
        # Multi-start Nelder-Mead
        # Start points: best point found + perturbations + random
        start_points = [best_x.copy()]
        for i in range(n_starts - 1):
            # Perturb best point
            x_pert = best_x + rng.normal(0, 1.0, dim)
            x_pert = np.clip(x_pert, -5, 5)
            start_points.append(x_pert)
        
        # Allocate NM iterations per start
        nm_iter_per_start = max(50, nm_budget // (n_starts * (dim + 1)))
        
        for sp in start_points:
            if calls_used >= budget or time.time() > deadline - 10:
                break
            # Evaluate starting point
            f_sp = evaluate(sp)
            # Run NM
            x_nm, f_nm = nelder_mead(sp, nm_iter_per_start, evaluate)
        
        # CMA-ES polish on the best point
        if cma_budget > 10 and best_x is not None and time.time() < deadline - 20:
            # Simple CMA-ES
            x_mean = best_x.copy()
            sigma = 1.0
            pop = min(4 + int(3 * np.log(dim)), 10)
            # Covariance matrix
            C = np.eye(dim)
            # Evolution paths
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Adaptation parameters
            mu = pop // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights**2)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (weights[0] - 1 + weights[1])) / ((dim + 2)**2 + mu_eff)
            
            # Precompute
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            
            # Inverse sqrt of C for sampling
            def inv_sqrtm(C):
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            B = inv_sqrtm(C)
            D = np.sqrt(np.diag(C @ B.T @ B))
            
            n_cma_iter = cma_budget // pop
            for it in range(n_cma_iter):
                if calls_used >= budget - pop or time.time() > deadline - 5:
                    break
                
                # Sample population
                offspring = []
                for i in range(pop):
                    z = rng.standard_normal(dim)
                    x = x_mean + sigma * (B @ z)
                    x = np.clip(x, -5, 5)
                    offspring.append(x)
                
                # Evaluate
                offspring_f = [evaluate(x) for x in offspring]
                
                # Sort
                order = np.argsort(offspring_f)
                offspring = [offspring[i] for i in order]
                offspring_f = [offspring_f[i] for i in order]
                
                # Update mean
                x_new_mean = np.sum(weights[:, np.newaxis] * np.array(offspring[:mu]), axis=0)
                
                # Update sigma
                x_old_mean = x_mean
                x_mean = x_new_mean
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (
                    B.T @ (x_mean - x_old_mean) / (sigma * chi_n) if sigma > 1e-10 else np.zeros(dim)
                )
                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (pop + 1))) > (1.4 + 2 / (dim + 1)) * chi_n:
                    sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                # Update covariance
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (
                    (x_mean - x_old_mean) / (sigma) if sigma > 1e-10 else np.zeros(dim)
                )
                C = (1 + c_1 * (c_1 - c_mu + np.outer(p_c, p_c)) * (1 - (1 - c_c)**2) - c_1) * C + \
                    c_mu * np.sum(weights[i] * np.outer(offspring[i] - x_old_mean, offspring[i] - x_old_mean) / (sigma**2) for i in range(mu))
                
                # Symmetrize
                C = (C + C.T) / 2
                
                # Update B and D
                B = inv_sqrtm(C)
                D = np.sqrt(np.diag(C @ B.T @ B))
        
        return best_x, best_f
# EVOLVE-BLOCK-END
