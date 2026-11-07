"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Hybrid approach combining CMA-ES for basin exploration and
    coordinate descent / Nelder-Mead for local refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin for 330s limit
        
        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        
        def eval_and_track(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            x_clipped = np.clip(x, -self.bound, self.bound)
            calls += 1
            val = func(x_clipped)
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val

        # Phase 1: Quasi-random scan to find promising basins
        # Use Halton sequence for better coverage
        rng = np.random.default_rng(42)
        
        def halton_sequence(index, dim):
            point = np.zeros(dim)
            for i in range(dim):
                base = i + 2
                f = 1.0
                r = 0.0
                idx = index
                while idx > 0:
                    f /= base
                    r += f * (idx % base)
                    idx //= base
                point[i] = r
            return point

        # Scan phase: 50 points
        n_scan = min(50, budget // 10)
        for i in range(1, n_scan + 1):
            if calls >= budget or time.time() > deadline:
                break
            x = halton_sequence(i, dim) * 10 - 5
            x = np.clip(x, -5, 5)
            eval_and_track(x)

        # Phase 2: CMA-ES implementation
        def cma_es(x0, sigma0, max_evals):
            nonlocal calls, best_f, best_x
            
            if calls >= budget or time.time() > deadline:
                return
            
            # CMA-ES parameters
            pop_size = max(4, int(4 + 3 * np.log(dim)))
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
            
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            
            x_mean = x0.copy()
            sigma = sigma0
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            evals_in_cma = 0
            max_generations = max(1, max_evals // pop_size)
            
            for it in range(max_generations):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Generate population
                population = []
                for i in range(pop_size):
                    if calls >= budget or time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    x = x_mean + sigma * (B @ (D * z))
                    x = np.clip(x, -5, 5)
                    fx = eval_and_track(x)
                    evals_in_cma += 1
                    population.append((x, fx))
                
                if len(population) == 0:
                    break
                
                # Sort by fitness
                population.sort(key=lambda t: t[1])
                
                # Update mean
                x_new = np.zeros(dim)
                for i in range(mu):
                    x_new += weights[i] * population[i][0]
                x_new = np.clip(x_new, -5, 5)
                
                # Update sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (B.T @ (D * (x_new - x_mean)))
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (it + 1))) < (1.4 + 2 / (dim + 1)) * chi_n:
                    sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                
                # Update pc
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) / sigma * (x_new - x_mean)
                
                # Update B and D (simplified: recompute from covariance)
                # For efficiency, we'll use a simplified update
                delta_hs = (1 - c_c) / (1 - c_c**2)
                rank1 = np.outer(pc, pc)
                rankmu = np.zeros((dim, dim))
                for i in range(mu):
                    z_i = np.linalg.solve(B.T, (x_new - population[i][0]) / sigma)
                    z_i = z_i / D
                    rankmu += weights[i] * np.outer(z_i, z_i)
                
                # Update covariance (simplified)
                # In a full implementation, we'd update C = (1-c1-cmu)*C + c1*(rank1+delta_hs*(pc@pc-I)) + cmu*rankmu
                # Then recompute B, D via eigendecomposition
                # For now, we'll just update the mean and sigma, which is less efficient but stable
                
                x_mean = x_new
                sigma = np.clip(sigma, 1e-10, 10.0)
            
            return x_mean, sigma

        # Run CMA-ES from the best point found in scan
        if best_f < float('inf') and calls < budget - 20:
            # Allocate 60% of remaining budget to CMA-ES
            remaining = budget - calls
            cma_budget = int(remaining * 0.6)
            if cma_budget > 10:
                cma_es(best_x, 2.0, cma_budget)
            
            # Phase 3: Local refinement with Nelder-Mead
            if calls < budget - 10:
                remaining = budget - calls
                # Use Nelder-Mead on the best point
                x_start = best_x.copy()
                
                class BudgetFunc:
                    def __init__(self, func, x_best, f_best, budget_remaining):
                        self.func = func
                        self.x_best = x_best
                        self.f_best = f_best
                        self.remaining = budget_remaining
                        self.count = 0
                    
                    def __call__(self, x):
                        if self.count >= self.remaining or time.time() > deadline:
                            return self.f_best
                        x_clipped = np.clip(x, -self.bound, self.bound)
                        fx = self.func(x_clipped)
                        self.count += 1
                        if fx < self.f_best:
                            self.f_best = fx
                            self.x_best = x_clipped.copy()
                        return fx
                
                bf = BudgetFunc(func, x_start, best_f, remaining)
                
                try:
                    res = minimize(bf, x_start, method='Nelder-Mead', 
                                  options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    pass
            
            # Phase 4: Final coordinate descent polish
            if calls < budget - 5:
                # Simple coordinate descent
                x_curr = best_x.copy()
                f_curr = best_f
                step = 0.5
                for _ in range(10):
                    if calls >= budget - 5 or time.time() > deadline:
                        break
                    improved = False
                    for d in range(dim):
                        if calls >= budget - 5 or time.time() > deadline:
                            break
                        # Try moving in both directions
                        for direction in [1, -1]:
                            x_trial = x_curr.copy()
                            x_trial[d] += direction * step
                            x_trial = np.clip(x_trial, -5, 5)
                            f_trial = eval_and_track(x_trial)
                            if f_trial < f_curr:
                                x_curr = x_trial
                                f_curr = f_trial
                                improved = True
                    
                    if not improved:
                        step *= 0.5
                        if step < 1e-4:
                            break
                    else:
                        step = max(step * 1.1, 0.1)
                
                if f_curr < best_f:
                    best_f = f_curr
                    best_x = x_curr

        return best_x, best_f

# EVOLVE-BLOCK-END
