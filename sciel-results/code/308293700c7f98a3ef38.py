"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: Multi-start exploration + CMA-ES + Nelder-Mead for 5D BBOB functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            calls += 1
            try:
                f = float(func(x))
            except Exception:
                f = best_f
            if np.isfinite(f) and f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def nelder_mead(func, x0, max_iter=100):
            """Nelder-Mead simplex method."""
            nonlocal calls
            dim = len(x0)
            
            # Initialize simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            for i in range(dim):
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += 0.1 if x0[i] < 4.9 else -0.1
                simplex[i + 1] = np.clip(simplex[i + 1], -5, 5)
            
            # Evaluate initial simplex
            fvals = np.array([safe_func(simplex[i]) for i in range(dim + 1)])
            
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink
            
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                if calls >= budget:
                    break
                
                # Sort by function values
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                
                # Check convergence
                if np.max(fvals) - np.min(fvals) < 1e-10:
                    break
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:dim], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = safe_func(x_ref)
                
                if f_ref < fvals[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = safe_func(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < fvals[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = safe_func(x_con)
                    
                    if f_con <= min(f_ref, fvals[-1]):
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = safe_func(simplex[i])
            
            return simplex[np.argmin(fvals)], np.min(fvals)

        # Phase 1: Multi-start exploration
        n_starts = 15
        start_budget = min(budget // 3, 50)
        
        # Generate diverse starting points
        np.random.seed(42)
        starts = []
        # Include origin
        starts.append(np.zeros(dim))
        # Random points in [-4, 4]^5
        for _ in range(n_starts - 1):
            x = np.random.uniform(-4, 4, size=dim)
            starts.append(x)
        
        # Evaluate all starts
        for x in starts:
            if calls >= start_budget:
                break
            safe_func(x)
        
        # Get best starting point
        # best_x and best_f are already updated by safe_func
        
        # Phase 2: CMA-ES from best point
        if calls < budget and time.time() < deadline:
            # CMA-ES parameters
            lambda_ = 10  # small population for 300 calls
            mu = lambda_ // 2
            
            # Recombination weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            
            # Step size adaptation
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            
            # Covariance matrix adaptation
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            
            # Initialize CMA-ES from best point found
            x_mean = best_x.copy()
            sigma = 1.0
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            C = np.eye(dim)
            
            remaining_budget = budget - calls
            
            while calls < budget:
                if time.time() > deadline:
                    break
                
                # Generate population
                pop = np.zeros((lambda_, dim))
                for i in range(lambda_):
                    z = np.random.standard_normal(dim)
                    y = B @ (D * z)
                    pop[i] = np.clip(x_mean + sigma * y, -5, 5)
                
                # Evaluate population
                fvals = np.zeros(lambda_)
                for i in range(lambda_):
                    if calls >= budget:
                        break
                    fvals[i] = safe_func(pop[i])
                
                if calls >= budget:
                    break
                
                # Sort by fitness
                indices = np.argsort(fvals)
                pop_sorted = pop[indices]
                fvals_sorted = fvals[indices]
                
                # Weighted recombination
                x_new = np.zeros(dim)
                for i in range(mu):
                    x_new += weights[i] * pop_sorted[i]
                
                # Update evolution paths
                y_diff = (x_new - x_mean) / sigma
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_diff
                cs_norm = np.linalg.norm(pc) / chi_n
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_diff
                
                # Update step size
                hsig = (cs_norm < (1.4 + 1 / dim) * chi_n)
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                
                # Update covariance matrix
                if hsig:
                    delta_hs = (1 - c_c) / c_c
                else:
                    delta_hs = 1
                rank1 = np.outer(pc, pc)
                rankmu = np.zeros((dim, dim))
                for i in range(mu):
                    zi = (pop_sorted[i] - x_mean) / sigma
                    rankmu += weights[i] * np.outer(zi, zi)
                C = (1 + c_c * (delta_hs - 1) - c_1 - c_mu) * C + c_1 * rank1 + c_mu * rankmu
                
                # Update eigen decomposition
                D2, B2 = np.linalg.eigh(C)
                D2 = np.sqrt(np.maximum(D2, 1e-20))
                B = B2
                D = D2
                
                x_mean = x_new
                
                # Check if CMA-ES is making progress
                if fvals_sorted[0] >= best_f - 1e-12:
                    # No improvement, break to Nelder-Mead
                    break
        
        # Phase 3: Nelder-Mead from best point
        if calls < budget and time.time() < deadline:
            x_nm, f_nm = nelder_mead(safe_func, best_x.copy(), max_iter=50)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
