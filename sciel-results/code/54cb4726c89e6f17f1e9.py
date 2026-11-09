"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES with Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        rng = np.random.RandomState(42)

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            x = np.clip(x, -5, 5)
            calls_made += 1
            try:
                f_val = float(func(x))
            except Exception:
                f_val = np.inf
            
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def nelder_mead(x0, f0, max_iter=100):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget - 5:
                return x0, f0
            
            # Initialize simplex
            simplex = np.zeros((dim + 1, dim))
            fvals = np.zeros(dim + 1)
            simplex[0] = x0
            fvals[0] = f0
            
            # Expand simplex
            for i in range(dim):
                simplex[i+1] = x0.copy()
                simplex[i+1][i] += 0.5
                fvals[i+1] = safe_call(simplex[i+1])
            
            if calls_made >= budget - 5:
                return x0, f0
                
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 2.0
            
            for _ in range(max_iter):
                if time.time() > deadline or calls_made >= budget - 5:
                    break
                
                # Sort by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                
                # Update global best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()
                
                # Check convergence
                if np.std(fvals) < 1e-10:
                    break
                
                # Centroid of best dim points
                centroid = np.mean(simplex[:dim], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = safe_call(x_ref)
                
                if f_ref < fvals[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = safe_call(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    # Simple reflection
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < fvals[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (x_ref - centroid)
                    
                    x_con = np.clip(x_con, -5, 5)
                    f_con = safe_call(x_con)
                    
                    if f_con < fvals[-1]:
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = safe_call(simplex[i])
            
            return simplex[np.argmin(fvals)], np.min(fvals)

        def cma_es(x0, f0, max_iter=150):
            nonlocal calls_made, best_x, best_f
            
            pop_size = 12
            if pop_size > budget - calls_made - 50: # Reserve for polish
                pop_size = max(4, (budget - calls_made - 50) // 2)
            
            sigma = 1.0
            cc = (4 + dim / (dim + 4)) / ((dim + 1.3) + 4 + dim / (dim + 4))
            cmu = min(0.5, (2 * (dim - 2) + 1 / (dim + 2)) / ((dim + 2) * (dim + 2) + 2))
            
            mean = x0.copy()
            C = np.eye(dim)
            sqrtC = np.eye(dim)
            
            dsigma = 1 + 2 * max(0, cmu - 0.5) + cmu
            dlambda = 1 + 2 * max(0, cmu - 0.5) + cmu
            
            ps = np.zeros(dim)
            pc = np.zeros((dim, dim))
            
            chi_mu = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            
            indices = np.arange(pop_size)
            half = pop_size // 2
            
            # Weights
            weights = np.zeros(pop_size)
            weights[:half] = np.log(half + 0.5) - np.log(np.arange(1, half + 1))
            weights /= weights.sum()
            mueff = 1 / np.sum(weights ** 2)
            
            pop_x = np.zeros((pop_size, dim))
            pop_f = np.zeros(pop_size)
            
            pop_x[0] = mean
            pop_f[0] = f0
            
            for i in range(1, pop_size):
                z = rng.standard_normal(dim)
                pop_x[i] = mean + sigma * (sqrtC @ z)
                pop_x[i] = np.clip(pop_x[i], -5, 5)
                pop_f[i] = safe_call(pop_x[i])
            
            for it in range(max_iter):
                if calls_made >= budget - 50 or time.time() > deadline:
                    break
                
                order = np.argsort(pop_f)
                pop_x = pop_x[order]
                pop_f = pop_f[order]
                
                if pop_f[0] < best_f:
                    best_f = pop_f[0]
                    best_x = pop_x[0].copy()
                
                if np.std(pop_f[:half]) < 1e-12 * (1 + abs(pop_f[0])):
                    break
                
                old_mean = mean.copy()
                mean = np.sum(weights[:, np.newaxis] * pop_x, axis=0)
                
                diff = (mean - old_mean) / (sigma * np.sqrt(np.sum(weights ** 2)))
                ps = (1 - dsigma) * ps + np.sqrt(dsigma * (2 - dsigma) * mueff / dim) * diff
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - dsigma) ** (2 * (it + 1)))) < (1.4 + 2 / (dim + 1)) * chi_mu
                pc = (1 - cmu) * pc + hsig * cmu * (diff / np.linalg.norm(diff) * np.sqrt(mueff) if np.linalg.norm(diff) > 0 else np.zeros(dim))
                
                C = (1 + cmu * (2 - cmu) * (1 - hsig) - cmu) * C
                rank_mu = np.outer(mean - old_mean, mean - old_mean) / np.sum(weights ** 2)
                rank_one = np.outer(diff, diff)
                C += cmu * (rank_mu + (1 - hsig) * rank_one)
                
                sigma *= np.exp((dsigma / 5 * dim) * (np.linalg.norm(ps) / chi_mu - 1))
                sigma = np.clip(sigma, 1e-10, 100)
                
                try:
                    eigenvalues, eigenvectors = np.linalg.eigh(C)
                    eigenvalues = np.clip(eigenvalues, 1e-12, None)
                    sqrtC = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
                except:
                    sqrtC = np.eye(dim)
                
                for i in range(1, pop_size):
                    z = rng.standard_normal(dim)
                    pop_x[i] = mean + sigma * (sqrtC @ z)
                    pop_x[i] = np.clip(pop_x[i], -5, 5)
                    pop_f[i] = safe_call(pop_x[i])
                
                pop_x[0] = best_x if best_x is not None else mean
                pop_f[0] = best_f if best_f != np.inf else f0
            
            return best_x, best_f

        # Phase 1: Global Search
        # Start with a few diverse points
        starts = [np.zeros(dim)]
        for _ in range(4):
            starts.append(rng.uniform(-4, 4, dim))
        
        best_start_x = None
        best_start_f = np.inf
        for s in starts:
            f = safe_call(s)
            if f < best_start_f:
                best_start_f = f
                best_start_x = s
        
        # Run CMA-ES from the best start
        # Reserve ~50 calls for polish
        cma_budget_limit = budget - 50
        cma_iter = (cma_budget_limit - calls_made) // 12
        cma_iter = max(10, min(cma_iter, 100))
        
        cma_es(best_start_x, best_start_f, max_iter=cma_iter)
        
        # Phase 2: Local Polish
        if calls_made < budget - 10 and best_x is not None:
            # Run Nelder-Mead from the best point found
            nelder_mead(best_x, best_f, max_iter=50)
            
            # If budget remains, try polishing a second best point if we kept track
            # For simplicity, just run NM again if budget allows, or stop.
            if calls_made < budget - 10:
                 # Re-run NM with more iterations if possible? 
                 # Or just let it be. NM usually converges fast.
                 pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
