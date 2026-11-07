"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Optimizer for BBOB-style f18 (Schaffers F7 ill-conditioned) on [-5,5]^5.
    
    Strategy: Multi-start with CMA-ES local search.
    - Uses a population-based approach with multiple restarts.
    - Each restart uses a simplified CMA-ES to handle ill-conditioning and local minima.
    - Global exploration via diverse random starts.
    - Dynamic budget allocation based on progress.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es(self, func, x0, budget, rng):
        """
        Simplified CMA-ES implementation.
        Returns best_x, best_f, calls_used.
        """
        dim = self.dim
        # CMA-ES parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(dim))  # ~13 for dim=5
        mu = lam // 2
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1 / np.sum(w**2)
        
        # Initial sigma
        sigma = 0.5
        x_mean = x0.copy()
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        
        # Step size control
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        p_sigma = np.zeros(dim)
        
        # Covariance matrix adaptation
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        d_c = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_c
        p_c = np.zeros(dim)
        
        best_x = x0.copy()
        best_f = func(x0)
        calls = 1
        
        # Precompute inverse sqrt of C for efficiency? No, dim is small.
        # We need to sample from N(0, C). Since C is updated, we use Cholesky or similar.
        # For simplicity and robustness, we will use the standard update rule.
        
        for gen in range(budget // lam + 1):
            if calls >= budget:
                break
            
            # Sample population
            Y = np.zeros((lam, dim))
            for i in range(lam):
                # Sample z ~ N(0, I)
                z = rng.standard_normal(dim)
                # y = x_mean + sigma * C_sqrt @ z
                # We need to maintain C_sqrt. In standard CMA, C is updated, and we recompute sqrt.
                # For dim=5, recomputing Cholesky is cheap.
                try:
                    L = np.linalg.cholesky(C)
                    Y[i] = x_mean + sigma * (L @ z)
                except np.linalg.LinAlgError:
                    # Fallback if C becomes non-positive definite
                    L = np.eye(dim)
                    Y[i] = x_mean + sigma * z
                
                # Clip to bounds
                Y[i] = np.clip(Y[i], -5, 5)
                
                # Evaluate
                fy = func(Y[i])
                calls += 1
                if fy < best_f:
                    best_f = fy
                    best_x = Y[i].copy()
            
            # Sort by fitness
            idx = np.argsort([func(Y[i]) for i in range(lam)]) # This is inefficient, re-evaluating.
            # We should have stored fitnesses. Let's fix the loop to store fitnesses.
            
            # Re-do the sampling loop to store fitnesses properly
            # Actually, let's just break and restart the logic inside the loop properly.
            # The above loop evaluated func twice for sorting. Let's optimize.
            # I will refactor the inner loop.
            
            # Break out and implement correctly below.
            pass 

        return best_x, best_f, calls

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        # Multi-start strategy
        # We will try several starts, running a limited CMA-ES from each.
        n_starts = 5
        # Allocate budget per start, reserving some for the best one to continue
        # Let's give each start 1/3 of the budget, and if one is clearly better, continue it.
        
        starts = []
        for i in range(n_starts):
            # Random start in [-5, 5]
            x0 = rng.uniform(-5, 5, dim)
            starts.append(x0)
            
        # Run CMA-ES from each start with a limited budget
        # We'll run them sequentially, updating the global best.
        # If a run finds a significantly better solution, we might allocate more budget to it.
        
        # Simplified approach:
        # 1. Evaluate all starts.
        # 2. Pick the best 2 starts.
        # 3. Run CMA-ES on the best one with remaining budget.
        # 4. If budget remains, run CMA-ES on the second best.
        
        # Step 1: Evaluate starts
        start_results = []
        for x0 in starts:
            if calls_used >= budget:
                break
            f0 = func(x0)
            calls_used += 1
            start_results.append((f0, x0))
        
        # Sort starts by fitness
        start_results.sort(key=lambda x: x[0])
        
        # Step 2: Run CMA-ES on the best start
        if start_results:
            best_f_start, best_x_start = start_results[0]
            if best_f_start < best_f:
                best_f = best_f_start
                best_x = best_x_start.copy()
            
            # Run CMA-ES from best start
            # Allocate 60% of remaining budget to the best start
            remaining_budget = budget - calls_used
            cma_budget = int(remaining_budget * 0.6)
            
            if cma_budget > 10:
                # Implement a proper CMA-ES run
                x_best, f_best, calls_cma = self._run_cma_es(func, best_x_start, cma_budget, rng, calls_used)
                calls_used += calls_cma
                if f_best < best_f:
                    best_f = f_best
                    best_x = x_best.copy()
                
                # Step 3: If budget remains, try the second best start
                if len(start_results) > 1 and calls_used < budget:
                    f_second, x_second = start_results[1]
                    # Only try if the first one didn't converge to a very good value
                    # and the second start was reasonably good
                    if f_second < best_f + 10: # Heuristic: if second start is close
                        remaining2 = budget - calls_used
                        cma_budget2 = remaining2
                        if cma_budget2 > 10:
                            x_best2, f_best2, calls_cma2 = self._run_cma_es(func, x_second, cma_budget2, rng, calls_used)
                            calls_used += calls_cma2
                            if f_best2 < best_f:
                                best_f = f_best2
                                best_x = x_best2.copy()
        
        return best_x, best_f

    def _run_cma_es(self, func, x0, budget, rng, calls_offset):
        """
        Runs CMA-ES from x0 for 'budget' function calls.
        Returns best_x, best_f, calls_used_in_this_run.
        """
        dim = self.dim
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1 / np.sum(w**2)
        
        sigma = 0.5
        x_mean = x0.copy()
        C = np.eye(dim)
        
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        p_sigma = np.zeros(dim)
        
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        d_c = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_c
        p_c = np.zeros(dim)
        
        best_x = x0.copy()
        best_f = func(x0)
        calls = 1
        
        while calls < budget:
            # Sample population
            Y = np.zeros((lam, dim))
            F = np.zeros(lam)
            
            # Compute Cholesky of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)
            
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                Y[i] = y
                F[i] = func(y)
                calls += 1
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = y.copy()
                
                if calls >= budget:
                    break
            
            if calls >= budget:
                break
            
            # Sort indices
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]
            
            # Update mean
            x_mean_new = np.sum(w[:, None] * Y[:mu], axis=0)
            
            # Update sigma
            if np.linalg.norm(L) > 1e-12:
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (L.T @ (x_mean_new - x_mean) / sigma)
            
            chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            sigma *= np.exp(c_sigma/d_sigma * (np.linalg.norm(p_sigma)/chi_n - 1))
            
            # Update C
            y_vec = (x_mean_new - x_mean) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * y_vec
            
            h_sig = (np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c)**(2*(calls/lam)))) / chi_n < 1.4 + 2/(dim+1)
            
            C = (1 - c_1 - c_mu + (not h_sig) * c_1 * c_c * (2 - c_c)) * C \
                + c_1 * (p_c[:, None] * p_c[None, :] + (not h_sig) * c_c * (2 - c_c)) \
                + c_mu * np.sum(w[:, None] * (Y[:mu] - x_mean)[:, None] * (Y[:mu] - x_mean)[None, :], axis=0)
            
            # Symmetrize
            C = (C + C.T) / 2
            
            x_mean = x_mean_new
            
            # Check for stagnation
            if np.linalg.norm(x_mean_new - x_mean) < 1e-8 and sigma < 1e-6:
                break
        
        return best_x, best_f, calls
# EVOLVE-BLOCK-END
