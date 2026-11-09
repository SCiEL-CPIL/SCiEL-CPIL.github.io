"""Portfolio-based CMA-ES optimizer for BBOB f7 (Step ellipsoidal) in 5D."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        dim = self.dim
        total_budget = self.budget
        
        # Global best tracking across all instances
        global_best_x = None
        global_best_f = np.inf
        
        # We will run multiple CMA-ES instances.
        # In 5D, CMA-ES is fast. We can afford several restarts.
        # Let's try 5 instances, each with a subset of the budget.
        # Or, run them sequentially, stopping early if one converges well.
        # A common strategy is to split the budget: e.g., 5 instances, each gets budget/5.
        # If an instance finds a very good solution, we can stop early or give more time to others?
        # Simpler: Fixed split. 5 instances, 100 evals each.
        
        num_instances = 5
        budget_per_instance = total_budget // num_instances
        
        # Ensure at least 10 evals per instance
        if budget_per_instance < 10:
            num_instances = max(1, total_budget // 10)
            budget_per_instance = total_budget // num_instances
            
        # Starting points for diversity
        # Center is often good, but we want diversity.
        # Use a seeded RNG for reproducibility within the instance
        rng_main = np.random.default_rng(12345)
        
        for inst_idx in range(num_instances):
            if time.time() > deadline:
                break
            
            # Check if we have enough time for a meaningful run
            # CMA-ES in 5D is very fast per eval.
            
            # Initialize this instance
            # Different starting points for each instance
            # Instance 0: Center
            # Instance 1-4: Random points in [-4, 4]
            if inst_idx == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng_main.uniform(-4, 4, dim)
            
            # Run CMA-ES for this instance
            best_x, best_f, calls_used = self._run_cma_es(func, x0, budget_per_instance, inst_idx)
            
            if best_f < global_best_f:
                global_best_f = best_f
                global_best_x = best_x.copy()
                
            # Early stopping heuristic:
            # If we found a very good solution (e.g., f < 1.0, since f_opt is in [-100, 100] and typical minima are near 0 for BBOB),
            # we might stop early to save time for other instances?
            # Actually, the budget is fixed per instance in this simple split.
            # But if time is running out, we break the outer loop.
            
            # Optional: If the first instance finds a very good solution, 
            # we could skip the rest? No, better to use the time to refine or check others.
            # But if we are close to deadline, we should stop.

        if global_best_x is None:
            # Fallback if something went wrong
            global_best_x = np.zeros(dim)
            global_best_f = func(global_best_x)
            
        return global_best_x, global_best_f

    def _run_cma_es(self, func, x0, budget, seed):
        """
        Runs a single CMA-ES instance.
        Returns (best_x, best_f, calls_made).
        """
        dim = self.dim
        rng = np.random.default_rng(seed)
        
        # CMA-ES Parameters
        lambda_pop = 4 + int(3 * np.log(dim))  # ~11 for dim=5
        mu = lambda_pop // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Adaptation rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        
        # Initialization
        x_mean = x0.copy()
        C = np.eye(dim)
        sigma = 2.0  # Initial step size
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls_made = 1
        
        # Pre-compute Chi_n
        chi_n = np.sqrt(dim) * (1 - 1. / (4. * dim) + 1. / (21. * dim ** 2))
        
        # Main loop
        while calls_made < budget:
            # Generate population
            ys = np.zeros((lambda_pop, dim))
            xs = np.zeros((lambda_pop, dim))
            fs = np.zeros(lambda_pop)
            
            for i in range(lambda_pop):
                y = rng.standard_normal(dim)
                ys[i] = y
                
                # Sample from N(0, C)
                # Use Cholesky decomposition for speed and stability in 5D
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                    
                z = L @ y
                x_i = x_mean + sigma * z
                x_i = np.clip(x_i, -5, 5)
                
                xs[i] = x_i
                fs[i] = func(x_i)
                calls_made += 1
                
                if fs[i] < best_f:
                    best_f = fs[i]
                    best_x = x_i.copy()
                    
                if calls_made >= budget:
                    break
            
            if calls_made >= budget:
                break
                
            # Sort by fitness
            idx = np.argsort(fs)
            xs_sorted = xs[idx]
            fs_sorted = fs[idx]
            ys_sorted = ys[idx]
            
            # Update mean
            x_old = x_mean
            x_mean = np.sum(weights[:mu] * xs_sorted[:mu], axis=0)
            
            # Update p_sigma and p_c
            # C_inv_sqrt is needed for p_sigma update in standard CMA-ES
            # But for simplicity and speed in 5D, we can use the standard update without full C_inv_sqrt 
            # if we accept slightly less optimal adaptation, OR compute it.
            # Let's compute it properly.
            try:
                L = np.linalg.cholesky(C)
                C_inv_sqrt = np.linalg.inv(L).T
            except:
                C_inv_sqrt = np.eye(dim)
                
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt @ (x_mean - x_old) / sigma
            
            # h_sigma check
            gen = calls_made / lambda_pop
            norm_p_sigma = np.linalg.norm(p_sigma)
            thresh = (1.4 + 2. / (dim + 1)) * np.sqrt((mueff + 2) - 2)
            # Approximate denominator sqrt(1 - (1-cs)^{2/gen})
            # For small gen, this is ~1. For large gen, it approaches 1/sqrt(cs).
            # Standard check: norm_p_sigma / sqrt(1 - (1-cs)^(2/gen)) < thresh
            # Let's use a simplified check for speed
            h_sigma = 1.0 if norm_p_sigma < thresh else 0.0
            
            p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mueff) * (x_mean - x_old) / sigma
            
            # Update C
            # Rank-1 update
            rank_one = np.outer(x_mean - x_old, x_mean - x_old)
            delta_h = (1 - h_sigma) * cc * (2 - cc) * np.outer(p_c, p_c)
            
            # Rank-mu update
            rank_mu = np.sum(weights[:mu][:, np.newaxis] * (ys_sorted[:mu] * ys_sorted[:mu].T), axis=0)
            
            C = (1 + c1 + cmu - c1 - cmu) * C + c1 * (rank_one + delta_h) + cmu * rank_mu
            C = (C + C.T) / 2
            
            # Ensure positive definite
            try:
                np.linalg.cholesky(C)
            except:
                C += 1e-6 * np.eye(dim)
                
            # Adapt sigma
            lambda_sigma = 1.0 / (cs * (2 - cs))
            sigma *= np.exp(cs / (2 * lambda_sigma) * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Early termination if sigma is very small
            if sigma < 1e-8:
                break
                
        return best_x, best_f, calls_made
# EVOLVE-BLOCK-END
