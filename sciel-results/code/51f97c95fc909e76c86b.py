"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """Multi-Start CMA-ES optimizer for Katsuura-style multimodal function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        rng = np.random.default_rng(42)

        best_x = np.zeros(n)
        best_f = float('inf')
        total_calls = 0

        def cma_es(x0, sigma0, max_calls, rng_local):
            """Run CMA-ES from x0 with initial sigma0, using up to max_calls evaluations."""
            nonlocal best_x, best_f, total_calls
            
            # CMA-ES parameters
            x = x0.copy()
            sigma = sigma0
            lambda_ = 4 + int(3 * np.log(n))
            mu = lambda_ // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / weights.sum()
            
            # Covariance matrix and other state
            C = np.eye(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            cs = (mueff + 2) / (n + mueff + 5)
            ds = 1 + 2 * max(0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
            cc = (4 + mueff / (n + 4)) / ((n + 2) + mueff / 2)
            c1 = (mueff - 2 + 2 / (n + mueff)) / ((n + 2) ** 2 + mueff - 2)
            cmu = min(1 - c1, 2 * (mueff - 2 + 2 / (mueff)) / ((n + 2) ** 2 + mueff))
            
            # Initial population
            pop = []
            for i in range(lambda_):
                z = rng_local.standard_normal(n)
                xi = x + sigma * z
                pop.append(xi)
                
            # Evaluate initial population
            pop_f = []
            for xi in pop:
                if total_calls >= budget or time.time() > deadline:
                    return
                yi = func(xi)
                total_calls += 1
                pop_f.append(yi)
                if yi < best_f:
                    best_f = yi
                    best_x = xi.copy()

            # Main loop
            while total_calls < budget and time.time() < deadline:
                # Sort population by fitness
                order = np.argsort(pop_f)
                pop = [pop[i] for i in order]
                pop_f = [pop_f[i] for i in order]
                
                # Update mean
                x_old = x
                x = sum(weights[i] * pop[i] for i in range(mu))
                
                # Update step size control (CSA)
                ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) / sigma * C @ (x - x_old)
                hsig = (np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * (total_calls - len(pop_f))))) < (1 + 2 / (n + 1)) * math.sqrt(n + 1)
                
                # Update evolution paths
                pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * (x - x_old) / sigma
                
                # Update covariance matrix
                delta_d = (1 - c1) * (1 + cmu * mueff)
                rank_one = np.outer(pc, pc)
                rank_mu = sum(weights[i] * np.outer(pop[i] - x_old, pop[i] - x_old) for i in range(mu))
                C = (1 - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu
                
                # Update sigma
                sigma *= math.exp(cs / ds * (np.linalg.norm(ps) / math.sqrt(n) - 1))
                
                # Check termination conditions
                if sigma < 1e-12 or np.max(np.abs(C)) < 1e-12:
                    break
                    
                # Generate new population
                new_pop = []
                for i in range(lambda_):
                    z = rng_local.standard_normal(n)
                    # Approximate Cholesky for sampling
                    try:
                        L = np.linalg.cholesky(C)
                        xi = x + sigma * (L @ z)
                    except np.linalg.LinAlgError:
                        # Fallback to diagonal if Cholesky fails
                        diag = np.sqrt(np.diag(C))
                        xi = x + sigma * (diag * z)
                    new_pop.append(xi)
                    
                # Evaluate new population
                new_pop_f = []
                for xi in new_pop:
                    if total_calls >= budget or time.time() > deadline:
                        return
                    yi = func(xi)
                    total_calls += 1
                    new_pop_f.append(yi)
                    if yi < best_f:
                        best_f = yi
                        best_x = xi.copy()
                        
                pop = new_pop
                pop_f = new_pop_f

        # Phase 1: Initial Sampling (Latin Hypercube)
        # Use 20% of budget for global exploration
        n_init = max(20, budget // 5)
        
        X_init = np.zeros((n_init, n))
        for j in range(n):
            # LHS with jitter
            X_init[:, j] = (np.arange(n_init) + rng.uniform(size=n_init)) / n_init
            rng.shuffle(X_init[:, j])
        X_init = X_init * 10 - 5 # Scale to [-5, 5]
        
        y_init = np.zeros(n_init)
        for i in range(n_init):
            if total_calls >= budget or time.time() > deadline:
                break
            x = X_init[i]
            y = func(x)
            total_calls += 1
            y_init[i] = y
            if y < best_f:
                best_f = y
                best_x = x.copy()

        # Identify top K points for multi-start
        top_k = 3
        if n_init < top_k:
            top_k = n_init
            
        # Get indices of best points
        best_indices = np.argsort(y_init)[:top_k]
        
        # Phase 2: Multi-Start CMA-ES
        # Distribute remaining budget among top_k starts
        remaining_budget = budget - total_calls
        if remaining_budget > 0:
            budget_per_start = remaining_budget // top_k
            
            # Create sub-rngs for reproducibility and parallelism simulation
            sub_rngs = [np.random.default_rng(42 + i) for i in range(top_k)]
            
            for i in range(top_k):
                if total_calls >= budget or time.time() > deadline:
                    break
                
                x_start = X_init[best_indices[i]].copy()
                # Initial sigma: roughly 10% of box width, scaled by distance to boundary
                sigma_start = 0.5
                
                cma_es(x_start, sigma_start, budget_per_start, sub_rngs[i])
                
                # If budget is left over (due to integer division or early termination), 
                # we can do a final polish with the best point found so far
                if total_calls < budget and time.time() < deadline:
                    # Simple local polish: small random steps around best_x
                    polish_calls = min(10, budget - total_calls)
                    for _ in range(polish_calls):
                        if total_calls >= budget or time.time() > deadline:
                            break
                        x_pert = best_x + rng.normal(0, 0.1, n)
                        x_pert = np.clip(x_pert, -5, 5)
                        y_pert = func(x_pert)
                        total_calls += 1
                        if y_pert < best_f:
                            best_f = y_pert
                            best_x = x_pert.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
