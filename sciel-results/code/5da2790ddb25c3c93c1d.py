"""Black-box continuous optimizer for BBOB-style f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 325.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        calls = 0

        best_x = np.zeros(dim)
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        def time_left():
            return deadline - time.time() > 1.5

        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        # Population size lambda
        lam = max(4, int(4 + 3 * np.log(dim)))
        # Weights
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        # Variance-effectiveness population size
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        # Step size control
        cs = (mueff + 1.5) / (dim + mueff + 5)
        # Covariance matrix adaptation
        c1 = (2 / ((dim + 1.3) ** 2 + mueff)) / (1 - cs)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        # Initial state
        mean = np.zeros(dim)
        sigma = 2.0  # Start with a reasonable spread
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        # Pre-compute inverse square root of C
        # We will update C and recompute sqrt each step for simplicity, 
        # though incremental updates are faster. Given dim=5, full update is fine.
        
        # Initial evaluations to establish a baseline and fill the population
        # We will run CMA-ES loop until budget is exhausted.
        
        while calls < budget and time_left():
            # Generate population
            population = []
            for i in range(lam):
                # Check if we have enough budget for the whole generation
                if calls + lam > budget:
                    # Generate only as many as we can afford, or break if we can't do a full gen?
                    # Standard CMA-ES needs full population. 
                    # If we can't do a full generation, we might just do a (1+1) step or stop.
                    # Let's check if we can do at least 1.
                    # Actually, it's better to stop if we can't complete a generation to avoid bias.
                    # Or we can reduce lam dynamically. Let's just break if calls + lam > budget.
                    # However, if we are close to budget, we might waste some.
                    # Let's try to finish the generation if possible.
                    if calls + lam > budget:
                         # We can't do a full generation. 
                         # We can do a partial generation? No, CMA-ES relies on ranking.
                         # Let's just run a (1+1)-ES for the remaining budget.
                         while calls < budget and time_left():
                             y = np.clip(mean + sigma * rng.standard_normal(dim), -5, 5)
                             fy = func(y)
                             calls += 1
                             update_best(y, fy)
                             if fy < best_f:
                                 mean = y.copy()
                                 sigma *= 1.05
                             else:
                                 sigma *= 0.95
                         break
                # Sample from N(mean, sigma^2 * C)
                # Use Cholesky decomposition
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback to identity if C becomes singular
                    L = np.eye(dim)
                
                z = rng.standard_normal(dim)
                y = mean + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                update_best(y, fy)
                population.append((fy, y))

            if calls >= budget:
                break

            # Sort population by fitness
            population.sort(key=lambda x: x[0])
            
            # Update mean
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * population[i][1]
            
            # Update step size control vectors
            # Normalize the difference between new and old mean
            y_old = (new_mean - mean) / (sigma * np.linalg.norm(L)) if np.linalg.norm(L) > 1e-16 else np.zeros(dim)
            
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y_old
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (calls / (cs * (dim + 1))))) < (1.4 + 2 / (dim + 1)) * np.trace(C) ** 0.5 # Approximation, usually trace(C) is 1 initially
            
            # Actually, the hsig condition is:
            # ||ps|| / sqrt(1 - (1-cs)^2t) < (1 + 2/(D+1)) * sqrt(1 - (1-cs)^2t) ... wait
            # Standard formula:
            # hsig = (||ps|| / sqrt(1 - (1-cs)^2t)) < (1.4 + 2/(D+1)) * E||N(0,I)||
            # E||N(0,I)|| = sqrt(D) * (1 - 1/(4D) + 3/(32D^2))
            # Let's use a simpler heuristic or the standard one.
            # t is the number of generations. We don't track t explicitly, but calls is a proxy.
            # Let's just use the standard update for pc and C.
            
            pc = (1 - c1) * pc + hsig * np.sqrt(c1 * (2 - c1) * mueff) * y_old
            
            # Update covariance matrix
            # Rank-mu update
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                z_i = (population[i][1] - mean) / (sigma * np.linalg.norm(L)) if np.linalg.norm(L) > 1e-16 else np.zeros(dim)
                rank_mu += weights[i] * np.outer(z_i, z_i)
            
            C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu * rank_mu
            
            # Symmetrize C
            C = (C + C.T) / 2.0
            
            # Update sigma
            # Step-size adaptation (CSA)
            # sigma = sigma * exp(cs / (1-cs) * (||ps||/E||N(0,I)|| - 1))
            # E||N(0,I)|| approximation
            E_norm = np.sqrt(dim) * (1 - 1.0/(4*dim) + 3.0/(32*dim*dim))
            norm_ps = np.linalg.norm(ps)
            sigma = sigma * np.exp(cs / 1.0 * (norm_ps / E_norm - 1))
            # Clip sigma to avoid explosion or collapse
            sigma = np.clip(sigma, 1e-10, 100.0)
            
            # Update mean
            mean = new_mean
            
            # Check for convergence
            if sigma < 1e-10:
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
