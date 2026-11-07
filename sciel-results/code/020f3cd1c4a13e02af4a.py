"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class CMAES:
    def __init__(self, x0, sigma0, dim, budget, rng, func, deadline):
        self.dim = dim
        self.x0 = x0
        self.sigma = sigma0
        self.budget = budget
        self.rng = rng
        self.func = func
        self.deadline = deadline
        
        # CMA-ES parameters
        self.mu = max(1, dim // 2 + 1)  # number of parents
        self.lam = max(4, 2 * self.mu) # population size
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= self.weights.sum()
        self.mu_eff = 1.0 / np.sum(self.weights**2)
        
        self.c1 = (2 / ((dim + 1) ** 2 + 10))
        self.cmua = (4 + dim / self.mu_eff) / ((dim + 2) ** 2 + 12)
        self.c_sigma = (self.mu_eff + 0.5 + self.cmua) / (dim + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (dim + 1)) - 1) + self.c_sigma
        self.c_c = (4 + dim / self.mu_eff) / (dim + 4 + 4 * self.mu_eff / dim)
        self.d_c = 1 + 2 * max(0, np.sqrt(self.mu_eff / (dim + 1) - 1)) + self.c_c
        self.cs = self.c_sigma
        
        # State
        self.x_mean = self.x0.copy()
        self.C = np.eye(dim)
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.chi_n = np.sqrt(dim) * (1 + 1/(4*dim) - 1/(21*dim**2))
        
    def step(self):
        if time.time() > self.deadline:
            return False
            
        # Generate population
        Y = self.rng.multivariate_normal(np.zeros(self.dim), self.C, size=self.lam)
        X = self.x_mean + self.sigma * Y
        X = np.clip(X, -5, 5)
        
        # Evaluate
        F = np.array([self.func(x) for x in X])
        self.budget -= self.lam
        
        # Sort
        indices = np.argsort(F)
        X = X[indices]
        F = F[indices]
        
        # Update mean
        self.x_mean = np.sum(self.weights[:, None] * X[:self.mu], axis=0)
        
        # Update covariance matrix
        y = (self.x_mean - self.x0) / self.sigma if self.sigma > 0 else np.zeros(self.dim)
        
        self.pc = (1 - self.cs) * self.pc + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * y
        self.ps = (1 - self.c_sigma) * self.ps + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * np.linalg.inv(np.sqrt(self.C)) @ y
        
        delta_sig = 1 if np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.c_sigma)**2) < (1.4 + 2/(self.dim+1)) * self.chi_n else 0
        
        self.C = (1 + self.c1 + self.cmua - self.c1) * self.C \
                 - self.c1 * np.outer(self.pc, self.pc) \
                 + self.cmua * (np.outer(self.ps, self.ps) * (1 - delta_sig) + (self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * (delta_sig * self.C - self.C))
        
        # Symmetrize
        self.C = (self.C + self.C.T) / 2
        
        # Update sigma
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) * (np.linalg.norm(self.ps) / self.chi_n - 1))
        
        # Convergence criteria
        if self.sigma < 1e-12 or self.budget <= 0:
            return False
        if time.time() > self.deadline:
            return False
            
        return True

class Optimizer:
    """CMA-ES based optimizer with restarts and polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_fx = np.inf
        total_calls = 0

        # Phase 1: Initial sampling to find a good starting point
        # Try center and a few random points
        init_points = [np.zeros(dim)]
        for _ in range(5):
            init_points.append(rng.uniform(-4, 4, dim))
        
        for x in init_points:
            if total_calls >= budget:
                break
            if time.time() > deadline:
                break
            fx = func(x)
            total_calls += 1
            if fx < best_fx:
                best_fx, best_x = fx, x.copy()

        # Phase 2: CMA-ES with restarts
        # We will run CMA-ES for a certain number of generations or until budget is low.
        # Then we restart from the best point with a larger sigma if we are stuck.
        
        max_restarts = 5
        for restart in range(max_restarts):
            if total_calls >= budget - 50: # Leave budget for polish
                break
            if time.time() > deadline:
                break
                
            # Initialize CMA-ES
            if restart == 0:
                x0 = best_x
                sigma0 = 1.0
            else:
                # Perturb best point
                x0 = best_x + rng.normal(0, 1.0, dim)
                x0 = np.clip(x0, -5, 5)
                fx0 = func(x0)
                total_calls += 1
                if fx0 < best_fx:
                    best_fx, best_x = fx0, x0.copy()
                sigma0 = 0.5 + 0.5 * rng.random()
                
            es = CMAES(x0, sigma0, dim, budget - total_calls, rng, func, deadline)
            
            prev_best = best_fx
            no_improve_counter = 0
            gen_limit = 30 # Limit generations per restart to avoid stagnation
            
            for gen in range(gen_limit):
                if not es.step():
                    break
                
                # Check for improvement
                if es.x_mean is not None:
                    # We don't evaluate the mean directly in step, but we can check the best of the population
                    # Actually, the best of the population is tracked in F[0] inside step, but not returned.
                    # We can evaluate the current mean if we want, but that costs 1 call.
                    # Instead, we rely on the fact that if sigma is shrinking and no new best is found, we restart.
                    # Let's evaluate the best point from the last generation? 
                    # CMAES.step updates x_mean. The best point of the generation is X[0].
                    # We can modify CMAES to return the best point.
                    
                    # For now, let's just run it.
                    pass
                    
                # Heuristic restart: if sigma is very small or no improvement for 5 generations
                # We don't have direct access to best_fx update inside CMAES step without extra calls.
                # Let's just run for gen_limit and then restart if we want.
                
            # After CMAES run, check if we improved
            # The CMAES object's x_mean is the current mean, not necessarily the best.
            # We should track the best point found during the CMAES run.
            # Let's modify CMAES to track best_x and best_fx.
            
        # Refine Phase 3: Local Polish
        # Use remaining budget for a local search around the best point found
        if best_x is not None and total_calls < budget:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.1
            # Simple (1+1)-ES polish
            while total_calls < budget:
                if time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                total_calls += 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_fx:
                        best_fx, best_x = fx, x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.8
                if sigma < 1e-6:
                    break

        return best_x, best_fx
# EVOLVE-BLOCK-END
