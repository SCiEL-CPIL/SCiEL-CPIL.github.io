import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        time_limit = 320.0
        
        # Island-based CMA-ES
        # We split the budget among a few "islands" (independent CMA-ES runs)
        # This helps escape local minima in multimodal functions like Rastrigin/Gallagher.
        
        num_islands = 3
        # Allocate budget roughly equally, ensuring at least 10 evals per island
        evals_per_island = max(10, budget // num_islands)
        
        best_x_global = None
        best_f_global = np.inf
        
        # Initial broad sampling to set a good starting point for islands
        # Use a small fraction of budget for initial global scan
        n_init = 5
        rng_global = np.random.default_rng(42)
        
        for _ in range(n_init):
            x = rng_global.uniform(-5, 5, dim)
            f = func(x)
            if f < best_f_global:
                best_f_global = f
                best_x_global = x.copy()
        
        calls_made = n_init
        
        # If we have very little budget, just do simple local search
        if budget - calls_made < 30:
             # Fallback to simple hill climbing
             x = best_x_global.copy()
             fx = best_f_global
             sigma = 0.5
             while calls_made < budget:
                 if time.time() - start_time > time_limit:
                     break
                 y = x + sigma * rng_global.standard_normal(dim)
                 y = np.clip(y, -5, 5)
                 fy = func(y)
                 calls_made += 1
                 if fy < fx:
                     x = y
                     fx = fy
                     if fy < best_f_global:
                         best_f_global = fy
                         best_x_global = y.copy()
                 else:
                     sigma *= 0.8
             return best_x_global, best_f_global

        # Define Island CMA-ES
        class CMAES:
            def __init__(self, x0, f0, seed):
                self.dim = dim
                self.x = x0.copy()
                self.f = f0
                self.sigma = 1.0
                self.C = np.eye(self.dim)
                self.pc = np.zeros(self.dim)
                self.cs = np.zeros(self.dim)
                self.rng = np.random.default_rng(seed)
                self.calls = 0
                self.last_improvement_call = 0
                self.best_f = f0
                self.best_x = x0.copy()
                
                # Parameters
                self.lambda_ = max(4, int(4 + 3 * np.log(self.dim))) # Population size
                self.mu = self.lambda_ // 2
                weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
                self.weights = weights / weights.sum()
                self.mu_eff = 1.0 / np.sum(self.weights**2)
                
                # Adaptation rates
                self.cc = (4 + self.mu_eff / self.dim) / (self.dim + 4 + 2 * self.mu_eff / self.dim)
                self.cs = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
                self.c1 = 2 / ((self.dim + 1.3)**2 + self.mu_eff)
                self.cmua = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.dim + 2)**2 + self.mu_eff))
                
                self.D = np.sqrt(self.C).mean() # Initial scale for sigma adaptation

            def step(self, func):
                if self.calls >= self.budget_allocated:
                    return self.best_x, self.best_f
                
                # Generate population
                Y = np.zeros((self.lambda_, self.dim))
                F = np.zeros(self.lambda_)
                
                for i in range(self.lambda_):
                    # Standard normal sampling
                    z = self.rng.standard_normal(self.dim)
                    # Transform using Cholesky of C
                    try:
                        L = np.linalg.cholesky(self.C)
                        y = self.x + self.sigma * (L @ z)
                    except np.linalg.LinAlgError:
                        # Fallback if C becomes non-positive definite
                        y = self.x + self.sigma * z
                        self.C = np.eye(self.dim)
                    
                    Y[i] = y
                    F[i] = func(y)
                    self.calls += 1
                    
                    if time.time() - start_time > time_limit:
                        return self.best_x, self.best_f

                # Sort by fitness
                idx = np.argsort(F)
                Y = Y[idx]
                F = F[idx]
                
                # Update best
                if F[0] < self.best_f:
                    self.best_f = F[0]
                    self.best_x = Y[0].copy()
                    self.last_improvement_call = self.calls

                # Update mean
                y_w = np.sum(self.weights[:, None] * Y[:self.mu], axis=0)
                self.x = y_w
                
                # Update covariance and sigma
                # Check for stagnation
                if self.calls - self.last_improvement_call > 15 * self.dim:
                    # Restart this island by jumping to a random point or best global
                    # For simplicity, we'll just increase sigma drastically
                    self.sigma *= 2.0
                    self.C = np.eye(self.dim)
                    self.pc = np.zeros(self.dim)
                    self.cs = np.zeros(self.dim)
                    # Optionally, jump towards global best
                    # self.x = self.x + 0.5 * (best_x_global - self.x)

                # Standard CMA-ES updates
                # Cumulation for sigma
                self.cs = (1 - self.cs) * self.cs + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * (
                    np.sqrt(self.c1) * np.sum(self.weights[:, None] * Y[:self.mu], axis=0) / self.sigma
                )
                # Actually, standard update:
                # pc = (1-cc)pc + hsig * cs * sqrt(mu_eff) * y_w / sigma
                # Let's use a simplified but robust update to avoid numerical issues in short runs
                
                # Simplified Covariance Update
                diff = Y[:self.mu] - self.x
                self.C = (1 - self.c1 - self.cmua) * self.C + self.c1 * (self.pc @ self.pc.T + (1 - self.cs**2) * np.eye(self.dim)) + self.cmua * np.sum(self.weights[:, None, None] * diff[:, :, None] * diff[:, None, :], axis=0)
                
                # Ensure C is positive definite
                self.C = (self.C + self.C.T) / 2
                try:
                    eigvals, eigvecs = np.linalg.eigh(self.C)
                    if np.min(eigvals) < 1e-10:
                        self.C += 1e-10 * np.eye(self.dim)
                except:
                    self.C = np.eye(self.dim)

                # Sigma update (simplified 1/5 rule equivalent for CMA)
                # Standard CMA uses cumulation, but for short runs, simple success rate works
                # Let's stick to the standard cumulation logic if possible, but it's complex.
                # Alternative: Just use (1+1)-ES logic inside the island? No, that defeats the purpose.
                # Let's use a simple heuristic for sigma:
                # If mean fitness of best mu improved over previous mean, increase sigma slightly?
                # Actually, let's just keep sigma constant or adapt via success rate of the best individual
                # compared to the previous best individual.
                
                # Better: Use the standard CMA-ES sigma update
                # hsig = np.linalg.norm(self.pc) / np.sqrt(1 - (1 - self.cs)**(2 * self.calls / (self.dim * self.cs))) < 1.4 + 2 / (self.dim - 1)
                # This is getting too complex for a short budget. 
                # Let's switch strategy: Use a **Differential Evolution** variant instead? 
                # Or just a robust (1+lambda)-ES?
                
                # Let's stick to a simpler, more robust population-based method: 
                # **Simplex Evolution Strategy** or **Nelder-Mead** hybrid?
                # Actually, CMA-ES is powerful but complex to implement correctly in 50 lines.
                # Let's implement a **Multi-start Nelder-Mead** with **Global Restart**.
                # This is structurally different from (1+1)-ES and robust.
                
                # Wait, I already wrote the CMA-ES. Let's fix the sigma update to be simple.
                # If the best individual in the new population is better than the previous best, 
                # we are making progress.
                if F[0] < self.f:
                    self.sigma *= 1.05
                else:
                    self.sigma *= 0.95
                self.sigma = np.clip(self.sigma, 0.01, 10.0)
                
                self.f = F[0]
                
                return self.best_x, self.best_f

        # Create Islands
        islands = []
        for i in range(num_islands):
            # Start each island near the global best found in init, with some jitter
            # Or start randomly? Random is better for multimodal.
            # Let's mix: 1 island starts at global best, others random.
            if i == 0 and best_x_global is not None:
                x0 = best_x_global + rng_global.normal(0, 0.5, dim)
                f0 = best_f_global
            else:
                x0 = rng_global.uniform(-5, 5, dim)
                f0 = func(x0)
                calls_made += 1
                if f0 < best_f_global:
                    best_f_global = f0
                    best_x_global = x0.copy()
            
            island = CMAES(x0, f0, seed=42 + i)
            island.budget_allocated = evals_per_island
            islands.append(island)

        # Main Loop: Round-robin among islands
        while calls_made < budget:
            if time.time() - start_time > time_limit:
                break
            
            # Select island with most remaining budget or random? 
            # Round robin is fair.
            for island in islands:
                if island.calls >= island.budget_allocated:
                    continue
                
                # Check if we should kill an island if it's far behind
                # For now, just run them all.
                
                x_island, f_island = island.step(func)
                # Update global best
                if f_island < best_f_global:
                    best_f_global = f_island
                    best_x_global = x_island.copy()
                
                calls_made += island.lambda_ # Approximate, step updates calls internally
                # Actually, step increments self.calls. We need to sync calls_made.
                # Let's just rely on island.calls and break if total calls > budget.
                
                # Check total budget
                total_calls = n_init + sum(isl.calls for isl in islands)
                if total_calls >= budget:
                    break
                
                if time.time() - start_time > time_limit:
                    break

        return best_x_global, best_f_global
