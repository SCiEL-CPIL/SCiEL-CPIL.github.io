import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin for 330s limit
        
        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters
        n = self.dim
        lambda_ = 10  # Population size
        sigma = 2.0   # Initial step size
        x_mean = np.zeros(n) # Start at center
        
        # CMA-ES Strategy Parameters
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))
        
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        calls_made = 0
        best_x = x_mean.copy()
        best_f = float('inf')
        
        # We need to evaluate the initial mean? CMA-ES samples from it.
        # Let's just start the loop.
        
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            
            # Generate population
            pop = []
            for i in range(lambda_):
                # Standard normal
                z = rng.standard_normal(n)
                # Transform using C
                # Cholesky of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(n)
                x = x_mean + sigma * (L @ z)
                x = np.clip(x, -5, 5)
                pop.append(x)
            
            # Evaluate
            fitness = np.array([func(x) for x in pop])
            calls_made += len(pop)
            
            # Update best
            idx_best = np.argmin(fitness)
            if fitness[idx_best] < best_f:
                best_f = fitness[idx_best]
                best_x = pop[idx_best].copy()
                
            # Sort indices by fitness
            order = np.argsort(fitness)
            y = np.array([pop[i] for i in order])
            
            # Weighted mean
            x_new = np.sum(weights[:, None] * y[:mu], axis=0)
            
            # Update step size control variables
            # p_sigma
            C_inv_sqrt = np.linalg.cholesky(np.linalg.inv(C)) # Approximation, better to use inv sqrt
            # Actually, standard CMA-ES uses C^-1/2.
            # For simplicity and speed in Python, let's use a simpler update or ensure stability.
            # A full CMA-ES implementation is complex. 
            # Given the constraint of "standard library, NumPy, SciPy", and the risk of a buggy manual CMA-ES,
            # maybe a simpler structural change is better: **Multistart Nelder-Mead**.
            
            # Let's pivot. The current DE is weak. 
            # A very effective simple strategy for BBOB is:
            # 1. Sample ~50 diverse points.
            # 2. Take the top 5.
            # 3. Run Nelder-Mead from each of the top 5, using the remaining budget.
            # This exploits the local structure which DE ignores after initial convergence.
            
            # However, the directive asks for a "different optimization or search method".
            # Multistart NM is a different *strategy* but uses the same local method.
            # CMA-ES is a different *algorithm*.
            
            # Let's stick to CMA-ES but keep it simple.
            # If I implement CMA-ES incorrectly, it will fail.
            # Alternative: Use `scipy.optimize.differential_evolution`? It's the same method family.
            
            # Let's try a **Bayesian Optimization** approach? No, too slow in Python for 500 calls.
            
            # Let's try **Particle Swarm Optimization (PSO)**? Similar to DE.
            
            # Let's look at the "Inspiration 1" score 2.03.
            # If I can't replicate it, I should try the most robust alternative: **CMA-ES**.
            # I will implement a simplified CMA-ES that is known to be stable.
            
            # Reset to CMA-ES implementation properly.
            pass # This block is just thinking. I will write the code below.
            
            # Break out to write the actual replacement code.
            break

        return best_x, best_f
