import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 330.0
        safety_margin = 10.0

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Standard initialization
        x0 = np.zeros(dim)
        sigma = 1.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Update rates
        mu = dim // 2 + 1
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        c_sigma = (2 + mueff / (2 * dim)) / (dim + 2 + mueff)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / dim) / ((dim + 2)**2 + mueff))
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        # Population size
        lam = 2 * mu
        
        # Initial population
        best_x = x0.copy()
        best_f = func(best_x)
        calls_made = 1
        
        # Pre-generate random numbers for speed if possible, or just use rng
        rng = np.random.default_rng(42)

        while calls_made < budget:
            # Check time
            if time.time() - start_time > timeout - safety_margin:
                break

            # Generate offspring
            Y = []
            F = []
            
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C becomes singular, reset C
                C = np.eye(dim)
                L = np.eye(dim)
                
            for i in range(lam):
                zi = rng.standard_normal(dim)
                xi = x0 + sigma * (L @ zi)
                # Clip to box
                xi = np.clip(xi, -5, 5)
                fi = func(xi)
                Y.append(xi)
                F.append(fi)
                calls_made += 1
                
                if time.time() - start_time > timeout - safety_margin:
                    # If we run out of time in the middle of a generation, 
                    # we can't complete the update logic safely without breaking state.
                    # However, the task says "exception later keeps the calls made".
                    # We should just break out and return current best.
                    return best_x, best_f

            # Sort offspring
            idx = np.argsort(F)
            Y = [Y[i] for i in idx]
            F = [F[i] for i in idx]
            
            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = Y[0]

            # Update mean
            x0_new = x0.copy()
            for i in range(mu):
                x0_new += weights[i] * Y[i]
                
            # Update path evolution strategies
            # pc = (1 - c_c) * pc + c_c * sqrt(mueff) * (x0_new - x0) / sigma
            x0_old = x0
            x0 = x0_new
            
            # Note: The standard CMA-ES formula for pc uses the difference of means
            # pc = (1 - c_c) * pc + c_c * sqrt(mueff) * (x_new - x_old) / sigma
            # But since we clipped x, the effective step is different. 
            # For robustness in a black-box with clipping, we might skip strict pc/ps 
            # or apply them heuristically. Given the short budget and dim=5, 
            # a simplified adaptive step size or even just adapting sigma based on success rate 
            # might be more robust than full CMA-ES if clipping distorts the distribution.
            
            # Let's stick to a robust (1+1)-ES variant with better step size control 
            # and random restarts if stuck, as full CMA-ES can be sensitive to 
            # boundary clipping in 5D with 500 budget. 
            # Actually, the prompt asks for a DIFFERENT strategy. 
            # The current is (1+1)-ES. 
            # Let's try a simple Random Search with Adaptive Step Size (like (1+1)-ES but 
            # with a more aggressive exploration phase) or a Multi-Start Nelder-Mead?
            
            # Re-evaluating: The current score is 1.0. 
            # Bent Cigar is ill-conditioned. (1+1)-ES with 1/5 rule is standard.
            # Maybe the issue is that (1+1)-ES gets stuck in local minima or 
            # doesn't explore the long axis well enough early on.
            
            # Let's implement a simple Covariance Matrix Adaptation (CMA) but 
            # simplified to avoid numerical issues with clipping.
            # Or, better yet, use a **Differential Evolution** (DE) strategy? 
            # DE is often very robust for black-box optimization.
            
            # Let's pivot to **Differential Evolution (DE)** with JADE or simple parameters.
            # DE uses population differences to generate new candidates.
            # It handles ill-conditioning well if F (scale factor) is adapted.

            # Resetting the loop to use DE logic instead of CMA-ES to keep it simple and robust.
            # I will restart the __call__ logic mentally to DE.
            
            # Wait, I'm already inside the loop. I should have planned this before.
            # Let's scrap the CMA-ES implementation in the loop and use a different 
            # approach for the final code.
            
            # Since I cannot break out of the thinking process and restart the code 
            # block easily, I will write the code using **Differential Evolution**.
            
            # DE Parameters:
            # NP = 10 (population size)
            # F = 0.5 (scale factor)
            # CR = 0.9 (crossover rate)
            # Strategy: DE/rand/1/bin
            
            # This requires a complete rewrite of the __call__ method.
            # I will output the complete program with DE.

            break # Break out of CMA-ES loop, we will replace the whole method

        return best_x, best_f

# The above draft was messy. I will now provide the clean, complete program using Differential Evolution.

import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 330.0
        safety_margin = 5.0
        
        dim = self.dim
        budget = self.budget
        
        # Differential Evolution Parameters
        # For dim=5, a population size of 10-20 is reasonable.
        # Budget is 500. 500 / 15 = ~33 generations.
        NP = 15 
        F = 0.8  # Scale factor
        CR = 0.9 # Crossover rate
        
        rng = np.random.default_rng(123)
        
        # Initialize population uniformly in [-5, 5]
        pop = rng.uniform(-5, 5, (NP, dim))
        f_pop = np.array([func(x) for x in pop])
        calls_made = NP
        
        best_idx = np.argmin(f_pop)
        best_x = pop[best_idx].copy()
        best_f = f_pop[best_idx]
        
        # Main DE Loop
        while calls_made + NP <= budget:
            # Time check
            if time.time() - start_time > timeout - safety_margin:
                break
            
            new_pop = []
            new_f = []
            
            for i in range(NP):
                # Select 3 distinct random individuals
                r1, r2, r3 = rng.choice(NP, 3, replace=False)
                
                # Differential mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Binomial Crossover
                trial = np.zeros(dim)
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                        
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls_made += 1
                
                # Selection
                if f_trial <= f_pop[i]:
                    new_pop.append(trial)
                    new_f.append(f_trial)
                else:
                    new_pop.append(pop[i])
                    new_f.append(f_pop[i])
                    
            pop = np.array(new_pop)
            f_pop = np.array(new_f)
            
            # Update global best
            best_idx = np.argmin(f_pop)
            if f_pop[best_idx] < best_f:
                best_f = f_pop[best_idx]
                best_x = pop[best_idx].copy()

        return best_x, best_f
