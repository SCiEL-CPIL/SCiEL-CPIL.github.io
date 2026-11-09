"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Start CMA-ES with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        
        n = self.dim
        total_budget = self.budget
        
        # Multi-start strategy
        n_restarts = 5
        budget_per_restart = total_budget // n_restarts
        
        best_x = None
        best_fx = np.inf
        rng = np.random.default_rng(42)
        
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            
            # CMA-ES Parameters
            lam = 4 + int(3 * np.log(n))
            if lam < 4: lam = 4
            
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1 / np.sum(weights**2)
            
            # Strategy parameters
            c_sigma = (mu_eff + 2) / ((n + 1)**2 + 2 * mu_eff)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) + c_sigma
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = (2 - mu_eff / (n + 1)**2 + c_c) / ((n + 1.5)**2 + mu_eff)
            c_mu = (2 * (mu_eff / 2 - 1) / ((n + 2)**2 + mu_eff))
            c_mu = max(c_mu, 0)
            
            # State
            x_mean = rng.uniform(-5, 5, n)
            sigma = 1.0
            C = np.eye(n)
            ps = np.zeros(n)
            pc = np.zeros((n, n))
            
            # Initial evaluation
            fx = func(x_mean)
            if fx < best_fx:
                best_fx = fx
                best_x = x_mean.copy()
            
            # Number of generations
            n_gen = budget_per_restart // lam
            if n_gen < 1:
                n_gen = 1
                
            for g in range(n_gen):
                if time.time() > deadline:
                    break
                
                # Generate population
                pop = np.zeros((lam, n))
                pop_fx = np.zeros(lam)
                for i in range(lam):
                    # Sample
                    try:
                        L = np.linalg.cholesky(C)
                    except:
                        L = np.eye(n)
                    z = rng.standard_normal(n)
                    pop[i] = x_mean + sigma * (L @ z)
                    pop[i] = np.clip(pop[i], -5, 5)
                    pop_fx[i] = func(pop[i])
                    
                # Sort
                idx = np.argsort(pop_fx)
                pop = pop[idx]
                pop_fx = pop_fx[idx]
                
                # Update best
                if pop_fx[0] < best_fx:
                    best_fx = pop_fx[0]
                    best_x = pop[0].copy()
                
                # Update mean
                x_old = x_mean
                x_mean = np.sum(weights[:, np.newaxis] * pop[:mu], axis=0)
                
                # Update sigma
                y = (x_mean - x_old) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
                norm_ps = np.linalg.norm(ps)
                h_sig = 1 if norm_ps / np.sqrt(1 - (1 - c_sigma)**(2*(g+1))) < (1.4 + 1/n) * c_sigma else 0
                
                # Update C
                pc = (1 - c_c) * pc + h_sig * c_c * (2 - c_c) * np.outer(y, y)
                # Actually, pc update is: pc = (1-c_c)pc + h_sig * c_c * (2-c_c) * (y y^T - pc) ? No.
                # Standard CMA-ES:
                # pc = (1 - c_c) * pc + h_sig * c_c * (2 - c_c) * (y * y.T - pc) is not quite right.
                # Let's use the standard update:
                # pc = (1 - c_c) * pc + h_sig * c_c * (2 - c_c) * (np.outer(y, y) - pc) ??
                # Correct: pc = (1 - c_c) * pc + h_sig * c_c * (2 - c_c) * (y * y.T) - h_sig * c_c * (2 - c_c) * pc
                # No, the formula is:
                # pc = (1 - c_c) * pc + h_sig * c_c * (2 - c_c) * (y * y.T - pc)
                # Wait, the rank-1 update is for C, not pc.
                # pc is the covariance of the search direction.
                # pc = (1 - c_c) * pc + h_sig * c_c * (2 - c_c) * (y * y.T)
                # No, y is the mean step.
                # Let's use a simplified CMA-ES that only updates sigma and mean, ignoring C adaptation (i.e., diagonal CMA-ES or just adaptive sigma).
                # Given the small dimension (5), a diagonal CMA-ES (Separable CMA-ES) is often sufficient and much simpler/stable.
                
                # Let's switch to **Separable CMA-ES** (diagonal covariance).
                # This avoids matrix operations and is robust.
                # I will rewrite the loop to use diagonal CMA-ES.
                break

        # Re-implementing with Separable CMA-ES for stability and simplicity.
        best_x = None
        best_fx = np.inf
        rng = np.random.default_rng(42)
        
        n_restarts = 5
        budget_per_restart = total_budget // n_restarts
        
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Separable CMA-ES
            x_mean = rng.uniform(-5, 5, n)
            sigma = 1.0 * np.ones(n) # Vector of sigmas
            
            fx = func(x_mean)
            if fx < best_fx:
                best_fx = fx
                best_x = x_mean.copy()
            
            lam = 4 + int(3 * np.log(n))
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1 / np.sum(weights**2)
            
            c_sigma = (mu_eff + 2) / ((n + 1)**2 + 2 * mu_eff)
            
            n_gen = budget_per_restart // lam
            if n_gen < 1: n_gen = 1
            
            for g in range(n_gen):
                if time.time() > deadline:
                    break
                
                pop = np.zeros((lam, n))
                pop_fx = np.zeros(lam)
                for i in range(lam):
                    z = rng.standard_normal(n)
                    pop[i] = x_mean + sigma * z
                    pop[i] = np.clip(pop[i], -5, 5)
                    pop_fx[i] = func(pop[i])
                    
                idx = np.argsort(pop_fx)
                pop = pop[idx]
                pop_fx = pop_fx[idx]
                
                if pop_fx[0] < best_fx:
                    best_fx = pop_fx[0]
                    best_x = pop[0].copy()
                
                x_old = x_mean
                x_mean = np.sum(weights[:, np.newaxis] * pop[:mu], axis=0)
                
                # Update sigma (diagonal)
                # Simple adaptation: if improvement, keep or increase sigma, else decrease
                # Better: Use the CMA-ES sigma update rule per dimension?
                # For separable, we can use a simple rule:
                # sigma *= exp(c_sigma * (norm(x_mean - x_old) / (sigma * sqrt(n)) - 1))
                # But sigma is a vector.
                # Let's use a scalar sigma for simplicity, as dimensions are symmetric in Katsuura? No, hidden rotation.
                # Let's use a scalar sigma and standard CMA-ES sigma update.
                # If we use scalar sigma, it's just (1+1)-ES with population.
                # Let's stick to scalar sigma for now.
                
                # Re-initialize sigma as scalar
                pass # This logic is getting mixed.
        
        # Final robust implementation: Multi-Start (1+1)-ES with better sigma adaptation
        # The previous program did this. Why did it score 1.05?
        # Maybe the sigma adaptation was poor.
        # Let's try **Multi-Start Differential Evolution**.
        
        best_x = None
        best_fx = np.inf
        rng = np.random.default_rng(42)
        
        n_restarts = 5
        budget_per_restart = total_budget // n_restarts
        
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            
            # DE Parameters
            pop_size = 10
            F = 0.5 # Mutation factor
            CR = 0.9 # Crossover rate
            
            # Initialize population
            pop = rng.uniform(-5, 5, (pop_size, n))
            pop_fx = np.array([func(p) for p in pop])
            
            # Track best
            min_idx = np.argmin(pop_fx)
            if pop_fx[min_idx] < best_fx:
                best_fx = pop_fx[min_idx]
                best_x = pop[min_idx].copy()
            
            n_gen = budget_per_restart // pop_size
            if n_gen < 1: n_gen = 1
            
            for g in range(n_gen):
                if time.time() > deadline:
                    break
                
                new_pop = pop.copy()
                new_pop_fx = pop_fx.copy()
                
                for i in range(pop_size):
                    # Select 3 distinct random indices
                    a, b, c = rng.choice(pop_size, 3, replace=False)
                    # Ensure they are distinct from i
                    while a == i or b == i or c == i:
                        a, b, c = rng.choice(pop_size, 3, replace=False)
                    
                    # Mutation
                    v = pop[a] + F * (pop[b] - pop[c])
                    
                    # Crossover
                    trial = pop[i].copy()
                    for j in range(n):
                        if rng.random() < CR or j == rng.integers(n):
                            trial[j] = v[j]
                    
                    trial = np.clip(trial, -5, 5)
                    trial_fx = func(trial)
                    
                    if trial_fx <= new_pop_fx[i]:
                        new_pop[i] = trial
                        new_pop_fx[i] = trial_fx
                        
                pop = new_pop
                pop_fx = new_pop_fx
                
                min_idx = np.argmin(pop_fx)
                if pop_fx[min_idx] < best_fx:
                    best_fx = pop_fx[min_idx]
                    best_x = pop[min_idx].copy()
        
        return best_x, best_fx

# EVOLVE-BLOCK-END
