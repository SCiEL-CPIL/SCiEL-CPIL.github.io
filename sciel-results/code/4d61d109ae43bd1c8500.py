import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # DE Parameters
        pop_size = 10
        # F: Scaling factor, CR: Crossover rate
        # JADE style adaptation
        F = 0.5
        CR = 0.5
        
        # Initialize population
        # Use a mix of random and a few points near 0 to ensure coverage
        pop = np.random.uniform(-5, 5, (pop_size, n))
        # Ensure at least one point is near the center
        pop[0] = np.zeros(n)
        
        fits = np.array([func(x) for x in pop])
        evals_used = pop_size
        
        best_idx = np.argmin(fits)
        best_x = pop[best_idx].copy()
        best_f = fits[best_idx]
        
        # Main loop
        while evals_used < budget:
            # Generate offspring
            new_pop = np.zeros((pop_size, n))
            new_fits = np.zeros(pop_size)
            
            for i in range(pop_size):
                if evals_used >= budget:
                    break
                
                # Select three distinct random indices different from i
                # Standard DE/rand/1
                while True:
                    a = np.random.randint(pop_size)
                    b = np.random.randint(pop_size)
                    c = np.random.randint(pop_size)
                    if a != i and b != i and c != i and a != b and b != c and a != c:
                        break
                
                # Create trial vector
                # x_trial = x_a + F * (x_b - x_c)
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Crossover
                # Binary crossover
                r = np.random.rand()
                j_rand = np.random.randint(n)
                trial = np.zeros(n)
                for j in range(n):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                evals_used += 1
                
                # Selection
                if f_trial <= fits[i]:
                    new_pop[i] = trial
                    new_fits[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                else:
                    new_pop[i] = pop[i]
                    new_fits[i] = fits[i]
            
            # Update population
            pop = new_pop
            fits = new_fits
            
            # Optional: Adapt F and CR (JADE)
            # For simplicity, we keep them fixed or use a simple adaptation
            # Let's try a simple adaptation: if progress is slow, increase F
            # But for now, fixed F=0.5, CR=0.5 is robust.
            
        return best_x, best_f
