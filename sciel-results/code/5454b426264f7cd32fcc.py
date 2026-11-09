import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # Differential Evolution parameters
        pop_size = max(10, 5 * n)
        if pop_size > budget:
            pop_size = max(2, budget)
            
        F = 0.8  # Scaling factor
        CR = 0.5  # Crossover rate
        
        # Initialize population uniformly in [-5, 5]
        P = np.random.uniform(-5, 5, (pop_size, n))
        F_P = np.array([func(x) for x in P])
        calls_made = pop_size
        
        best_idx = np.argmin(F_P)
        best_x = P[best_idx].copy()
        best_f = F_P[best_idx]
        
        # Main DE loop
        # Each generation costs pop_size evaluations
        # We can do multiple generations
        while calls_made + pop_size <= budget:
            # Create trial vectors
            for i in range(pop_size):
                # Select 3 distinct random indices different from i
                indices = [j for j in range(pop_size) if j != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                # Mutation: v = P[a] + F * (P[b] - P[c])
                v = P[a] + F * (P[b] - P[c])
                
                # Crossover
                # Binomial crossover
                r = np.random.rand()
                jrand = np.random.randint(n)
                trial = P[i].copy()
                for j in range(n):
                    if r <= CR or j == jrand:
                        trial[j] = v[j]
                        
                # Clip to box
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = func(trial)
                calls_made += 1
                
                # Selection
                if f_trial < F_P[i]:
                    P[i] = trial
                    F_P[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        
        # Use remaining budget for local search
        remaining = budget - calls_made
        if remaining > 0 and best_x is not None:
            # Simple local search: random perturbations with decreasing step
            current_x = best_x.copy()
            current_f = best_f
            step = 1.0
            for _ in range(remaining):
                x = current_x + np.random.normal(0, step, n)
                x = np.clip(x, -5, 5)
                f = func(x)
                if f < current_f:
                    current_f = f
                    current_x = x.copy()
                    # Reduce step slightly
                    step = max(step * 0.9, 1e-5)
                else:
                    # Reduce step more aggressively on failure
                    step = max(step * 0.8, 1e-5)
            
            if current_f < best_f:
                best_f = current_f
                best_x = current_x.copy()
                
        return best_x, best_f
