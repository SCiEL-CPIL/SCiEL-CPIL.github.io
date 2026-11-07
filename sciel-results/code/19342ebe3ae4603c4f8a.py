"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Differential Evolution implementation for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.call_count = 0

    def _make_func(self, func):
        """Wrap func to count calls."""
        def wrapped(x):
            self.call_count += 1
            return float(func(x))
        return wrapped

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        self.call_count = 0
        
        # Bounds
        lb = -5.0
        ub = 5.0
        
        # Create a counting wrapper
        wrapped_func = self._make_func(func)
        
        # DE Parameters
        pop_size = 10
        F = 0.8
        CR = 0.9
        
        rng = np.random.default_rng(42)
        
        # Initialize population
        pop = rng.uniform(lb, ub, (pop_size, dim))
        fits = np.array([wrapped_func(x) for x in pop])
        
        best_x = pop[np.argmin(fits)].copy()
        best_f = np.min(fits)
        
        # Number of generations
        # We have budget - pop_size calls left.
        # Each generation costs pop_size calls.
        num_gens = (budget - pop_size) // pop_size
        
        for gen in range(num_gens):
            if self.call_count >= budget:
                break
            
            new_pop = []
            new_fits = []
            
            for i in range(pop_size):
                # Select three distinct random individuals
                while True:
                    a = rng.integers(0, pop_size)
                    b = rng.integers(0, pop_size)
                    c = rng.integers(0, pop_size)
                    if a != b and b != c and a != c and a != i and b != i and c != i:
                        break
                
                # Mutation
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Crossover
                j_rand = rng.integers(0, dim)
                trial = np.zeros(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                
                # Clip to bounds
                trial = np.clip(trial, lb, ub)
                
                # Evaluate
                f_trial = wrapped_func(trial)
                
                # Selection
                if f_trial <= fits[i]:
                    new_pop.append(trial)
                    new_fits.append(f_trial)
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                else:
                    new_pop.append(pop[i])
                    new_fits.append(fits[i])
            
            pop = np.array(new_pop)
            fits = np.array(new_fits)
            
        return best_x, best_f
# EVOLVE-BLOCK-END
