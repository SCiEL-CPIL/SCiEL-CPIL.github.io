"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """MOEA/D style multi-objective optimizer for ZDT6."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        # Use 20 to get decent coverage. 200/20 = 10 generations.
        pop_size = 20
        if self.budget < pop_size:
            pop_size = self.budget
            if pop_size == 0:
                return np.array([])
                
        # Weight vectors for 2 objectives
        weights = np.linspace(0.0, 1.0, pop_size)
        W = np.column_stack([weights, 1 - weights])
        
        # Neighborhood size
        T = 5
        
        # Initialize population
        X = lo + (hi - lo) * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Ideal point
        z_star = np.min(F, axis=0)
        
        # Precompute neighbors based on weights
        neighbors = []
        for i in range(pop_size):
            dists = np.abs(weights - weights[i])
            idxs = np.argsort(dists)
            neighbors.append(idxs[:T])
            
        def is_dominated(f_a, f_b):
            return np.all(f_a <= f_b) and np.any(f_a < f_b)

        # Main loop
        while used < self.budget:
            n_update = min(pop_size, self.budget - used)
            
            for i in range(n_update):
                # Select neighbor j and k
                j = np.random.choice(neighbors[i])
                if i == j:
                    j = (i + 1) % pop_size
                k = np.random.choice(neighbors[i])
                if k == i:
                    k = (i + 1) % pop_size
                if k == j:
                    k = (i + 2) % pop_size
                    if i == k: k = (i + 3) % pop_size

                # DE Mutation
                F_diff = 0.5 + 0.5 * np.random.random()
                x_trial = X[i] + F_diff * (X[j] - X[k])
                
                # Mutation
                if np.random.random() < 1.0 / self.dim:
                    x_trial = x_trial + np.random.normal(0, 0.1, self.dim)
                
                x_trial = np.clip(x_trial, lo, hi)
                
                # Evaluate
                f_trial = problem(x_trial)
                used += 1
                
                # Update z_star
                z_star = np.minimum(z_star, f_trial)
                
                # Acceptance
                if is_dominated(f_trial, F[i]):
                    X[i] = x_trial
                    F[i] = f_trial
                else:
                    # Tchebycheff check
                    # Use current min/max for normalization
                    f_min_curr = np.min(F, axis=0)
                    f_max_curr = np.max(F, axis=0)
                    f_range_curr = f_max_curr - f_min_curr
                    f_range_curr[f_range_curr < 1e-10] = 1e-10
                    
                    f_i_norm = (F[i] - f_min_curr) / f_range_curr
                    f_t_norm = (f_trial - f_min_curr) / f_range_curr
                    
                    tcheb_i = np.max(W[i] * f_i_norm)
                    tcheb_t = np.max(W[i] * f_t_norm)
                    
                    if tcheb_t < tcheb_i:
                        X[i] = x_trial
                        F[i] = f_trial

        return X
# EVOLVE-BLOCK-END
