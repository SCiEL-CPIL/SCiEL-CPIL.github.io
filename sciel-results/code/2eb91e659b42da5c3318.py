"""Multi-objective optimizer for ZDT4 using structure-aware initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for ZDT4 that leverages knowledge of the problem structure.
    ZDT4 has a deceptive landscape where x_i=0 is a local optimum for the 
    auxiliary function G(x), but the global minima are at x_i = k + 0.5.
    We initialize x_i near 0.5 to target the global Pareto front directly.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        pop_size = min(20, self.budget // 5)
        if pop_size < 2:
            pop_size = 2
            
        # Initialize population
        # x_0 is the first objective, uniform in [0, 1] (or bounds)
        # x_1...x_9 are the deceptive variables. 
        # Global minima of h(x) = x^2 - 10cos(4pi x) are at x = k + 0.5.
        # Within [-5, 5], x=0.5 and x=-0.5 are the best.
        # We initialize x_i near 0.5 to escape the x_i=0 trap.
        
        X = np.zeros((pop_size, self.dim))
        # x_0: uniform in bounds[0]
        X[:, 0] = lo[0] + (hi[0] - lo[0]) * np.random.rand(pop_size)
        
        # x_1...x_{n-1}: near 0.5
        # Add small noise to avoid exact duplicates and allow local search
        for i in range(1, self.dim):
            # Use 0.5 as the center for the global minimum
            # Clip to bounds just in case, though 0.5 is usually inside [-5, 5]
            center = 0.5
            noise = 0.01 * np.random.randn(pop_size)
            X[:, i] = np.clip(center + noise, lo[i], hi[i])
            
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Local search / Evolutionary step
        # We want to spread x_0 across [0, 1] and keep x_i close to 0.5
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Select a parent (randomly from non-dominated or best)
            # For simplicity and speed, we can just pick a random index
            # and mutate it.
            idx = np.random.randint(pop_size)
            parent = X[idx].copy()
            
            # Mutation
            # x_0: larger step to explore the front
            parent[0] += 0.05 * np.random.randn()
            parent[0] = np.clip(parent[0], lo[0], hi[0])
            
            # x_i: very small step to stay in the basin of x_i=0.5
            for i in range(1, self.dim):
                parent[i] += 0.001 * np.random.randn()
                parent[i] = np.clip(parent[i], lo[i], hi[i])
                
            fc = problem(parent)
            used += 1
            
            # Replacement strategy
            # Replace the individual that is dominated by the new one, 
            # or the worst in terms of sum of objectives if no dominance.
            # Simple: replace the one with the largest sum of objectives
            # among those dominated by the new point, or just the worst overall.
            
            # Check if new point dominates any existing point
            replaced = False
            for j in range(pop_size):
                # Check if parent (new) dominates X[j]
                if np.all(fc <= F[j]) and np.any(fc < F[j]):
                    X[j] = parent
                    F[j] = fc
                    replaced = True
                    break
            
            if not replaced:
                # Replace the worst individual (highest sum of objectives)
                worst_idx = np.argmax(np.sum(F, axis=1))
                X[worst_idx] = parent
                F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END
