"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        DEADLINE = 310
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget
        
        n_eval = 0
        all_points = []
        
        # Population size for evolutionary search
        pop_size = min(20, max(10, budget // 10))
        
        # Initialize population with random points
        X = np.random.uniform(lo, hi, size=(pop_size, dim))
        F = np.array([problem(x) for x in X])
        n_eval = pop_size
        for i in range(pop_size):
            all_points.append(F[i].copy())
        
        # Evolutionary loop: mutate non-dominated parents
        while n_eval < budget:
            if time.time() - start_time > DEADLINE:
                break
            
            # Find non-dominated solutions
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            
            # Get non-dominated parents
            non_dom_idx = np.where(~dominated)[0]
            if len(non_dom_idx) == 0:
                non_dom_idx = np.array([0])
            
            # Generate a batch of children by mutating non-dominated parents
            batch_size = min(10, budget - n_eval)
            for _ in range(batch_size):
                if n_eval >= budget:
                    break
                if time.time() - start_time > DEADLINE:
                    break
                
                # Select a parent from non-dominated set
                parent_idx = non_dom_idx[np.random.randint(len(non_dom_idx))]
                
                # Mutate: Gaussian perturbation
                mutation_scale = 0.1 * (hi - lo)
                child = X[parent_idx].copy() + np.random.standard_normal(dim) * mutation_scale
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                n_eval += 1
                all_points.append(fc.copy())
                
                # Replace worst solution in population if child is better
                # Use sum of objectives as a simple quality metric
                worst_idx = np.argmax(np.sum(F, axis=1))
                if np.sum(fc) < np.sum(F[worst_idx]):
                    X[worst_idx] = child
                    F[worst_idx] = fc
                    # Recompute non-dominated set
                    dominated = np.zeros(n, dtype=bool)
                    for i in range(n):
                        for j in range(n):
                            if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                                dominated[i] = True
                                break
                    non_dom_idx = np.where(~dominated)[0]
                    if len(non_dom_idx) == 0:
                        non_dom_idx = np.array([0])
        
        return all_points
# EVOLVE-BLOCK-END
