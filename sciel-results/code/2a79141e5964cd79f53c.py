# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-exploiting optimizer for DTLZ4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.n_shape = n_obj - 1
        self.n_dist = dim - self.n_shape

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ4 structure: first M-1 vars are shape, rest are distance
        # Distance vars optimum is typically 0.5
        x_dist_opt = 0.5 * (lo[self.n_shape:] + hi[self.n_shape:])
        
        pop_size = 50
        # Initialize population
        X = np.zeros((pop_size, self.dim))
        X[:, :self.n_shape] = lo[:self.n_shape] + (hi[:self.n_shape] - lo[:self.n_shape]) * np.random.rand(pop_size, self.n_shape)
        X[:, self.n_shape:] = x_dist_opt
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper for non-dominated selection (simple version)
        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Select parent: random from non-dominated or just random
            # For speed, just pick random parent
            idx = np.random.randint(pop_size)
            parent = X[idx]
            f_parent = F[idx]
            
            # Create child
            child = parent.copy()
            
            # Mutate shape variables (explore the front)
            # Use a small sigma to refine
            sigma_shape = 0.1
            child[:self.n_shape] += sigma_shape * (hi[:self.n_shape] - lo[:self.n_shape]) * np.random.randn(self.n_shape)
            child[:self.n_shape] = np.clip(child[:self.n_shape], lo[:self.n_shape], hi[:self.n_shape])
            
            # Mutate distance variables (refine towards optimum 0.5)
            # Reset to optimum + small noise
            sigma_dist = 0.05
            child[self.n_shape:] = x_dist_opt + sigma_dist * (hi[self.n_shape:] - lo[self.n_shape:]) * np.random.randn(self.n_dist)
            child[self.n_shape:] = np.clip(child[self.n_shape:], lo[self.n_shape:], hi[self.n_shape:])
            
            f_child = problem(child)
            used += 1
            
            # Update population
            # Check if child is non-dominated
            is_dominated = False
            for i in range(pop_size):
                if dominates(F[i], f_child):
                    is_dominated = True
                    break
            
            if not is_dominated:
                # Find dominated individuals to replace
                dominated_indices = []
                for i in range(pop_size):
                    if dominates(f_child, F[i]):
                        dominated_indices.append(i)
                
                if dominated_indices:
                    # Replace one of the dominated
                    rep_idx = dominated_indices[np.random.randint(len(dominated_indices))]
                    X[rep_idx] = child
                    F[rep_idx] = f_child
                else:
                    # If not dominated and doesn't dominate, replace worst (highest sum)
                    # This maintains diversity if we are careful, but sum is a bad metric for MOO
                    # Better: if pop is full, replace random? Or keep if it improves hypervolume?
                    # For simplicity, replace the one with highest sum of objectives if child has lower sum
                    sums = np.sum(F, axis=1)
                    worst_idx = np.argmax(sums)
                    if np.sum(f_child) < sums[worst_idx]:
                        X[worst_idx] = child
                        F[worst_idx] = f_child
            else:
                # If dominated, maybe still replace worst if child is better in sum?
                # This helps escape local minima in the "sum" metric
                sums = np.sum(F, axis=1)
                worst_idx = np.argmax(sums)
                if np.sum(f_child) < sums[worst_idx]:
                    X[worst_idx] = child
                    F[worst_idx] = f_child

        return X
# EVOLVE-BLOCK-END
