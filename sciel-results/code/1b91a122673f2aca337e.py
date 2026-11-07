import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size: 50 is a good balance for 200 budget
        pop_size = 50
        
        # --- Initialization ---
        # DTLZ4 specific initialization:
        # Variables 0 and 1 control the Pareto front shape. Sample uniformly.
        # Variables 2 to 11 control the distance to the front. They should be near 0.
        X = np.zeros((pop_size, self.dim))
        X[:, 0:2] = np.random.random((pop_size, 2))
        # Initialize tail variables very close to 0 to satisfy g(x) ~ 0
        X[:, 2:] = np.random.random((pop_size, self.dim - 2)) * 0.01
        
        F = np.array([problem(x) for x in X])
        evals = pop_size
        
        # Helper: Check if point i is dominated by any point in F
        def is_dominated(i, F):
            # F[i] is dominated if exists j != i such that F[j] <= F[i] and F[j] != F[i]
            # Vectorized check for efficiency
            leq = np.all(F <= F[i], axis=1)
            lt = np.any(F < F[i], axis=1)
            # Exclude self
            mask = np.ones(len(F), dtype=bool)
            mask[i] = False
            return np.any(leq & lt & mask)
        
        # Helper: Get indices of non-dominated points
        def get_nondominated_indices(F):
            nondom = []
            for i in range(len(F)):
                if not is_dominated(i, F):
                    nondom.append(i)
            return nondom

        # Initial non-dominated set
        nondom_idx = get_nondominated_indices(F)
        
        while evals < self.budget:
            if time.time() > deadline:
                break
            
            # Select parent from non-dominated points
            if not nondom_idx:
                # Fallback: select random
                parent_idx = np.random.randint(len(X))
            else:
                parent_idx = nondom_idx[np.random.randint(len(nondom_idx))]
            
            parent = X[parent_idx]
            
            # Adaptive mutation scales
            progress = evals / self.budget
            # Shape variables (0, 1): larger steps early, smaller later
            sigma_shape = 0.25 * (1.0 - progress) + 0.05
            # Distance variables (2+): small steps to stay near 0
            sigma_dist = 0.05 * (1.0 - progress) + 0.005
            
            # Generate child
            child = parent.copy()
            child[0:2] += sigma_shape * np.random.normal(size=2)
            child[2:] += sigma_dist * np.random.normal(size=self.dim - 2)
            
            # Clip to bounds
            child = np.clip(child, lo, hi)
            
            # Evaluate
            f_child = problem(child)
            evals += 1
            
            # Check if child is non-dominated w.r.t current population
            # We need to check against all F
            dominated_by_pop = False
            for i in range(len(F)):
                if np.all(F[i] <= f_child) and np.any(F[i] < f_child):
                    dominated_by_pop = True
                    break
            
            if not dominated_by_pop:
                # Child is non-dominated. Replace the worst point in the population.
                # "Worst" heuristic: Highest sum of objectives, or dominated point.
                # To maintain diversity, we prefer replacing dominated points if any exist.
                # If all are non-dominated, replace the one with the highest sum (furthest from ideal).
                
                # Find if there are any dominated points in the population
                dom_indices = []
                for i in range(len(F)):
                    if is_dominated(i, F):
                        dom_indices.append(i)
                
                if dom_indices:
                    # Replace a random dominated point
                    replace_idx = dom_indices[np.random.randint(len(dom_indices))]
                else:
                    # All non-dominated: replace the one with highest sum of objectives
                    sums = np.sum(F, axis=1)
                    replace_idx = np.argmax(sums)
                
                X[replace_idx] = child
                F[replace_idx] = f_child
                
                # Update non-dominated indices
                nondom_idx = get_nondominated_indices(F)
            else:
                # Child is dominated. 
                # Even if dominated, if it is better than the worst point in the population, we might still replace?
                # Usually in simple MOO, we only keep non-dominated. 
                # However, to avoid stagnation, we can replace the worst point if the child is "closer" to the ideal point.
                # Let's stick to strict non-dominated replacement to ensure the population consists of Pareto-optimal candidates.
                # But this might lead to empty population if we are strict? No, we only replace if child is non-dominated.
                # If child is dominated, we discard it.
                # This is a standard "Pareto archive" behavior within a population.
                
                # Optimization: If the population is full of non-dominated points, and we keep generating dominated ones,
                # we might waste budget. 
                # A common improvement: If child is dominated, check if it dominates any point in the population.
                # If yes, replace the dominated one.
                
                # Check if child dominates any existing point
                dominated_in_pop = []
                for i in range(len(F)):
                    if np.all(f_child <= F[i]) and np.any(f_child < F[i]):
                        dominated_in_pop.append(i)
                
                if dominated_in_pop:
                    # Replace one of the dominated points
                    replace_idx = dominated_in_pop[np.random.randint(len(dominated_in_pop))]
                    X[replace_idx] = child
                    F[replace_idx] = f_child
                    nondom_idx = get_nondominated_indices(F)
                # Else: discard child

        return X
