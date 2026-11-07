"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer using an NSGA-II-inspired evolutionary strategy.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        if self.budget <= 0:
            return None

        # Determine population size and number of generations
        # We want to use the full budget.
        # Let's aim for a population size of 50 if possible, allowing for 4 generations.
        # If budget is smaller, adjust.
        if self.budget < 50:
            pop_size = self.budget
            n_gens = 1
        else:
            pop_size = 50
            n_gens = self.budget // pop_size
            if n_gens < 1:
                n_gens = 1
                pop_size = self.budget

        # Initialize Population
        population = np.random.rand(pop_size, self.dim) * span + lo
        
        # Store all evaluated points and their fitness
        all_X = []
        all_F = []
        
        def evaluate(X):
            nonlocal all_X, all_F
            F = np.zeros((X.shape[0], self.n_obj))
            for i in range(X.shape[0]):
                if time.time() > deadline:
                    break
                F[i] = problem(X[i])
                all_X.append(X[i])
                all_F.append(F[i])
            return F

        # Evaluate initial population
        F = evaluate(population)
        
        # Evolutionary Loop
        for gen in range(1, n_gens):
            if time.time() > deadline:
                break
            
            # Generate Offspring
            # 50% Random Restart (exploration)
            # 50% Mutation of current population (exploitation)
            offspring = np.zeros((pop_size, self.dim))
            
            n_random = pop_size // 2
            offspring[:n_random] = np.random.rand(n_random, self.dim) * span + lo
            
            # Mutate a random subset of the current population
            parent_indices = np.random.choice(pop_size, size=(pop_size - n_random), replace=True)
            # Scale noise to 10% of the span
            noise = np.random.normal(0, 0.1 * span, size=(pop_size - n_random, self.dim))
            offspring[n_random:] = population[parent_indices] + noise
            offspring = np.clip(offspring, lo, hi)
            
            # Evaluate Offspring
            F_off = evaluate(offspring)
            
            # Selection: Combine parents and offspring, select best pop_size
            # We need the last 2*pop_size points from our history?
            # No, we should select from the current population and the new offspring.
            
            # Current Population: population, F
            # Offspring: offspring, F_off
            
            # Combine
            combined_X = np.vstack([population, offspring])
            combined_F = np.vstack([F, F_off])
            
            # Non-dominated Sort
            # Simple implementation:
            # 1. Find non-dominated set (Front 1)
            # 2. Remove them, find next front, etc.
            
            # For efficiency, we can just use a simple heuristic if N is small,
            # but proper non-domination is better.
            
            def non_dominated_sort(F):
                n = F.shape[0]
                ranks = np.zeros(n, dtype=int)
                fronts = []
                remaining = np.arange(n)
                
                while len(remaining) > 0:
                    front = []
                    for i in remaining:
                        dominated = False
                        for j in remaining:
                            if i == j:
                                continue
                            # Check if j dominates i
                            # j dominates i if F[j] <= F[i] and F[j] < F[i] in at least one
                            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                                dominated = True
                                break
                        if not dominated:
                            front.append(i)
                    
                    if not front:
                        # Should not happen if there are points, but break to be safe
                        break
                        
                    for i in front:
                        ranks[i] = len(fronts)
                    fronts.append(front)
                    remaining = np.array([x for x in remaining if x not in front])
                    
                return ranks, fronts

            ranks, fronts = non_dominated_sort(combined_F)
            
            # Select pop_size individuals based on rank and crowding distance
            # For simplicity, we just pick the first pop_size from the sorted fronts.
            # Proper NSGA-II uses crowding distance within the last front.
            
            # Sort indices by rank
            sorted_indices = np.argsort(ranks, kind='mergesort')
            
            # We need exactly pop_size individuals.
            # If the last front is too large, we need to truncate it.
            # For now, just taking the top pop_size by rank is a valid (though less diverse) selection.
            
            new_indices = sorted_indices[:pop_size]
            
            population = combined_X[new_indices]
            F = combined_F[new_indices]

        return np.array(all_X)

# EVOLVE-BLOCK-END
