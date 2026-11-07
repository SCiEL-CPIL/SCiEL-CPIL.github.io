# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Initial population
        pop_size = 50
        # Initialize with random points
        # We use the seed provided by the evaluator (np.random.seed(seed) is called before)
        # But to be safe and reproducible within the function if needed, 
        # we rely on the global np.random state set by the evaluator.
        
        # However, the contract says: "For seeds 0 and 1 the evaluator calls np.random.seed(seed)"
        # So np.random is deterministic.
        
        population = np.random.uniform(lo, hi, size=(pop_size, self.dim))
        
        # Evaluate initial population
        evals_used = 0
        objectives = np.empty((pop_size, self.n_obj))
        
        for i in range(pop_size):
            if evals_used >= self.budget or time.time() > deadline:
                break
            try:
                objectives[i] = problem(population[i])
            except Exception:
                objectives[i] = np.full(self.n_obj, 11.0) # Penalty
            evals_used += 1
            
        # Keep track of all evaluated points for hypervolume
        all_points = [objectives[i] for i in range(evals_used) if np.all(objectives[i] < 11.0)]
        
        # Evolution loop
        while evals_used < self.budget and time.time() < deadline:
            # Selection: Non-dominated sort + Crowding distance
            # For simplicity and speed with small pop, we do a simple non-dominated sort
            
            # 1. Find non-dominated set
            non_dominated_idx = self._find_non_dominated(objectives)
            
            if len(non_dominated_idx) == 0:
                break
                
            # If we have a non-dominated set, we mutate them.
            # To maximize hypervolume, we want to spread out the non-dominated points.
            # We will mutate the non-dominated points to create offspring.
            
            # Determine how many offspring to create
            # We can create up to min(pop_size, budget - evals_used) offspring
            max_offspring = min(pop_size, self.budget - evals_used)
            
            # Select parents from non-dominated set
            # If non-dominated set is smaller than max_offspring, repeat some
            parent_indices = non_dominated_idx
            if len(parent_indices) < max_offspring:
                # Cycle through
                parent_indices = np.tile(parent_indices, int(np.ceil(max_offspring / len(parent_indices))))[:max_offspring]
            else:
                # Randomly pick max_offspring from non-dominated
                parent_indices = np.random.choice(parent_indices, size=max_offspring, replace=False)
                
            offspring_pop = np.empty((max_offspring, self.dim))
            offspring_obj = np.empty((max_offspring, self.n_obj))
            
            for i in range(max_offspring):
                if time.time() > deadline:
                    break
                parent = population[parent_indices[i]]
                
                # Mutation: Gaussian mutation with adaptive sigma
                # Start with larger sigma, reduce over time
                progress = evals_used / self.budget
                sigma = 0.5 * (1.0 - progress) + 0.01
                
                child = parent + np.random.normal(0, sigma, self.dim)
                # Clip to bounds
                child = np.clip(child, lo, hi)
                
                offspring_pop[i] = child
                try:
                    offspring_obj[i] = problem(child)
                except Exception:
                    offspring_obj[i] = np.full(self.n_obj, 11.0)
                evals_used += 1
                
                if np.all(offspring_obj[i] < 11.0):
                    all_points.append(offspring_obj[i])

            # Update population
            # Combine parent and offspring, select best
            combined_pop = np.vstack([population, offspring_pop])
            combined_obj = np.vstack([objectives, offspring_obj])
            
            # Select next population
            # Simple strategy: Keep all non-dominated points from combined set, 
            # fill rest with best dominated points (lowest objective sum)
            
            # Find non-dominated in combined
            nd_idx = self._find_non_dominated(combined_obj)
            
            if len(nd_idx) > pop_size:
                # Too many non-dominated, use crowding distance to select
                # Simplified: just take random or first pop_size
                # Better: select evenly spaced in objective space
                nd_points = combined_obj[nd_idx]
                # Normalize
                min_obj = nd_points.min(axis=0)
                max_obj = nd_points.max(axis=0)
                range_obj = max_obj - min_obj
                range_obj[range_obj == 0] = 1.0
                norm_points = (nd_points - min_obj) / range_obj
                
                # Calculate crowding distance (simplified: sum of distances to neighbors in sorted order)
                # For 2 objectives, sort by f1, calculate distance to neighbors
                sort_idx = np.argsort(norm_points[:, 0])
                sorted_norm = norm_points[sort_idx]
                
                distances = np.zeros(len(sorted_norm))
                if len(sorted_norm) > 2:
                    for i in range(1, len(sorted_norm) - 1):
                        dist = np.sum(np.abs(sorted_norm[i] - sorted_norm[i-1]) + np.abs(sorted_norm[i] - sorted_norm[i+1]))
                        distances[i] = dist
                    distances[0] = np.inf
                    distances[-1] = np.inf
                else:
                    distances[:] = np.inf
                    
                # Map distances back to original indices
                # We want to pick pop_size points with largest distances
                # Sort distances descending
                dist_order = np.argsort(distances)[::-1]
                selected_nd_local = dist_order[:pop_size]
                selected_nd_global = nd_idx[selected_nd_local]
                
                next_pop_idx = selected_nd_global
            else:
                # Fewer non-dominated than pop_size
                next_pop_idx = nd_idx.copy()
                
                # Fill remaining slots with best dominated points
                if len(next_pop_idx) < pop_size:
                    dominated_idx = np.setdiff1d(np.arange(len(combined_pop)), next_pop_idx)
                    if len(dominated_idx) > 0:
                        dominated_obj = combined_obj[dominated_idx]
                        # Sort by sum of objectives (lower is better)
                        sums = dominated_obj.sum(axis=1)
                        best_dominated_local = np.argsort(sums)[:pop_size - len(next_pop_idx)]
                        best_dominated_global = dominated_idx[best_dominated_local]
                        next_pop_idx = np.concatenate([next_pop_idx, best_dominated_global])
                        
            # Ensure we have exactly pop_size
            if len(next_pop_idx) < pop_size:
                # Fallback: random fill
                missing = pop_size - len(next_pop_idx)
                remaining = np.setdiff1d(np.arange(len(combined_pop)), next_pop_idx)
                if len(remaining) > 0:
                    fill_idx = np.random.choice(remaining, size=missing, replace=False)
                    next_pop_idx = np.concatenate([next_pop_idx, fill_idx])
            
            next_pop_idx = next_pop_idx[:pop_size]
            
            population = combined_pop[next_pop_idx]
            objectives = combined_obj[next_pop_idx]

        return all_points

    def _find_non_dominated(self, objectives):
        """
        Find indices of non-dominated points.
        A point is non-dominated if no other point is better in all objectives and strictly better in at least one.
        """
        n = len(objectives)
        is_dominated = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(i + 1, n):
                if is_dominated[j]:
                    continue
                # Check if i dominates j
                if np.all(objectives[i] <= objectives[j]) and np.any(objectives[i] < objectives[j]):
                    is_dominated[j] = True
                # Check if j dominates i
                elif np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                    is_dominated[i] = True
                    break
                    
        return np.where(~is_dominated)[0]
# EVOLVE-BLOCK-END
