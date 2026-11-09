"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective evolutionary optimizer with non-dominated archive and crowding distance."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        pop_size = 40
        # Initialize population
        X = lo + range_ * np.random.rand(pop_size, self.dim)
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper to check domination
        def is_dominated(f_new, F_set):
            # Returns True if any point in F_set dominates f_new
            # Dominance: a <= b and a != b
            # Vectorized: 
            # less_or_equal = np.all(F_set <= f_new, axis=1)
            # less = np.any(F_set < f_new, axis=1)
            # return np.any(less_or_equal & less)
            return np.any(np.all(F_set <= f_new, axis=1) & np.any(F_set < f_new, axis=1))
            
        def is_non_dominated(f_new, F_set):
            # Returns True if f_new is not dominated by any point in F_set
            # AND f_new does not dominate any point in F_set? 
            # For archive insertion, we usually check if it's dominated.
            # If not dominated, it's a candidate for non-dominated set.
            # We also need to remove points it dominates.
            return not is_dominated(f_new, F_set)

        def get_non_dominated_indices(F_set):
            # Returns indices of non-dominated points in F_set
            n = len(F_set)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                # Check if any j != i dominates i
                # F[j] <= F[i] and F[j] != F[i]
                # To be efficient, we can vectorize partially, but n is small (40)
                for j in range(n):
                    if i == j or not nd[j]:
                        continue
                    if np.all(F_set[j] <= F_set[i]) and np.any(F_set[j] < F_set[i]):
                        nd[i] = False
                        break
            return np.where(nd)[0]

        def crowding_distance(F_set, nd_indices):
            # Calculate crowding distance for non-dominated points
            # Returns array of distances for all points, 0 for dominated
            cd = np.zeros(len(F_set))
            if len(nd_indices) <= 2:
                # If 0 or 1 or 2 points, assign infinity (or large number) to keep them
                cd[nd_indices] = np.inf
                return cd
            
            for k in range(self.n_obj):
                # Sort non-dominated points by objective k
                sorted_idx = nd_indices[np.argsort(F_set[nd_indices, k])]
                # Assign inf to the two extremes
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                # Calculate range
                f_min = F_set[sorted_idx[0], k]
                f_max = F_set[sorted_idx[-1], k]
                if f_max == f_min:
                    continue
                for i in range(1, len(sorted_idx) - 1):
                    idx_prev = sorted_idx[i-1]
                    idx_next = sorted_idx[i+1]
                    cd[sorted_idx[i]] += (F_set[idx_next, k] - F_set[idx_prev, k]) / (f_max - f_min)
            return cd

        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Find non-dominated points in current population
            nd_indices = get_non_dominated_indices(F)
            
            if len(nd_indices) == 0:
                # Fallback: random parent
                parent_idx = np.random.randint(len(X))
            else:
                # Select parent from non-dominated points
                parent_idx = nd_indices[np.random.randint(len(nd_indices))]
            
            # Mutate
            # Adaptive sigma? Start with 0.1, maybe reduce?
            # WFG4 is deceptive. Let's stick with 0.1 * range for exploration.
            sigma = 0.1
            mutation = sigma * range_ * np.random.randn(self.dim)
            child = X[parent_idx] + mutation
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Check if child is dominated by current population
            if is_dominated(fc, F):
                # Child is worse than some points, discard
                continue
            
            # Child is non-dominated with respect to population
            # Determine replacement
            # 1. If there are dominated points in population, replace a random one
            dominated_indices = np.where(~np.isin(np.arange(len(X)), nd_indices))[0]
            
            if len(dominated_indices) > 0:
                # Replace a random dominated point
                replace_idx = dominated_indices[np.random.randint(len(dominated_indices))]
                X[replace_idx] = child
                F[replace_idx] = fc
            else:
                # All points are non-dominated. 
                # We need to trim the set to pop_size.
                # Add child temporarily
                X_new = np.vstack([X, child])
                F_new = np.vstack([F, fc])
                
                # Find new non-dominated set
                nd_new_indices = get_non_dominated_indices(F_new)
                
                if len(nd_new_indices) > pop_size:
                    # Calculate crowding distance for the new non-dominated set
                    cd = crowding_distance(F_new, nd_new_indices)
                    # Find the point with minimum crowding distance among the non-dominated set
                    # We want to remove the one with smallest CD
                    min_cd_idx = nd_new_indices[np.argmin(cd[nd_new_indices])]
                    # Remove that point
                    mask = np.ones(len(X_new), dtype=bool)
                    mask[min_cd_idx] = False
                    X = X_new[mask]
                    F = F_new[mask]
                else:
                    # Just keep the new set (it might be smaller than pop_size if some old points were dominated by child? 
                    # No, child is non-dominated, so it doesn't dominate others unless they are identical? 
                    # Actually, child might dominate some old non-dominated points if they are "worse" in some sense? 
                    # Wait, "non-dominated" means no point in the set dominates it. 
                    # If child is non-dominated wrt F, it doesn't dominate any point in F? 
                    # No, it can dominate points in F. 
                    # Example: F has (1,1) and (2,2). Child is (1, 1.5). 
                    # Child is not dominated by (1,1) because 1.5 > 1. 
                    # Child is not dominated by (2,2) because 1 < 2. 
                    # Does Child dominate (2,2)? 1 < 2 and 1.5 < 2. Yes. 
                    # So (2,2) becomes dominated. 
                    # My `is_dominated(fc, F)` check only checks if child is dominated. 
                    # I need to handle the case where child dominates existing points.
                    
                    # Let's re-evaluate:
                    # If child is non-dominated wrt F, it means no f in F dominates child.
                    # It is possible that child dominates some f in F.
                    # In that case, those f are now dominated and should be removed.
                    
                    # So, we should update the set to: (F U {child}) \ {points dominated by F U {child}}
                    
                    # Let's just rebuild the population from the new set.
                    # The new population should be the non-dominated set of (F U {child}).
                    # If the size is > pop_size, trim by CD.
                    # If size <= pop_size, we can just use it. 
                    # But we maintain a fixed pop_size for diversity? 
                    # If we drop to size 10, we lose diversity. 
                    # It's better to keep dominated points? No, dominated points are useless for HV.
                    # But we need candidates for mutation. 
                    # If we only keep non-dominated, we might get stuck in a local region.
                    # However, with random initialization and mutation, we should be okay.
                    
                    X = X_new[nd_new_indices]
                    F = F_new[nd_new_indices]
                    
                    # If we fell below pop_size, we could re-randomize some slots? 
                    # No, let's just let it grow/shrink naturally or refill?
                    # To keep it simple, if len(X) < pop_size, we don't refill immediately.
                    # The loop will continue.
                    
        return X
# EVOLVE-BLOCK-END
