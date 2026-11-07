import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        time_limit = 300  # seconds
        safety_margin = 10  # seconds
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)
        
        pop_size = 100
        X = np.zeros((pop_size, self.dim))
        # Initialize x0, x1 uniformly
        X[:, :2] = rng.random((pop_size, 2))
        # Initialize z (rest) uniformly but small to help g(z)
        X[:, 2:] = rng.random((pop_size, self.dim - 2)) * 0.5
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)
        
        def get_non_dominated(F):
            nd_mask = np.ones(len(F), dtype=bool)
            for i in range(len(F)):
                if not nd_mask[i]:
                    continue
                for j in range(len(F)):
                    if i == j or not nd_mask[j]:
                        continue
                    if dominates(F[j], F[i]):
                        nd_mask[i] = False
            return nd_mask
        
        # Initial non-dominated sort
        nd_mask = get_non_dominated(F)
        
        while used < self.budget:
            # Check time
            if time.time() - start_time > time_limit - safety_margin:
                break
                
            # Select parents from non-dominated set if available
            if np.any(nd_mask):
                parents_idx = np.where(nd_mask)[0]
            else:
                parents_idx = np.arange(pop_size)
                
            p1_idx = parents_idx[rng.integers(len(parents_idx))]
            p2_idx = parents_idx[rng.integers(len(parents_idx))]
            
            p1 = X[p1_idx]
            p2 = X[p2_idx]
            
            # Crossover for x0, x1
            alpha = rng.random()
            child_x01 = alpha * p1[:2] + (1 - alpha) * p2[:2]
            
            # Mutation for x0, x1
            child_x01 += 0.1 * rng.standard_normal(2)
            child_x01 = np.clip(child_x01, 0, 1)
            
            # Mutation for z
            # Pick z from one of the parents, mutate with pull to 0
            if rng.random() > 0.5:
                child_z = p1[2:].copy()
            else:
                child_z = p2[2:].copy()
                
            # Pull to 0: shrink by 0.5 and add noise
            child_z = child_z * 0.5 + 0.05 * rng.standard_normal(self.dim - 2)
            child_z = np.clip(child_z, 0, 1)
            
            child = np.concatenate([child_x01, child_z])
            fc = problem(child)
            used += 1
            
            # Replace worst point
            # Find dominated points
            # If child is non-dominated, replace a random dominated point
            # If child is dominated, replace the worst dominated point if it dominates the child? No, just replace worst.
            
            # Simple strategy: Replace the point with the highest sum of objectives if it is dominated by child
            # Or just replace the worst point in the population if child is better than it.
            
            # Let's identify the worst point to replace.
            # We want to maintain diversity.
            # If child is non-dominated, replace a dominated point.
            # If child is dominated, replace the worst dominated point only if child is better than it.
            
            # Calculate dominance of child against all
            child_dominates = np.array([dominates(fc, f) for f in F])
            is_dominated_by_any = np.any(np.all(F <= fc, axis=1) & np.any(F < fc, axis=1))
            
            if not is_dominated_by_any:
                # Child is non-dominated. Replace a dominated point if exists.
                dominated_mask = ~get_non_dominated(F)
                if np.any(dominated_mask):
                    dom_idx = np.where(dominated_mask)[0]
                    # Replace the one with highest sum
                    sums = np.sum(F[dom_idx], axis=1)
                    worst_in_dom = dom_idx[np.argmax(sums)]
                    X[worst_in_dom] = child
                    F[worst_in_dom] = fc
                else:
                    # All are non-dominated. Replace the one with highest sum?
                    # This might collapse diversity.
                    # Maybe don't replace if no dominated points?
                    # But we need to use the evaluation.
                    # Let's replace the worst (highest sum) to keep the population "tight"
                    worst_idx = np.argmax(np.sum(F, axis=1))
                    X[worst_idx] = child
                    F[worst_idx] = fc
            else:
                # Child is dominated.
                # Check if it dominates any existing point.
                if np.any(child_dominates):
                    # Replace one of the points it dominates
                    dom_candidates = np.where(child_dominates)[0]
                    # Replace the one with highest sum
                    sums = np.sum(F[dom_candidates], axis=1)
                    worst_in_dom = dom_candidates[np.argmax(sums)]
                    X[worst_in_dom] = child
                    F[worst_in_dom] = fc
                else:
                    # Child is dominated and doesn't dominate anyone.
                    # Discard? Or replace the worst if it's very bad?
                    # To ensure we don't waste evaluations, we might just skip replacement
                    # but the budget is counted.
                    # Let's replace the worst point if the child is "less bad" than the worst.
                    worst_idx = np.argmax(np.sum(F, axis=1))
                    if np.sum(fc) < np.sum(F[worst_idx]):
                        X[worst_idx] = child
                        F[worst_idx] = fc
            
            # Update non-dominated mask occasionally?
            # It's expensive. Let's do it every 10 steps.
            if used % 10 == 0:
                nd_mask = get_non_dominated(F)
        
        return X
