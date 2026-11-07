import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Reference point for hypervolume (used for pruning if needed, though we just return all non-dominated)
        # The scoring uses a specific reference point, but for the algorithm, 
        # we just want to find the Pareto front.
        
        # 1. Initialization
        # Use a mix of random and grid-like points to ensure spread
        n_init = 20
        X = []
        F = []
        
        # Latin Hypercube Sampling for better initial spread
        # Simple implementation for small n
        indices = np.argsort(np.random.rand(n_init, dim), axis=0)
        X_init = (indices + 0.5) / n_init
        X_init = X_init * (hi - lo) + lo
        
        for x in X_init:
            if budget <= 0: break
            f = problem(x)
            X.append(x)
            F.append(f)
            budget -= 1
            
        X = np.array(X)
        F = np.array(F)
        
        # 2. Main Evolutionary Loop
        # We maintain a list of non-dominated solutions
        # To speed up dominance checks, we can keep them in a list and filter
        
        def is_dominated(f, F_set):
            # Returns True if f is dominated by any point in F_set
            # F_set is (N, n_obj)
            if len(F_set) == 0:
                return False
            # f <= F_set[i] for all i, and f < F_set[i] for at least one i
            # Vectorized check
            # all_le = np.all(F_set <= f, axis=1)
            # any_lt = np.any(F_set < f, axis=1)
            # return np.any(all_le & any_lt)
            
            # More efficient: 
            # Dominates if: (F_set <= f).all(axis=1) & (F_set < f).any(axis=1)
            # Wait, A dominates B if A <= B and A < B.
            # So B is dominated by A if A <= B and A < B.
            # Here we check if f is dominated by any point in F_set.
            # So we check if any point in F_set dominates f.
            # Point p in F_set dominates f if p <= f and p < f.
            
            le = np.all(F_set <= f, axis=1)
            lt = np.any(F_set < f, axis=1)
            return np.any(le & lt)

        def remove_dominated(F_set):
            # Returns indices of non-dominated points in F_set
            n = len(F_set)
            if n == 0: return []
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]: continue
                # Check if i dominates j
                # i dominates j if F[i] <= F[j] and F[i] < F[j]
                le = np.all(F_set[i] <= F_set, axis=1)
                lt = np.any(F_set[i] < F_set, axis=1)
                dom_mask = le & lt
                # i does not dominate itself in this check? 
                # F[i] <= F[i] is True. F[i] < F[i] is False. So lt is False for i.
                # So i is not marked as dominating itself.
                dominated[dom_mask] = True
                dominated[i] = False # Ensure i is not marked dominated by itself if logic was loose, but lt handles it.
            
            return np.where(~dominated)[0]

        # Initial non-dominated set
        nd_indices = remove_dominated(F)
        X_nd = X[nd_indices]
        F_nd = F[nd_indices]
        
        all_X = [X_nd.copy()] # List of arrays? No, just keep the current non-dominated set.
        # Actually, we just need to return the final non-dominated set.
        # The problem says "Every returned objective vector is recorded".
        # Wait, "Every returned objective vector is recorded... and the return value of __call__ is ignored."
        # This implies the evaluator calls `problem(x)` and records the result?
        # "The evaluator calls np.random.seed(seed), creates Optimizer... and calls it once."
        # "Every returned objective vector is recorded" -> This refers to the output of `problem(x)` calls made inside `__call__`?
        # "problem(x) clips ... and returns 3 objective values."
        # "Every returned objective vector is recorded" -> The evaluator likely wraps `problem` to record calls.
        # So I just need to make up to `self.budget` calls to `problem`.
        
        # So the goal is to maximize the hypervolume of the set of *all* points evaluated?
        # "Exact hypervolume of all recorded vectors" -> Yes, the union of all points evaluated.
        # This means I don't need to maintain a non-dominated set for the *result*, 
        # but I should use non-dominated points as parents to guide the search towards the front.
        
        # Let's refine the loop to use the current non-dominated front for selection.
        
        while budget > 0:
            if time.time() > deadline:
                break
                
            if len(F_nd) == 0:
                # Fallback to random
                x = lo + (hi - lo) * np.random.rand(dim)
            else:
                # Select parents from non-dominated set
                # Use tournament selection or just random from non-dominated
                # To encourage diversity, pick two random distinct non-dominated points
                if len(F_nd) == 1:
                    p1 = F_nd[0]
                    p2 = F_nd[0]
                    x1 = X_nd[0]
                    x2 = X_nd[0]
                else:
                    idx1 = np.random.randint(len(F_nd))
                    idx2 = np.random.randint(len(F_nd))
                    p1 = F_nd[idx1]
                    p2 = F_nd[idx2]
                    x1 = X_nd[idx1]
                    x2 = X_nd[idx2]
                    
                # Crossover: Blend or SBX
                # Simple uniform crossover or blend
                # Let's use a blend crossover with high probability of mutation
                if np.random.rand() < 0.5:
                    # Crossover
                    beta = np.random.rand(dim)
                    x = beta * x1 + (1 - beta) * x2
                else:
                    # Just pick one parent and mutate
                    x = x1.copy() if np.random.rand() < 0.5 else x2.copy()
                
                # Mutation
                # Adaptive step size
                # Start with larger steps, shrink over time
                # t is fraction of budget used
                t = 1.0 - (budget / self.budget)
                step_size = 0.1 * (1.0 - t) + 0.01
                # Gaussian mutation
                x = x + step_size * (hi - lo) * np.random.randn(dim)
                x = np.clip(x, lo, hi)
            
            f = problem(x)
            budget -= 1
            
            # Update non-dominated set for future selection
            # Check if x is dominated by F_nd
            if not is_dominated(f, F_nd):
                # Add x to non-dominated set
                F_nd = np.vstack([F_nd, f.reshape(1, -1)])
                X_nd = np.vstack([X_nd, x.reshape(1, -1)])
                
                # Remove points in F_nd dominated by x
                # x dominates y if x <= y and x < y
                if len(F_nd) > 1:
                    # Check which points in F_nd are dominated by x
                    # F_nd includes x now.
                    # We want to remove y where x <= y and x < y.
                    # Note: x is in F_nd. x does not dominate itself.
                    
                    # Vectorized check: 
                    # le = np.all(F_nd <= x, axis=1) -> F_nd[i] <= x
                    # lt = np.any(F_nd < x, axis=1) -> F_nd[i] < x
                    # We want to keep i if NOT (le & lt)
                    # But we must keep x itself.
                    
                    le = np.all(F_nd <= x, axis=1)
                    lt = np.any(F_nd < x, axis=1)
                    dominated_by_x = le & lt
                    
                    # The new point x is at the last index
                    # It won't dominate itself because lt is False for itself.
                    # So dominated_by_x[last] will be False.
                    
                    keep_mask = ~dominated_by_x
                    F_nd = F_nd[keep_mask]
                    X_nd = X_nd[keep_mask]
                    
                    # To prevent the non-dominated set from growing too large (memory/speed),
                    # we can cap it. If it gets too big, we might lose diversity or speed up.
                    # With 200 budget, the front won't be huge.
            
            # Optional: If F_nd is very large, prune it?
            # For 4D/3Obj, the front is likely small.

        # Final Polish?
        # If we have time/budget left, we could try to improve the best points.
        # But "budget" is the hard limit on evaluations.
        # The loop stops when budget is 0.
        
        return None # Return value is ignored
