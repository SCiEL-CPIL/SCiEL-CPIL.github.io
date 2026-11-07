import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        # We need to be careful with the budget.
        # Let's split the budget: 70% for global exploration (Sobol), 30% for local refinement.
        n_init = int(self.budget * 0.7)
        n_refine = self.budget - n_init
        
        # Ensure n_init is at least a small number
        n_init = max(n_init, 10)
        
        # 1. Sobol Initialization
        # scipy.stats.qmc.Sobol requires dimensionality
        sampler = qmc.Sobol(d=self.dim, scramble=True)
        # Sobol generates points in [0, 1]^d
        # We need n_init points. Sobol requires power of 2 for best performance, but works for any N.
        # Let's just generate n_init points.
        sobol_samples = sampler.random(n_init)
        
        # Scale to bounds
        X_init = lo + sobol_samples * (hi - lo)
        
        F_init = np.array([problem(x) for x in X_init], dtype=float)
        used = n_init
        
        # Store all evaluated points to maintain a global archive for hypervolume calculation
        # The evaluator records every call to problem().
        # We need to maintain a list of all non-dominated points found so far to guide the search?
        # Actually, the evaluator records ALL returned vectors.
        # So we just need to make good calls to problem().
        
        # Identify non-dominated points in the initial set
        def is_dominated(f, F):
            # f is dominated if there exists g in F such that g <= f and g != f
            # F is (N, M)
            # Check if any row in F dominates f
            # g <= f (all objectives) and g < f (at least one)
            leq = np.all(F <= f, axis=1)
            lt = np.any(F < f, axis=1)
            return np.any(leq & lt)
            
        def non_dominated_set(F):
            n = len(F)
            nd_indices = []
            for i in range(n):
                # Check if F[i] is dominated by any other point
                dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    nd_indices.append(i)
            return nd_indices

        # Get initial non-dominated indices
        nd_indices = non_dominated_set(F_init)
        
        # If no non-dominated points found (unlikely), pick the best
        if not nd_indices:
            nd_indices = [int(np.argmin(np.sum(F_init, axis=1)))]

        # 2. Refinement Phase
        # We have n_refine evaluations left.
        # Strategy: Pick a point from the current non-dominated set, mutate it, evaluate.
        # If the new point is non-dominated with respect to the *global* set of all evaluated points, keep it.
        # To keep the global set manageable, we can just maintain the list of all evaluated points.
        # With 200 points, checking dominance is O(N^2) which is 40,000 ops, very fast.
        
        all_X = X_init
        all_F = F_init
        
        rng = np.random.default_rng(42)
        
        for _ in range(n_refine):
            if time.time() > deadline:
                break
            if used >= self.budget:
                break
                
            # Select a parent from the current non-dominated set
            # Recalculate non-dominated set from all_F to be accurate
            # This is O(N^2 * M). N=200, M=3. 200*200*3 = 120,000. Fast.
            current_nd = non_dominated_set(all_F)
            
            if not current_nd:
                # Fallback
                parent_idx = int(np.argmin(np.sum(all_F, axis=1)))
            else:
                # Randomly pick from non-dominated to maintain diversity
                parent_idx = current_nd[rng.integers(len(current_nd))]
                
            parent = all_X[parent_idx].copy()
            
            # Mutation: Gaussian perturbation
            # Scale the mutation by the range of the variables
            # Use a decay or fixed small sigma
            sigma = 0.1 * (hi - lo)
            child = parent + rng.normal(0, sigma)
            child = np.clip(child, lo, hi)
            
            f_child = problem(child)
            used += 1
            
            # Add to global pool
            all_X = np.vstack([all_X, child])
            all_F = np.vstack([all_F, f_child])
            
        return all_F
