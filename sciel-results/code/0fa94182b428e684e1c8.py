# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using a combination of:
    1. Structured initial sampling with cosine spacing
    2. NSGA-II-style evolutionary search
    3. Deadline-aware execution
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s limit
        
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # DTLZ4 specific: last few variables are the decision variables for the front
        # For DTLZ4, the Pareto front is determined by the last (n_obj-1) variables
        # and the first (dim - n_obj + 1) variables affect convergence
        
        # We'll use a population-based approach
        pop_size = min(100, max(20, budget // 2))
        
        # Initial population: mix of random and structured sampling
        # Use cosine spacing for the front variables to concentrate near vertices
        X = np.zeros((pop_size, dim))
        
        # For DTLZ4, variables 0 to dim-2 are "position" variables
        # Variable dim-1 (last) is the front variable when n_obj=3
        # Actually for DTLZ4 with n_obj objectives:
        # - First (dim - n_obj + 1) variables are position variables (affect convergence)
        # - Last (n_obj - 1) variables define the Pareto front shape
        
        n_pos = dim - n_obj + 1  # Number of position variables
        n_front = n_obj - 1  # Number of front variables
        
        # Generate structured samples for the front variables
        # For n_obj=3, n_front=2, we need to sample on a 2D simplex
        
        # Create initial population with good spread
        for i in range(pop_size):
            # Position variables: use cosine spacing to concentrate near 0
            # (for DTLZ4, x^alpha with alpha=100, so values near 0 or 1 are on front)
            if i < pop_size // 2:
                # Use cosine spacing for position variables
                t = (i + 0.5) / (pop_size // 2)
                X[i, :n_pos] = np.cos(t * np.pi / 2)  # Concentrates near 1
                # Actually for DTLZ4, we want x^100, so values near 0 or 1
                # Let's use a mix
                X[i, :n_pos] = rng.random(n_pos) ** 10
            else:
                X[i, :n_pos] = rng.random(n_pos)
            
            # Front variables: use structured sampling on simplex
            # For n_obj=3, we sample on the 2D simplex x1+x2+x3=1
            if n_front == 2:
                # Sample 2 variables, third is 1 - sum
                # Use cosine spacing for better coverage
                u = (i + 0.5) / pop_size
                v = rng.random()
                # Map to simplex
                X[i, n_pos] = u
                X[i, n_pos + 1] = v * (1 - u)
                # The remaining front variable is implicitly 1 - u - v*(1-u)
                # But DTLZ4 uses the last (n_obj-1) variables directly
                # Let me check: for DTLZ4 with 3 objectives, the last 2 variables
                # are the front variables, and they should be in [0,1] with
                # the constraint that the "implicit" third is 1 - sum
                # Actually, standard DTLZ4: the last (m-1) variables are in [0,1]
                # and the front is defined by sin/cos of these
                # Let me just use random for now and refine
                X[i, n_pos:n_pos+n_front] = rng.random(n_front)
            else:
                X[i, n_pos:n_pos+n_front] = rng.random(n_front)
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper: compute dominance
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)
        
        def non_dominated_indices(F):
            n = len(F)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j or not nd[j]:
                        continue
                    if dominates(F[j], F[i]):
                        nd[i] = False
                        break
            return np.where(nd)[0]
        
        # Evolutionary loop
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Find non-dominated solutions
            nd_idx = non_dominated_indices(F)
            
            if len(nd_idx) == 0:
                # All dominated, pick random parent
                parent_idx = rng.integers(len(F))
            else:
                # Pick from non-dominated with crowding-like preference
                # Simple: pick random non-dominated
                parent_idx = nd_idx[rng.integers(len(nd_idx))]
            
            parent = X[parent_idx].copy()
            
            # Mutation: Gaussian with adaptive step size
            # Step size decreases over time
            progress = used / budget
            step = 0.15 * (1 - 0.5 * progress)
            
            child = parent + step * rng.standard_normal(dim)
            child = np.clip(child, lo, hi)
            
            # For DTLZ4, apply the power transformation to position variables
            # x' = x^10 to concentrate near 0
            # But we're already in the transformed space? No, we're in the original space
            # The problem function handles the DTLZ4 computation
            # So we just clip to [0,1]
            
            fc = problem(child)
            used += 1
            
            # Replace the worst solution (highest sum of objectives)
            # But be careful: we want to maintain diversity
            # Simple approach: replace a dominated solution if possible
            dominated_mask = np.array([
                any(dominates(F[j], F[i]) for j in range(len(F)) if j != i)
                for i in range(len(F))
            ])
            
            if np.any(dominated_mask):
                # Replace a random dominated solution
                dom_indices = np.where(dominated_mask)[0]
                replace_idx = dom_indices[rng.integers(len(dom_indices))]
            else:
                # Replace the solution with the highest objective sum
                replace_idx = int(np.argmax(np.sum(F, axis=1)))
            
            X[replace_idx] = child
            F[replace_idx] = fc
        
        return X
# EVOLVE-BLOCK-END
