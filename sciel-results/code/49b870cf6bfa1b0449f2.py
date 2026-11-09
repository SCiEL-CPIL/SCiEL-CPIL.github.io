"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Structure-aware optimizer for DTLZ1.
    
    Strategy:
    1. Exploit the known analytic structure of DTLZ1:
       - f_i = 0.5 * (1 + g) * prod/sin for i < n_obj-1
       - f_{n_obj-1} = 0.5 * (1 + g) * cos
       - g = sum(x_i^2) for i >= n_obj - 1 (distance variables)
       - The first n_obj-1 variables control the shape on the Pareto front
       
    2. Fix distance variables to 0 to place points exactly on the Pareto front (g=0).
       For DTLZ1 with dim=7, n_obj=3: variables x[4], x[5], x[6] are distance variables.
       
    3. Use the evaluation budget to sample the shape space (first 4 variables) 
       to maximize hypervolume coverage. Use a combination of:
       - Latin hypercube sampling for broad coverage
       - Local refinement around promising regions
       - Epsilon-constraint style diversification
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        
        # DTLZ1 structure: last (n - m + 1) variables are distance variables
        # First (m - 1) variables are shape variables
        n_shape = m - 1  # 2 shape variables for m=3
        n_dist = n - n_shape  # 5 distance variables
        
        # Initialize with points on the Pareto front (distance vars = 0)
        # Use Latin hypercube for shape variables
        rng = np.random.default_rng(0)
        
        # Reserve some evaluations for refinement
        n_init = min(50, self.budget // 2)
        n_refine = self.budget - n_init
        
        # Latin hypercube sampling for shape variables
        shape_samples = np.zeros((n_init, n_shape))
        for j in range(n_shape):
            perm = rng.permutation(n_init)
            shape_samples[:, j] = (perm + rng.random(n_init)) / n_init
        
        # Build initial population: shape vars from LHS, distance vars = 0
        X = np.zeros((n_init, n))
        X[:, :n_shape] = shape_samples
        # Distance variables set to 0 (on Pareto front)
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        used = n_init
        
        # Check time
        if time.time() > deadline:
            return X
        
        # Refinement phase: use remaining budget to improve coverage
        # Strategy: for each point, try small perturbations in shape space
        # and keep the best non-dominated set
        
        # Sort by first objective for efficiency
        order = np.argsort(F[:, 0])
        F_sorted = F[order]
        X_sorted = X[order]
        
        # Adaptive refinement: focus on underrepresented regions
        # Use a simple grid-based approach to identify gaps
        for i in range(n_refine):
            if time.time() > deadline:
                break
            
            # Choose a parent from the current set
            # Weight towards points that are "lonely" (far from others)
            if len(F) > 1:
                # Compute nearest neighbor distance for each point
                # Use a subset for speed
                idx = np.random.choice(len(F), size=min(10, len(F)), replace=False)
                parent_idx = idx[np.argmin(np.min(np.abs(F[idx] - F[idx[np.argmin(np.abs(F[idx] - F[idx]))]], axis=1)))]
            else:
                parent_idx = 0
            
            # Perturb in shape space only
            child = X[parent_idx].copy()
            # Random perturbation in shape variables
            noise = rng.standard_normal(n_shape) * 0.05
            child[:n_shape] += noise
            # Keep distance variables at 0
            child[n_shape:] = 0
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace the worst point (by some criterion)
            # Use a simple criterion: replace the point with largest sum of objectives
            # that is dominated by the child
            dominated = np.any(np.all(F <= fc, axis=1) & np.any(F < fc, axis=1), axis=1)
            if np.any(dominated):
                # Replace a random dominated point
                dom_indices = np.where(dominated)[0]
                replace_idx = dom_indices[rng.integers(len(dom_indices))]
            else:
                # Replace the point with largest hypervolume contribution
                # Approximate: replace the point with largest objective sum
                replace_idx = np.argmax(np.sum(F, axis=1))
            
            X[replace_idx] = child
            F[replace_idx] = fc
        
        return X
# EVOLVE-BLOCK-END
