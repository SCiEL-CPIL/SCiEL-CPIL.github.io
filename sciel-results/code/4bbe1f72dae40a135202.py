"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Adaptive Tchebycheff Multi-Objective Optimizer with Gap-Filling Strategy."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        used = 0
        archive_X = []
        archive_F = []
        
        def add_to_archive(x, f):
            nonlocal archive_X, archive_F
            f = np.array(f)
            # Check domination
            dominated = False
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    dominated = True
                    break
            
            if dominated:
                return False
            
            # Remove points dominated by f
            new_X = []
            new_F = []
            for ax, af in zip(archive_X, archive_F):
                if not (np.all(f <= af) and np.any(f < f)):
                    new_X.append(ax)
                    new_F.append(af)
            
            new_X.append(x)
            new_F.append(f)
            archive_X = new_X
            archive_F = new_F
            return True

        def get_best_point_in_direction(weights):
            """
            Find the best point in the archive along a specific weight direction.
            Used to determine where to place new weight vectors.
            """
            if not archive_F:
                return None
            # Project points onto the weight vector
            projections = [np.dot(f, weights) for f in archive_F]
            min_proj_idx = np.argmin(projections)
            return archive_F[min_proj_idx]

        def tchebycheff_obj(x, weights, ref_point, rho=100.0):
            """
            Augmented Tchebycheff scalarization.
            minimizes max_i { weights_i * (f_i(x) - ref_i) } + rho * sum_i { weights_i * (f_i(x) - ref_i) }
            """
            f = problem(x)
            used_evals[0] += 1
            
            # Clip f to ensure finite values for calculation, though problem should return finite
            f = np.array(f)
            
            # Standard Tchebycheff
            tcheb = np.max(weights * (f - ref_point))
            
            # Augmented term to ensure unique solutions
            aug = rho * np.sum(weights * (f - ref_point))
            
            return tcheb + aug

        # We need to track evaluations inside the objective function for minimize
        used_evals = [0]
        
        # Patch tchebycheff_obj to use the outer scope 'used' or a mutable list
        # Since 'used' is an integer in the outer scope, we can't increment it directly inside 
        # a nested function without 'nonlocal'. Let's use a mutable list 'used' wrapper.
        
        used_wrapper = [0]
        
        def tcheb_obj_wrapped(x, weights, ref_point, rho=100.0):
            f = problem(x)
            used_wrapper[0] += 1
            f = np.array(f)
            tcheb = np.max(weights * (f - ref_point))
            aug = rho * np.sum(weights * (f - ref_point))
            return tcheb + aug

        # Phase 1: Initial Sampling
        # Use Latin Hypercube for initial diverse set
        n_init = 30
        lhs_points = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            perm = np.arange(n_init)
            np.random.shuffle(perm)
            lhs_points[:, d] = (perm + 0.5) / n_init
        
        X_init = lo + lhs_points * span
        for x in X_init:
            if time.time() > deadline or used_wrapper[0] >= self.budget:
                break
            f = problem(x)
            used_wrapper[0] += 1
            add_to_archive(x, f)
        
        # Initialize reference point for Tchebycheff
        # If we have points, use the worst objective values found so far as a rough ref
        # But the problem states ref point for scoring is fixed. For optimization, 
        # we need a ref point that is "bad" (larger) than current best to make f-ref positive.
        # Let's estimate a bad reference point.
        # Since we don't know the scale, we can use the max of current archive + some margin.
        
        def update_ref_point():
            if not archive_F:
                # Default guess if no points yet (shouldn't happen after init)
                return np.array([1000.0, 100.0, 10.0]) 
            max_f = np.max(archive_F, axis=0)
            # Add a small margin to ensure we are "above" the front
            # Scale margin by magnitude
            margin = np.abs(max_f) * 0.1 + 1.0
            return max_f + margin

        ref_point = update_ref_point()

        # Phase 2: Adaptive Decomposition
        # We maintain a set of weight vectors.
        # Start with a uniform grid in weight space (simplex)
        # For n_obj=3, weights sum to 1.
        
        def generate_uniform_weights(n_levels=5):
            """Generate uniform weights on the simplex for 3 objectives."""
            weights = []
            for i in range(n_levels + 1):
                for j in range(n_levels + 1 - i):
                    k = n_levels - i - j
                    w = np.array([i, j, k], dtype=float)
                    w /= n_levels
                    weights.append(w)
            return weights
        
        # Start with a coarse grid
        initial_weights = generate_uniform_weights(n_levels=4)
        
        # We will iteratively refine this set.
        # Strategy:
        # 1. Evaluate all current weight vectors using L-BFGS-B from multiple starts.
        # 2. Add new weight vectors that point towards "gaps" in the current front.
        
        # To manage budget, we limit the number of weight vectors we actively pursue.
        # Let's say we keep a list of 'active' weights.
        
        active_weights = initial_weights[:]
        
        # We'll loop until budget or time is out.
        # Each "iteration" involves picking a weight, optimizing it, and potentially adding new weights.
        
        # To avoid re-optimizing the same weight repeatedly with no improvement,
        # we can track the best value found for each weight (approximate).
        # However, since the landscape is static, once a weight is optimized to convergence,
        # re-optimizing it is wasteful.
        # So, we will maintain a pool of weights. We pick a weight, optimize it.
        # If it produces a new non-dominated point, we keep it.
        # We generate new weights based on the current front geometry.
        
        # Let's use a queue for weights to process.
        weights_to_process = initial_weights[:]
        
        # Multi-start local search function
        def local_search(weight, ref_pt):
            """
            Perform multi-start L-BFGS-B for a given weight vector.
            Returns the best x found.
            """
            best_x = None
            best_val = np.inf
            
            # Start points:
            # 1. Random points
            # 2. Points from current archive (if any)
            
            starts = []
            
            # Add random starts
            n_random_starts = 3
            for _ in range(n_random_starts):
                starts.append(np.random.uniform(lo, hi))
            
            # Add archive points if available
            if archive_X:
                # Pick a few random archive points as starts
                n_archive_starts = min(2, len(archive_X))
                if n_archive_starts > 0:
                    idxs = np.random.choice(len(archive_X), size=n_archive_starts, replace=False)
                    for idx in idxs:
                        starts.append(archive_X[idx].copy())
            
            # Also try boundary points
            # For 5 dims, corners are 32. Too many.
            # Let's just use random and archive.
            
            for start_x in starts:
                if time.time() > deadline or used_wrapper[0] >= self.budget:
                    break
                
                # Define objective for minimize
                def obj(x):
                    return tcheb_obj_wrapped(x, weight, ref_pt)
                
                try:
                    res = minimize(obj, start_x, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50, 'maxfun': 20})
                    if res.success and res.fun < best_val:
                        best_val = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass
            
            return best_x

        # Main loop
        while time.time() < deadline and used_wrapper[0] < self.budget:
            
            if not weights_to_process:
                # If queue is empty, generate new weights based on gaps
                # Simple strategy: Random perturbation of existing weights or uniform grid refinement
                # Let's just generate a few random weights
                for _ in range(5):
                    w = np.random.dirichlet(np.ones(self.n_obj))
                    weights_to_process.append(w)
                continue
            
            # Pop a weight
            weight = weights_to_process.pop(0)
            
            # Update reference point if archive has changed significantly
            # Check if ref_point is still "above" all archive points
            if archive_F:
                max_f = np.max(archive_F, axis=0)
                if not np.all(ref_point > max_f):
                    ref_point = update_ref_point()
            
            # Perform local search
            best_x = local_search(weight, ref_point)
            
            if best_x is not None:
                f = problem(best_x)
                used_wrapper[0] += 1
                improved = add_to_archive(best_x, f)
                
                if improved:
                    # If we found a new point, it might be good to add nearby weights
                    # to explore the neighborhood of this new front point
                    # Perturb the weight slightly
                    for _ in range(2):
                        w_pert = weight + np.random.normal(0, 0.05, self.n_obj)
                        w_pert = np.clip(w_pert, 0, 1)
                        w_pert /= np.sum(w_pert)
                        weights_to_process.append(w_pert)
            
            # Check budget
            if used_wrapper[0] >= self.budget:
                break

        return archive_X, archive_F

# EVOLVE-BLOCK-END
