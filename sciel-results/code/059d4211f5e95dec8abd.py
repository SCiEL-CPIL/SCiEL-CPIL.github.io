import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """
    Multi-objective optimizer for WFG9 using Latin Hypercube Sampling 
    and local refinement.
    """
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)
        
        # 1. Initial Phase: Latin Hypercube Sampling for broad coverage
        # Use 180 points to leave 20 for refinement
        n_init = min(180, self.budget)
        sampler = qmc.LatinHypercube(d=self.dim, n=n_init, seed=42)
        X_init = sampler.random() * (hi - lo) + lo
        
        # Evaluate initial points
        F = np.array([problem(x) for x in X_init])
        used = n_init
        
        # Maintain a set of non-dominated points
        # We'll store them in lists for easy appending
        nd_points = []
        nd_values = []
        
        def is_dominated(f, points):
            """Check if f is dominated by any point in points."""
            if not points:
                return False
            points_arr = np.array(points)
            # f is dominated if there exists p such that p <= f and p != f
            # p <= f (all objectives)
            cond1 = np.all(points_arr <= f, axis=1)
            # p != f (at least one strictly less)
            cond2 = np.any(points_arr < f, axis=1)
            return np.any(cond1 & cond2)
        
        def dominates(f, points):
            """Check if f dominates any point in points. Returns indices of dominated points."""
            if not points:
                return []
            points_arr = np.array(points)
            # f <= p (all objectives)
            cond1 = np.all(f <= points_arr, axis=1)
            # f != p
            cond2 = np.any(f < points_arr, axis=1)
            return np.where(cond1 & cond2)[0]
            
        # Populate initial non-dominated set
        for i in range(n_init):
            if not is_dominated(F[i], nd_values):
                # Remove any points dominated by F[i]
                dom_indices = dominates(F[i], nd_values)
                for idx in dom_indices:
                    nd_points.pop(idx)
                    nd_values.pop(idx)
                nd_points.append(X_init[i].copy())
                nd_values.append(F[i].copy())

        # 2. Refinement Phase
        n_refine = self.budget - used
        
        if n_refine > 0 and nd_points:
            # Select top candidates for refinement
            # Use the non-dominated points themselves as seeds for local search
            # Shuffle them to ensure diversity
            indices = list(range(len(nd_points)))
            rng.shuffle(indices)
            
            # We will distribute the remaining budget among the top 10 non-dominated points
            n_candidates = min(10, len(indices))
            evals_per_candidate = max(1, n_refine // n_candidates)
            
            for idx in indices[:n_candidates]:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                x_seed = nd_points[idx]
                
                for _ in range(evals_per_candidate):
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                        
                    # Local perturbation
                    # Step size relative to bounds
                    step_size = 0.05 * (hi - lo)
                    x_new = x_seed + rng.standard_normal(self.dim) * step_size
                    x_new = np.clip(x_new, lo, hi)
                    
                    f_new = problem(x_new)
                    used += 1
                    
                    # Update non-dominated set
                    if not is_dominated(f_new, nd_values):
                        dom_indices = dominates(f_new, nd_values)
                        for d_idx in dom_indices:
                            nd_points.pop(d_idx)
                            nd_values.pop(d_idx)
                        nd_points.append(x_new.copy())
                        nd_values.append(f_new.copy())
                        # Update seed to the new better point for subsequent iterations in this loop?
                        # Yes, moving towards the front is good.
                        x_seed = x_new

            # Use any remaining budget for random restarts to escape local optima
            while used < self.budget:
                if time.time() > deadline:
                    break
                x_rand = lo + rng.random(self.dim) * (hi - lo)
                f_rand = problem(x_rand)
                used += 1
                if not is_dominated(f_rand, nd_values):
                    dom_indices = dominates(f_rand, nd_values)
                    for d_idx in dom_indices:
                        nd_points.pop(d_idx)
                        nd_values.pop(d_idx)
                    nd_points.append(x_rand.copy())
                    nd_values.append(f_rand.copy())

        # Return the initial population (ignored by evaluator, but required by signature)
        return X_init
