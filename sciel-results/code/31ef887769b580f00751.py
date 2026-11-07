"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Multi-objective optimizer using Sobol exploration followed by hypervolume-guided local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hypervolume_contribution(self, point, archive, ref):
        """Compute the marginal hypervolume contribution of a point to the archive."""
        # Only count if point is strictly better than ref in all objectives
        if not np.all(point < ref):
            return 0.0
        
        # For 3-objective case, compute contribution
        # A point contributes hypervolume that would be lost if it were removed
        # Approximation: compute HV of archive with point vs without point
        # For efficiency, use a simplified approach
        
        n = len(archive)
        if n == 0:
            return 0.0
        
        # Compute HV of archive
        def hypervolume_3d(points, ref):
            if len(points) == 0:
                return 0.0
            # Sort by first objective
            sorted_pts = sorted(points, key=lambda p: p[0])
            hv = 0.0
            for i in range(len(sorted_pts)):
                p = sorted_pts[i]
                if not np.all(p < ref):
                    continue
                # Width in obj 1: from p[0] to next point's obj 1 or ref[0]
                if i + 1 < len(sorted_pts):
                    w1 = sorted_pts[i+1][0] - p[0]
                else:
                    w1 = ref[0] - p[0]
                if w1 <= 0:
                    continue
                # For obj 2 and 3, find the maximum extent
                # This is a simplification - for exact HV in 3D we need more work
                # Use a simpler approximation
                w2 = ref[1] - p[1]
                w3 = ref[2] - p[2]
                if w2 > 0 and w3 > 0:
                    hv += w1 * w2 * w3
            return hv
        
        # This is too slow for large archives. Use a different approach.
        # Instead, we'll use a greedy selection approach in the main loop.
        return 0.0

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s limit
        
        n = self.dim
        n_obj = self.n_obj
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Phase 1: Sobol quasi-random sampling for uniform coverage
        n_sobol = min(150, self.budget)
        
        # Generate Sobol sequence
        sobol_engine = qmc.Sobol(d=n, scramble=True)
        sobol_samples = sobol_engine.random(n_sobol)  # in [0, 1]^n
        
        # Map to bounds
        X_sobol = lo + (hi - lo) * sobol_samples
        
        # Evaluate all Sobol points
        all_X = []
        all_F = []
        for x in X_sobol:
            if time.time() > deadline - 5:
                break
            f = problem(x)
            all_X.append(x.copy())
            all_F.append(f.copy())
        
        used = len(all_X)
        
        # Phase 2: Hypervolume-guided local search
        # Maintain a non-dominated archive and refine it
        # For simplicity, we'll do local search around the best points
        
        if used < self.budget:
            ref = np.array([1705.0, 12.0, 0.3])
            
            # Find non-dominated points
            F_arr = np.array(all_F)
            X_arr = np.array(all_X)
            
            dominated = np.zeros(len(F_arr), dtype=bool)
            for i in range(len(F_arr)):
                for j in range(len(F_arr)):
                    if i != j and np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated[i] = True
                        break
            
            nondom_idx = np.where(~dominated)[0]
            
            if len(nondom_idx) == 0:
                nondom_idx = np.array([0])
            
            # Local search around non-dominated points
            rng = np.random.default_rng(42)
            
            # Sort non-dominated points by their sum of objectives (ascending)
            sums = F_arr[nondom_idx].sum(axis=1)
            sorted_idx = nondom_idx[np.argsort(sums)]
            
            for target_idx in sorted_idx:
                if used >= self.budget:
                    break
                if time.time() > deadline - 5:
                    break
                
                parent = X_arr[target_idx]
                parent_f = F_arr[target_idx]
                
                # Try several perturbations around this point
                n_trials = min(3, self.budget - used)
                for _ in range(n_trials):
                    if used >= self.budget:
                        break
                    if time.time() > deadline - 5:
                        break
                    
                    # Gaussian perturbation with decreasing step size
                    progress = used / self.budget
                    step_size = 0.2 * (1 - 0.5 * progress)
                    
                    child = parent + step_size * (hi - lo) * rng.standard_normal(n)
                    child = np.clip(child, lo, hi)
                    
                    fc = problem(child)
                    used += 1
                    
                    all_X.append(child.copy())
                    all_F.append(fc.copy())
                    X_arr = np.array(all_X)
                    F_arr = np.array(all_F)
        
        return all_X
# EVOLVE-BLOCK-END
