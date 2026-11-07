"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """LHS Init + Vectorized Archive + Multi-start Nelder-Mead Polish."""

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
        
        # 1. LHS Initialization for maximum initial spread
        # Use 25% of budget for initialization, min 10 points
        n_init = max(10, self.budget // 4)
        
        # Generate LHS points
        # For each dimension, create a permutation of indices 0..n_init-1
        # This ensures exactly one point in each slice of the hypercube per dimension
        lhs_points = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            perm = np.arange(n_init)
            np.random.shuffle(perm) # Using global np.random as evaluator seeds it
            # Map indices to [0, 1) then to [lo, hi)
            # Using (perm + 0.5) / n_init for center of each slice
            lhs_points[:, d] = (perm + 0.5) / n_init
        
        X_init = lo + lhs_points * span
        F_init = np.array([problem(x) for x in X_init])
        used = n_init
        
        # Maintain a list of (x, f) pairs for the archive
        # We will keep all non-dominated points found so far
        # To be efficient, we store them in a list and filter periodically or on insertion
        # Given budget 200, a list of up to 200 points is manageable for O(N^2) checks if optimized,
        # but let's use a simple list and filter.
        
        archive_X = list(X_init)
        archive_F = list(F_init)
        
        # Helper to check if a new point is dominated by any in archive
        # If yes, don't add. If it dominates existing ones, remove them.
        def add_to_archive(x, f):
            nonlocal archive_X, archive_F
            
            # Check if f is dominated by any point in archive
            # archive_F is list of arrays
            # Vectorized check: for each a in archive_F, is a <= f and a != f?
            # Since N is small (<200), simple loop is fast enough in Python if we avoid overhead
            # But let's try to be clever.
            
            dominated = False
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    dominated = True
                    break
            
            if dominated:
                return False
            
            # Remove points dominated by f
            new_archive_X = []
            new_archive_F = []
            for i, (ax, af) in enumerate(zip(archive_X, archive_F)):
                # Is af dominated by f?
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            
            new_archive_X.append(x)
            new_archive_F.append(f)
            
            archive_X = new_archive_X
            archive_F = new_archive_F
            return True

        # 2. Exploration Phase: Random sampling with archive update
        # Use remaining budget for random sampling to fill gaps
        # We will stop exploration when we have enough points or time runs out
        # Let's allocate 50% of remaining budget to random exploration
        remaining_budget = self.budget - used
        n_explore = int(remaining_budget * 0.5)
        
        for _ in range(n_explore):
            if time.time() > deadline:
                break
            x = lo + span * np.random.random(self.dim)
            f = problem(x)
            used += 1
            add_to_archive(x, f)
            
        # 3. Exploitation Phase: Multi-start Nelder-Mead on Pareto Front
        # Take the current archive (Pareto front) and try to improve each point
        # using Nelder-Mead.
        # We have used 'used' evaluations. We have self.budget - used left.
        # Each Nelder-Mead run on 5 vars takes ~15-20 evals.
        # We can run it on a subset of the front if budget is tight.
        
        # Sort archive by sum of objectives to prioritize "good" points
        if len(archive_X) > 0:
            # Convert to arrays for sorting
            archive_X_arr = np.array(archive_X)
            archive_F_arr = np.array(archive_F)
            
            # Sort by sum of objectives
            sums = np.sum(archive_F_arr, axis=1)
            sorted_idx = np.argsort(sums)
            
            # We will try to improve the best points first
            # Limit number of points to improve to fit budget
            # Estimate evals per NM run: dim * 2 + 5 = 15 roughly
            evals_per_nm = self.dim * 2 + 5
            max_nm_runs = max(1, (self.budget - used) // evals_per_nm)
            
            # Select top points for polishing
            points_to_polish = sorted_idx[:max_nm_runs]
            
            for idx in points_to_polish:
                if time.time() > deadline:
                    break
                if used >= self.budget:
                    break
                    
                x0 = archive_X_arr[idx]
                
                # Run Nelder-Mead on the sum of objectives (scalarization)
                # This is a heuristic to find local optima in the multi-objective space
                def objective_scalar(x):
                    # Check budget
                    if used >= self.budget:
                        return 1e9
                    fx = problem(x)
                    used += 1
                    return np.sum(fx)
                    
                try:
                    from scipy.optimize import minimize
                    res = minimize(objective_scalar, x0, method='Nelder-Mead', 
                                   options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 20, 'maxfev': 15})
                    
                    # If the result is better, add to archive
                    if res.success or res.fun < np.sum(archive_F_arr[idx]):
                        # We need the actual function values for the archive
                        # res.fun is the scalar sum. We need the vector.
                        # We don't have it directly from minimize.
                        # We must evaluate the final point again?
                        # No, we can just add x0 if it improved, but we need F.
                        # To save budget, we assume the last evaluation in minimize was the best.
                        # However, minimize doesn't return the vector F.
                        # We will do one extra evaluation to get the vector for the best point found.
                        # This costs 1 eval.
                        if used < self.budget:
                            x_best = res.x
                            f_best = problem(x_best)
                            used += 1
                            add_to_archive(x_best, f_best)
                except Exception:
                    pass

        # Return the archive (ignored by evaluator, but good practice)
        return archive_X

# EVOLVE-BLOCK-END
