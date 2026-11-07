"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Weighted-Sum Grid Search with L-BFGS-B Local Refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        center = (lo + hi) / 2.0
        
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
            new_archive_X = []
            new_archive_F = []
            for i, (ax, af) in enumerate(zip(archive_X, archive_F)):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_archive_X.append(ax)
                    new_archive_F.append(af)
            
            new_archive_X.append(x)
            new_archive_F.append(f)
            
            archive_X = new_archive_X
            archive_F = new_archive_F
            return True

        # Phase 1: Initial Exploration with LHS to seed the archive
        n_init = 30
        lhs_points = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            perm = np.arange(n_init)
            np.random.shuffle(perm)
            lhs_points[:, d] = (perm + 0.5) / n_init
        
        X_init = lo + lhs_points * span
        for x in X_init:
            if time.time() > deadline or used >= self.budget:
                break
            f = problem(x)
            used += 1
            add_to_archive(x, f)
        
        # Phase 2: Systematic Weighted Sum Optimization
        # Generate weight vectors on the simplex for 3 objectives
        # We use a set of representative weights to cover the front
        weights = []
        # Add vertices
        weights.append(np.array([1.0, 0.0, 0.0]))
        weights.append(np.array([0.0, 1.0, 0.0]))
        weights.append(np.array([0.0, 0.0, 1.0]))
        # Add midpoints of edges
        weights.append(np.array([0.5, 0.5, 0.0]))
        weights.append(np.array([0.5, 0.0, 0.5]))
        weights.append(np.array([0.0, 0.5, 0.5]))
        # Add center
        weights.append(np.array([1.0/3, 1.0/3, 1.0/3]))
        
        # Add a few random weights to ensure coverage if the front is non-convex
        for _ in range(10):
            w = np.random.random(3)
            w = w / np.sum(w)
            weights.append(w)
            
        weights = np.array(weights)
        
        # For each weight, run L-BFGS-B
        # Budget remaining: self.budget - used
        # Let's allocate ~10 evals per weight run
        maxfev_per_run = 10
        
        for w in weights:
            if time.time() > deadline or used >= self.budget:
                break
            
            # Start from a random point or the center
            # Using center is often good for smooth problems, but random helps escape local minima
            # Let's use a point from the current archive if available, else center
            if archive_X:
                # Pick a random point from the archive as start
                start_idx = np.random.randint(len(archive_X))
                x0 = archive_X[start_idx].copy()
            else:
                x0 = center.copy()
            
            def weighted_objective(x):
                nonlocal used
                if used >= self.budget:
                    return 1e10
                fx = problem(x)
                used += 1
                return np.dot(w, fx)
            
            try:
                res = minimize(
                    weighted_objective, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': 5, 'maxfev': maxfev_per_run, 'ftol': 1e-4}
                )
                
                # Evaluate the best point found by the optimizer to get the vector F
                # Note: minimize might have used all budget, check used
                if used < self.budget:
                    x_best = res.x
                    f_best = problem(x_best)
                    used += 1
                    add_to_archive(x_best, f_best)
                else:
                    # If we ran out of budget during minimize, we might have missed adding the last point
                    # But minimize calls the objective which increments used.
                    # We need to be careful not to exceed budget.
                    pass
                    
            except Exception:
                pass

        # Phase 3: Final Polish on the best points
        # If we have budget left, do a quick local search on the top 5 archive points
        if archive_F and used < self.budget:
            archive_F_arr = np.array(archive_F)
            sums = np.sum(archive_F_arr, axis=1)
            sorted_idx = np.argsort(sums)
            
            # Pick top 5
            top_indices = sorted_idx[:5]
            
            for idx in top_indices:
                if time.time() > deadline or used >= self.budget:
                    break
                
                x0 = archive_X[idx].copy()
                
                def sum_objective(x):
                    nonlocal used
                    if used >= self.budget:
                        return 1e10
                    fx = problem(x)
                    used += 1
                    return np.sum(fx)
                
                try:
                    res = minimize(
                        sum_objective, 
                        x0, 
                        method='L-BFGS-B', 
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 3, 'maxfev': 5, 'ftol': 1e-5}
                    )
                    if used < self.budget:
                        x_best = res.x
                        f_best = problem(x_best)
                        used += 1
                        add_to_archive(x_best, f_best)
                except Exception:
                    pass

        return archive_X

# EVOLVE-BLOCK-END
