import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        used = [0]  # Mutable counter for nested functions
        archive_X = []
        archive_F = []
        
        def add_to_archive(x, f):
            f = np.array(f)
            # Check if dominated by any existing point in archive
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
            
            archive_X.clear()
            archive_F.clear()
            archive_X.extend(new_archive_X)
            archive_F.extend(new_archive_F)
            return True

        def safe_eval(x):
            if used[0] >= self.budget:
                return None
            try:
                f = problem(x)
                used[0] += 1
                return f
            except Exception:
                return None

        # Phase 1: Latin Hypercube Sampling for initial exploration
        n_init = max(10, min(20, self.budget // 10))
        
        lhs_points = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            perm = np.arange(n_init)
            np.random.shuffle(perm)
            lhs_points[:, d] = (perm + 0.5) / n_init
        
        X_init = lo + lhs_points * (hi - lo)
        
        for x in X_init:
            if time.time() > deadline or used[0] >= self.budget:
                break
            f = safe_eval(x)
            if f is not None:
                add_to_archive(x, f)
        
        # Phase 2: Adaptive Tchebycheff-based local search
        # Generate weight vectors on the simplex
        n_weights = 12
        # Vertices (3)
        weights_list = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            # Edge midpoints (3)
            np.array([0.5, 0.5, 0.0]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.0, 0.5, 0.5]),
            # Center (1)
            np.array([1.0/3, 1.0/3, 1.0/3]),
            # Additional structured weights (5)
            np.array([0.7, 0.2, 0.1]),
            np.array([0.2, 0.7, 0.1]),
            np.array([0.1, 0.2, 0.7]),
            np.array([0.6, 0.3, 0.1]),
            np.array([0.3, 0.5, 0.2]),
        ]
        
        # Allocate budget for local optimization
        remaining_budget = self.budget - used[0]
        
        # Use multiple starts per weight for better convergence
        n_starts = 2
        maxfev_per_run = max(1, remaining_budget // (n_weights * n_starts))
        
        # Reference point for Tchebycheff: use nadir from archive or bounds
        def get_reference_point():
            if archive_F:
                return np.max(archive_F, axis=0)
            else:
                return np.array([1705.0, 12.0, 0.3])
        
        def get_ideal_point():
            if archive_F:
                return np.min(archive_F, axis=0)
            else:
                return np.array([0.0, 0.0, 0.0])
        
        for w in weights_list:
            if time.time() > deadline or used[0] >= self.budget:
                break
            
            ref_point = get_reference_point()
            ideal_point = get_ideal_point()
            
            # Normalize weights for Tchebycheff
            # Use range-based normalization
            ranges = ref_point - ideal_point
            ranges = np.where(ranges > 1e-10, ranges, 1.0)
            lambda_t = w / ranges
            
            for start_idx in range(n_starts):
                if time.time() > deadline or used[0] >= self.budget:
                    break
                
                # Multi-start: use different initial points
                if start_idx == 0:
                    x0 = (lo + hi) / 2.0
                else:
                    # Start near an archive point if available
                    if archive_X and len(archive_X) > 0:
                        idx = np.random.randint(len(archive_X))
                        x0 = archive_X[idx].copy()
                        # Add small perturbation
                        x0 = x0 + np.random.uniform(-0.1, 0.1, self.dim)
                        x0 = np.clip(x0, lo, hi)
                    else:
                        x0 = lo + np.random.uniform(0, 1, self.dim) * (hi - lo)
                
                def tcheb_obj(x, lambda_t=lambda_t, ideal_point=ideal_point):
                    if used[0] >= self.budget:
                        return 1e10
                    f = problem(x)
                    used[0] += 1
                    # Tchebycheff scalarization: max_i lambda_i * (f_i - ideal_i)
                    return np.max(lambda_t * (f - ideal_point))
                
                try:
                    res = minimize(
                        tcheb_obj, 
                        x0, 
                        method='L-BFGS-B', 
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 8, 'maxfev': maxfev_per_run, 'ftol': 1e-6}
                    )
                    
                    # Evaluate the best point found
                    if used[0] < self.budget:
                        x_best = res.x
                        f_best = safe_eval(x_best)
                        if f_best is not None:
                            add_to_archive(x_best, f_best)
                            
                except Exception:
                    pass

        # Phase 3: Final Polish on archive points with sum objective
        if archive_F and used[0] < self.budget:
            for i, (ax, af) in enumerate(zip(archive_X, archive_F)):
                if time.time() > deadline or used[0] >= self.budget:
                    break
                
                x0 = ax.copy()
                
                def sum_obj(x):
                    if used[0] >= self.budget:
                        return 1e10
                    f = problem(x)
                    used[0] += 1
                    return np.sum(f)
                
                try:
                    res = minimize(
                        sum_obj, 
                        x0, 
                        method='L-BFGS-B', 
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 5, 'maxfev': 3, 'ftol': 1e-8, 'gtol': 1e-8}
                    )
                    if used[0] < self.budget:
                        x_best = res.x
                        f_best = safe_eval(x_best)
                        if f_best is not None:
                            add_to_archive(x_best, f_best)
                except Exception:
                    pass

        return archive_X
