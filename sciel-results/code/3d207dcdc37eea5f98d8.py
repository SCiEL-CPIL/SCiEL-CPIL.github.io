"""Multi-objective optimizer on RE21 using Sobol sampling and Nelder-Mead local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0
        B = int(self.budget)
        d = self.dim
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        
        # Reference point for hypervolume calculation
        ref = np.array([3000.0, 0.06])

        # Archive stores (x, f) for non-dominated points
        xs = []
        fs = []
        
        def dominates(f1, f2):
            """Returns True if f1 strictly dominates f2 (minimization)"""
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def update_archive(x_new, f_new):
            """
            Adds x_new, f_new to the archive if it is non-dominated.
            Removes any existing points dominated by the new point.
            """
            if len(fs) == 0:
                xs.append(x_new)
                fs.append(f_new)
                return True
            
            # Check if new point is dominated by any existing point
            for f_exist in fs:
                if dominates(f_exist, f_new):
                    return False
            
            # Remove any existing points dominated by the new point
            new_xs = []
            new_fs = []
            for i in range(len(fs)):
                if not dominates(f_new, fs[i]):
                    new_xs.append(xs[i])
                    new_fs.append(fs[i])
            
            new_xs.append(x_new)
            new_fs.append(f_new)
            
            xs.clear()
            fs.clear()
            for i in range(len(new_xs)):
                xs.append(new_xs[i])
                fs.append(new_fs[i])
            return True

        def hypervolume_2d(points, ref):
            """
            Calculate 2D hypervolume for a list of points.
            """
            if len(points) == 0:
                return 0.0
            p = np.array(points)
            # Filter points strictly better than ref
            mask = (p[:, 0] < ref[0]) & (p[:, 1] < ref[1])
            p = p[mask]
            if len(p) == 0:
                return 0.0
            
            # Sort by f1 ascending
            p = p[np.argsort(p[:, 0])]
            
            f1 = p[:, 0]
            f2 = p[:, 1]
            
            total_hv = 0.0
            for i in range(len(p)):
                x1 = f1[i]
                x2 = f1[i+1] if i < len(p)-1 else ref[0]
                y1 = f2[i]
                y2 = f2[i+1] if i < len(p)-1 else ref[1]
                
                w = max(0.0, x2 - x1)
                h = max(0.0, y1 - y2)
                total_hv += w * h
                
            return total_hv

        def marginal_hv_contribution(f_new, existing_fs, ref):
            """
            Calculate the marginal hypervolume contribution of f_new to the set existing_fs.
            """
            if len(existing_fs) == 0:
                if f_new[0] >= ref[0] or f_new[1] >= ref[1]:
                    return 0.0
                return (ref[0] - f_new[0]) * (ref[1] - f_new[1])
            
            # Check if dominated
            for f_exist in existing_fs:
                if dominates(f_exist, f_new):
                    return 0.0
            
            # Calculate HV of existing set
            hv_old = hypervolume_2d(existing_fs, ref)
            
            # Calculate HV of new set
            new_set = list(existing_fs) + [f_new]
            hv_new = hypervolume_2d(new_set, ref)
            
            return max(0.0, hv_new - hv_old)

        # Sobol-like initialization using Halton sequence for quasi-random coverage
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def sobol_like_sample(n, dim):
            """Generate n quasi-random points in [0,1]^dim using Halton sequences"""
            points = np.zeros((n, dim))
            for j in range(dim):
                base = j + 2  # 2, 3, 5, 7, ...
                for i in range(n):
                    points[i, j] = halton(i + 1, base)
            return points

        # Phase 1: Deterministic Sobol-like sampling for initial coverage
        n_init = min(100, B)
        if n_init > 0:
            init_points = sobol_like_sample(n_init, d)
            for i in range(n_init):
                if time.time() > deadline:
                    break
                x = lo + span * init_points[i]
                f = problem(x)
                f = np.array(f, dtype=np.float64)
                if np.all(np.isfinite(f)):
                    update_archive(x, f)
                B -= 1

        # Phase 2: Nelder-Mead local optimization on top hypervolume contributors
        if len(xs) > 0 and B > 0:
            # Compute marginal HV contributions for all archive points
            if len(xs) <= 100:
                contribs = []
                for i in range(len(xs)):
                    c = marginal_hv_contribution(fs[i], fs, ref)
                    contribs.append(c)
                contribs = np.array(contribs)
                order = np.argsort(-contribs)
            else:
                order = np.argsort(-np.array([fs[i][0] + fs[i][1] for i in range(len(fs))]))
            
            # Select top 10 points for local optimization
            n_polish = min(10, len(xs))
            
            for idx in order[:n_polish]:
                if B <= 0 or time.time() > deadline:
                    break
                
                x_start = xs[idx].copy()
                f_start = fs[idx].copy()
                
                # Define objective: weighted sum of objectives (epsilon-constraint style)
                # We'll try multiple weight vectors to cover the Pareto front
                # For each starting point, optimize with a few different weights
                weights_list = [
                    np.array([0.5, 0.5]),
                    np.array([0.7, 0.3]),
                    np.array([0.3, 0.7]),
                    np.array([0.9, 0.1]),
                    np.array([0.1, 0.9]),
                ]
                
                for w in weights_list:
                    if B <= 0 or time.time() > deadline:
                        break
                    
                    def objective(x):
                        nonlocal B
                        if B <= 0:
                            return 1e10
                        fx = problem(x)
                        fx = np.array(fx, dtype=np.float64)
                        B -= 1
                        if not np.all(np.isfinite(fx)):
                            return 1e10
                        # Weighted sum with scaling to balance magnitudes
                        # Scale f1 by 1/3000 and f2 by 1/0.06 to normalize
                        scaled = np.array([fx[0] / 3000.0, fx[1] / 0.06])
                        return np.dot(w, scaled)
                    
                    # Run Nelder-Mead with tight bounds via clipping
                    bounds_list = list(zip(lo, hi))
                    try:
                        result = minimize(
                            objective,
                            x_start,
                            method='Nelder-Mead',
                            bounds=bounds_list,
                            options={'maxiter': 30, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True}
                        )
                    except Exception:
                        continue
                    
                    if result.success and B > 0:
                        x_opt = np.clip(result.x, lo, hi)
                        f_opt = problem(x_opt)
                        f_opt = np.array(f_opt, dtype=np.float64)
                        B -= 1
                        if np.all(np.isfinite(f_opt)):
                            update_archive(x_opt, f_opt)

        # Phase 3: If budget remains, do a second round of Sobol sampling in promising regions
        if B > 0 and len(xs) > 0 and time.time() < deadline:
            # Identify the region around the best points and sample more densely
            # Use the centroid of the top points
            if len(xs) <= 100:
                contribs = []
                for i in range(len(xs)):
                    c = marginal_hv_contribution(fs[i], fs, ref)
                    contribs.append(c)
                contribs = np.array(contribs)
                order = np.argsort(-contribs)
            else:
                order = np.argsort(-np.array([fs[i][0] + fs[i][1] for i in range(len(fs))]))
            
            # Sample around the top 3 points with smaller step sizes
            n_extra = min(30, B)
            for idx in order[:3]:
                if B <= n_extra or time.time() > deadline:
                    break
                x_center = xs[idx].copy()
                # Local Halton sampling around this point
                for i in range(10):
                    if B <= 0 or time.time() > deadline:
                        break
                    # Small random perturbation
                    offset = np.random.random(d) - 0.5
                    x = x_center + offset * 0.1 * span
                    x = np.clip(x, lo, hi)
                    f = problem(x)
                    f = np.array(f, dtype=np.float64)
                    if np.all(np.isfinite(f)):
                        update_archive(x, f)
                    B -= 1

        if len(fs) > 0:
            return np.array(fs)
        else:
            return np.array([])

# EVOLVE-BLOCK-END
