"""Multi-objective optimizer on RE21 using Sobol sampling + Nelder-Mead refinement."""
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
        deadline = start_time + 280.0  # Safety margin under 330s kill
        B = int(self.budget)
        d = self.dim
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        
        # Reference point for hypervolume calculation (as per task)
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
            Returns True if the point was added to the archive.
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

        def hypervolume_2d(points, ref_pt):
            """
            Calculate 2D hypervolume for a list of non-dominated points.
            """
            if len(points) == 0:
                return 0.0
            p = np.array(points)
            # Filter points strictly better than ref
            mask = (p[:, 0] < ref_pt[0]) & (p[:, 1] < ref_pt[1])
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
                x2 = f1[i+1] if i < len(p)-1 else ref_pt[0]
                y1 = f2[i]
                y2 = f2[i+1] if i < len(p)-1 else ref_pt[1]
                
                w = max(0.0, x2 - x1)
                h = max(0.0, y1 - y2)
                total_hv += w * h
                
            return total_hv

        def marginal_hv_contribution(idx):
            """Compute the marginal HV contribution of point idx in the archive."""
            if len(fs) <= 1:
                return hypervolume_2d([fs[idx]], ref)
            
            # HV with the point
            hv_with = hypervolume_2d(fs, ref)
            # HV without the point
            fs_without = fs[:idx] + fs[idx+1:]
            hv_without = hypervolume_2d(fs_without, ref)
            return max(0.0, hv_with - hv_without)

        # Generate Sobol sequence points
        # Use scipy's Sobol sequence for quasi-random sampling
        n_sobol = min(128, B)
        from scipy.stats import qmc
        sobol = qmc.Sobol(d=d, scramble=True)
        sobol_points = sobol.random(n=n_sobol)
        
        # Evaluate Sobol points
        for i in range(n_sobol):
            if B <= 0 or time.time() > deadline:
                break
            x = lo + span * sobol_points[i]
            f = problem(x)
            f = np.array(f, dtype=np.float64)
            if np.all(np.isfinite(f)):
                update_archive(x, f)
                B -= 1

        # Nelder-Mead refinement on top HV contributors
        if len(xs) > 0 and B > 0 and time.time() < deadline:
            # Get top 10 points by marginal HV contribution
            n_top = min(len(xs), 10)
            contribs = []
            for i in range(len(xs)):
                c = marginal_hv_contribution(i)
                contribs.append((c, i))
            contribs.sort(reverse=True)
            top_indices = [idx for _, idx in contribs[:n_top]]
            
            # For each top point, run Nelder-Mead to optimize each objective
            # We'll create a combined objective that balances both objectives
            for idx in top_indices:
                if B <= 0 or time.time() > deadline:
                    break
                
                x0 = xs[idx].copy()
                
                # Try to optimize objective 1 primarily
                def obj1(x):
                    nonlocal B
                    if B <= 0 or time.time() > deadline:
                        return 1e10
                    fx = problem(x)
                    fx = np.array(fx, dtype=np.float64)
                    if not np.all(np.isfinite(fx)):
                        return 1e10
                    B -= 1
                    return fx[0]
                
                # Run Nelder-Mead with limited evaluations
                try:
                    res1 = minimize(obj1, x0, method='Nelder-Mead', 
                                   bounds=list(zip(lo, hi)),
                                   options={'maxiter': 15, 'xatol': 1e-6, 'fatol': 1e-6})
                    if res1.success and B > 0:
                        x_opt1 = np.clip(res1.x, lo, hi)
                        f_opt1 = problem(x_opt1)
                        f_opt1 = np.array(f_opt1, dtype=np.float64)
                        if np.all(np.isfinite(f_opt1)):
                            update_archive(x_opt1, f_opt1)
                            B -= 1
                except:
                    pass
                
                # Try to optimize objective 2 primarily
                if B <= 0 or time.time() > deadline:
                    continue
                    
                def obj2(x):
                    nonlocal B
                    if B <= 0 or time.time() > deadline:
                        return 1e10
                    fx = problem(x)
                    fx = np.array(fx, dtype=np.float64)
                    if not np.all(np.isfinite(fx)):
                        return 1e10
                    B -= 1
                    return fx[1]
                
                try:
                    res2 = minimize(obj2, x0, method='Nelder-Mead',
                                   bounds=list(zip(lo, hi)),
                                   options={'maxiter': 15, 'xatol': 1e-6, 'fatol': 1e-6})
                    if res2.success and B > 0:
                        x_opt2 = np.clip(res2.x, lo, hi)
                        f_opt2 = problem(x_opt2)
                        f_opt2 = np.array(f_opt2, dtype=np.float64)
                        if np.all(np.isfinite(f_opt2)):
                            update_archive(x_opt2, f_opt2)
                            B -= 1
                except:
                    pass
        
        # Return the archive
        if len(fs) > 0:
            return np.array(fs)
        else:
            return np.array([])

# EVOLVE-BLOCK-END
