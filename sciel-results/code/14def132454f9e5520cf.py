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
        deadline = start_time + 290.0  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        range_ = hi - lo
        mid = (lo + hi) / 2.0
        
        archive_x = []
        archive_f = []
        evals = 0
        ref_point = np.array([3000.0, 0.06])
        
        def time_ok():
            return time.time() < deadline

        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        def is_dominated(f_new, archive_f_list):
            if len(archive_f_list) == 0:
                return False
            for f in archive_f_list:
                if dominates(f, f_new):
                    return True
            return False

        def add_to_archive(x, f):
            nonlocal evals
            # Check if dominated by existing
            if is_dominated(f, archive_f):
                return False
            
            # Remove points dominated by new
            new_archive_x = []
            new_archive_f = []
            for xi, fi in zip(archive_x, archive_f):
                if dominates(f, fi):
                    continue
                new_archive_x.append(xi)
                new_archive_f.append(fi)
                
            new_archive_x.append(x)
            new_archive_f.append(f)
            
            archive_x[:] = new_archive_x
            archive_f[:] = new_archive_f
            return True

        def halton_sequence(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def halton_point(n, dim):
            p = np.zeros(dim)
            primes = [2, 3, 5, 7]
            for d in range(dim):
                p[d] = halton_sequence(n, primes[d])
            return p

        def compute_hv_2d(points, ref):
            """Compute hypervolume for 2D points relative to ref point."""
            if len(points) == 0:
                return 0.0
            
            # Filter points that are better than ref in all objectives
            valid = [p for p in points if p[0] < ref[0] and p[1] < ref[1]]
            if len(valid) == 0:
                return 0.0
            
            # Sort by first objective (ascending)
            valid.sort(key=lambda x: x[0])
            
            hv = 0.0
            max_y = -np.inf
            for i, p in enumerate(valid):
                # The contribution is (ref[1] - p[1]) * (next_x - p[0])
                # where next_x is the x of the next point or ref[0]
                next_x = valid[i+1][0] if i+1 < len(valid) else ref[0]
                if p[1] > max_y:
                    max_y = p[1]
                    hv += (ref[1] - max_y) * (next_x - p[0])
            
            return hv

        def local_simplex_search(x_start, f_start, weights, max_iter=30):
            """Local Nelder-Mead search on weighted sum objective."""
            nonlocal evals
            
            def objective(x):
                nonlocal evals
                if evals >= self.budget:
                    return np.inf
                f = problem(x)
                evals += 1
                return np.dot(weights, f)
            
            # Set up bounds for scipy
            bounds_scipy = list(zip(lo, hi))
            
            try:
                result = minimize(objective, x_start, method='Nelder-Mead',
                                  bounds=bounds_scipy,
                                  options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-6})
                return result.x, result.fun
            except:
                return x_start, f_start

        # 1. Initialization: Halton sequence for diverse coverage
        init_size = 30
        for i in range(1, init_size + 1):
            if not time_ok():
                break
            p = halton_point(i, self.dim)
            x = lo + range_ * p
            f = problem(x)
            evals += 1
            add_to_archive(x, f)
        
        if not time_ok():
            return archive_x

        # 2. Decomposition-based optimization with weighted sum + local search
        # Generate a set of weight vectors to cover the Pareto front
        n_weights = 8
        weights_list = []
        for i in range(n_weights):
            if i == 0:
                w = np.array([1.0, 0.0])
            elif i == n_weights - 1:
                w = np.array([0.0, 1.0])
            else:
                # Use a spread of weights, emphasizing intermediate trade-offs
                t = i / (n_weights - 1)
                # Use a non-uniform spread to get more points in the middle
                t_adj = 0.5 - 0.5 * np.cos(np.pi * t)
                w = np.array([1.0 - t_adj, t_adj])
            weights_list.append(w)
        
        # For each weight vector, do local search from multiple starting points
        # and add the best result to the archive
        
        for w in weights_list:
            if evals >= self.budget or not time_ok():
                break
            
            # Start from the point in archive that minimizes the weighted sum
            if archive_f:
                archive_f_arr = np.array(archive_f)
                weighted_sums = archive_f_arr @ w
                best_idx = np.argmin(weighted_sums)
                x_start = archive_x[best_idx].copy()
                f_start = archive_f[best_idx].copy()
            else:
                x_start = mid.copy()
                f_start = problem(x_start)
                evals += 1
                add_to_archive(x_start, f_start)
            
            # Do local search from this starting point
            if evals >= self.budget or not time_ok():
                break
            
            x_opt, f_opt_val = local_simplex_search(x_start, f_start, w, max_iter=25)
            
            if evals < self.budget and time_ok():
                # Evaluate the final point
                f_opt = problem(x_opt)
                evals += 1
                add_to_archive(x_opt, f_opt)
        
        # 3. Final refinement: cycle through archive and do fine local search
        if evals < self.budget and time_ok():
            # Get current archive
            current_archive_x = list(archive_x)
            current_archive_f = list(archive_f)
            
            for idx in range(len(current_archive_x)):
                if evals >= self.budget or not time_ok():
                    break
                
                x_parent = current_archive_x[idx]
                f_parent = current_archive_f[idx]
                
                # Try to improve using small perturbations in each dimension
                progress = evals / self.budget
                base_step = 0.02 * (1.0 - 0.5 * progress)
                
                best_x = x_parent.copy()
                best_f = f_parent.copy()
                
                for d in range(self.dim):
                    if evals >= self.budget or not time_ok():
                        break
                    
                    # Try both directions
                    for direction in [1, -1]:
                        if evals >= self.budget or not time_ok():
                            break
                        
                        x_try = best_x.copy()
                        step_size = base_step * range_[d]
                        x_try[d] += direction * step_size
                        x_try = np.clip(x_try, lo, hi)
                        
                        if np.allclose(x_try, best_x):
                            continue
                        
                        f_try = problem(x_try)
                        evals += 1
                        
                        if dominates(f_try, best_f):
                            best_x = x_try
                            best_f = f_try
                        elif not is_dominated(f_try, archive_f) and not dominates(best_f, f_try):
                            # Add non-dominated points to archive
                            add_to_archive(x_try, f_try)
                
                if not np.allclose(best_x, x_parent):
                    add_to_archive(best_x, best_f)

        return archive_x
