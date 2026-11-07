import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin: evaluator kills at 330s
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
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
            if is_dominated(f, archive_f):
                return False
            
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

        def compute_hv_2d(f_list, ref_point):
            if not f_list:
                return 0.0
            pts = [f for f in f_list if np.all(f < ref_point)]
            if not pts:
                return 0.0
            pts = sorted(pts, key=lambda x: x[0])
            hv = 0.0
            for i in range(len(pts)):
                f1, f2 = pts[i]
                next_f1 = pts[i+1][0] if i < len(pts) - 1 else ref_point[0]
                width = next_f1 - f1
                height = ref_point[1] - f2
                if width > 0 and height > 0:
                    hv += width * height
            return hv

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

        def local_search(x0, f0, cell_lo, cell_hi, max_iters=15):
            """Local simplex search within a cell to improve a point."""
            nonlocal evals
            if evals >= self.budget or not time_ok():
                return x0, f0
            
            # Initialize simplex
            alpha = 0.1
            simplex = [x0.copy()]
            obj_values = [f0.copy()]
            
            for d in range(self.dim):
                x_new = x0.copy()
                step = (cell_hi[d] - cell_lo[d]) * alpha
                x_new[d] = x0[d] + step
                x_new = np.clip(x_new, cell_lo, cell_hi)
                simplex.append(x_new)
                
                if evals >= self.budget or not time_ok():
                    break
                f_new = problem(x_new)
                evals += 1
                obj_values.append(f_new)
            
            # Simplex iterations
            for _ in range(max_iters):
                if evals >= self.budget or not time_ok():
                    break
                if len(simplex) < 2:
                    break
                
                # Sort by sum of objectives
                indices = sorted(range(len(simplex)), key=lambda i: np.sum(obj_values[i]))
                simplex = [simplex[i] for i in indices]
                obj_values = [obj_values[i] for i in indices]
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = obj_values[-1]
                
                # Reflection
                x_reflect = centroid + (centroid - worst)
                x_reflect = np.clip(x_reflect, cell_lo, cell_hi)
                
                if evals >= self.budget or not time_ok():
                    break
                f_reflect = problem(x_reflect)
                evals += 1
                
                # Accept if better than worst
                if np.all(f_reflect <= worst_f) and np.any(f_reflect < worst_f):
                    simplex[-1] = x_reflect
                    obj_values[-1] = f_reflect
                else:
                    # Try contraction
                    x_contract = centroid + 0.5 * (worst - centroid)
                    x_contract = np.clip(x_contract, cell_lo, cell_hi)
                    
                    if evals >= self.budget or not time_ok():
                        break
                    f_contract = problem(x_contract)
                    evals += 1
                    
                    if np.all(f_contract <= worst_f) and np.any(f_contract < worst_f):
                        simplex[-1] = x_contract
                        obj_values[-1] = f_contract
                    else:
                        # Shrink towards best
                        best = simplex[0]
                        for i in range(1, len(simplex)):
                            simplex[i] = best + 0.5 * (simplex[i] - best)
                            simplex[i] = np.clip(simplex[i], cell_lo, cell_hi)
                            if evals >= self.budget or not time_ok():
                                break
                            f_new = problem(simplex[i])
                            evals += 1
                            obj_values[i] = f_new
            
            # Return best point
            best_idx = np.argmin([np.sum(f) for f in obj_values])
            return simplex[best_idx], obj_values[best_idx]

        # Phase 1: Halton initialization for better coverage
        init_size = 30
        for i in range(1, init_size + 1):
            if evals >= self.budget or not time_ok():
                break
            p = halton_point(i, self.dim)
            x = lo + range_ * p
            f = problem(x)
            evals += 1
            add_to_archive(x, f)
        
        if not time_ok():
            return archive_x

        # Phase 2: Main search loop with adaptive exploration/exploitation
        best_hv_so_far = compute_hv_2d(archive_f, ref_point)
        last_improvement_eval = evals
        
        while evals < self.budget and time_ok():
            # Decide whether to explore or exploit
            no_improvement = evals - last_improvement_eval
            
            if no_improvement > 50:
                # Force exploration
                x = lo + range_ * np.random.rand(self.dim)
            elif archive_x and np.random.rand() < 0.7:
                # Exploit: pick a random point from archive
                idx = np.random.randint(len(archive_x))
                parent = archive_x[idx]
                # Adaptive step size
                progress = evals / self.budget
                step_scale = 0.1 * (1.0 - 0.5 * progress)
                step = step_scale * range_
                x = parent + step * np.random.normal(size=self.dim)
                x = np.clip(x, lo, hi)
            else:
                # Moderate exploration
                if np.random.rand() < 0.5:
                    x = lo + range_ * np.random.rand(self.dim)
                else:
                    x = mid + 0.2 * range_ * np.random.normal(size=self.dim)
                    x = np.clip(x, lo, hi)
            
            f_child = problem(x)
            evals += 1
            
            was_added = add_to_archive(x, f_child)
            
            if was_added:
                current_hv = compute_hv_2d(archive_f, ref_point)
                if current_hv > best_hv_so_far:
                    best_hv_so_far = current_hv
                    last_improvement_eval = evals

        # Phase 3: Local polish on archive points to sharpen the front
        if evals < self.budget and time_ok() and archive_x:
            # Sort archive by hypervolume contribution (approximate by sum of objectives)
            sorted_indices = sorted(range(len(archive_f)), key=lambda i: np.sum(archive_f[i]))
            
            for idx in sorted_indices:
                if evals >= self.budget or not time_ok():
                    break
                
                x_parent = archive_x[idx].copy()
                f_parent = archive_f[idx].copy()
                
                # Determine local region around this point
                local_radius = 0.05 * range_
                local_lo = np.clip(x_parent - local_radius, lo, hi)
                local_hi = np.clip(x_parent + local_radius, lo, hi)
                
                # Fine local search
                x_fine, f_fine = local_search(x_parent, f_parent, local_lo, local_hi, max_iters=10)
                add_to_archive(x_fine, f_fine)
        
        return archive_x
