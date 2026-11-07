import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
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

        # 1. Initialization: Halton sequence for diverse coverage
        init_size = 40
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

        # 2. Systematic Local Refinement Strategy
        # We will cycle through the Pareto front and try to improve each point
        # using a coordinate-descent-like local search.
        
        stagnation_count = 0
        max_stagnation = 10 # If no improvement in 10 full sweeps, do global jump
        
        while evals < self.budget and time_ok():
            if not archive_x:
                # Fallback to random if archive is empty
                x = lo + range_ * np.random.rand(self.dim)
                f = problem(x)
                evals += 1
                add_to_archive(x, f)
                continue

            improved_in_sweep = False
            
            # Iterate over a copy of the archive to allow modifications
            # We prioritize points that are "extreme" or have high HV contribution
            # For simplicity, we just iterate through all current non-dominated points
            current_archive_x = list(archive_x)
            current_archive_f = list(archive_f)
            
            for idx in range(len(current_archive_x)):
                if evals >= self.budget or not time_ok():
                    break
                
                x_parent = current_archive_x[idx]
                f_parent = current_archive_f[idx]
                
                # Try to improve x_parent using local search
                # Step size decreases as we approach the end of the budget
                progress = evals / self.budget
                base_step = 0.05 * (1.0 - 0.5 * progress)
                
                best_x = x_parent
                best_f = f_parent
                local_improved = False
                
                # Try perturbations in each dimension
                for d in range(self.dim):
                    if evals >= self.budget or not time_ok():
                        break
                    
                    # Positive step
                    x_try = best_x.copy()
                    step_size = base_step * range_[d]
                    x_try[d] += step_size
                    x_try = np.clip(x_try, lo, hi)
                    
                    if np.allclose(x_try, best_x):
                        continue
                        
                    f_try = problem(x_try)
                    evals += 1
                    
                    # Check if f_try dominates best_f or is non-dominated
                    if dominates(f_try, best_f):
                        best_x = x_try
                        best_f = f_try
                        local_improved = True
                    elif not is_dominated(f_try, archive_f) and not dominates(best_f, f_try):
                        # It's a new non-dominated point, but doesn't dominate current best
                        # We might still want to keep it in the archive, but for local search 
                        # we are trying to improve the specific lineage.
                        # For now, we only accept if it dominates the current best in this lineage
                        pass

                    # Negative step
                    if evals >= self.budget or not time_ok():
                        break
                        
                    x_try = best_x.copy()
                    x_try[d] -= step_size
                    x_try = np.clip(x_try, lo, hi)
                    
                    if np.allclose(x_try, best_x):
                        continue
                        
                    f_try = problem(x_try)
                    evals += 1
                    
                    if dominates(f_try, best_f):
                        best_x = x_try
                        best_f = f_try
                        local_improved = True
                    elif not is_dominated(f_try, archive_f) and not dominates(best_f, f_try):
                        pass

                if local_improved:
                    # Update the archive with the improved point
                    # Remove the old point if it is dominated by the new one
                    # add_to_archive handles dominance, but we need to ensure we replace the old one
                    # If best_x is the same as x_parent, no change.
                    # If best_x is different and dominates x_parent, add_to_archive will remove x_parent.
                    was_added = add_to_archive(best_x, best_f)
                    if was_added:
                        improved_in_sweep = True
                        stagnation_count = 0
                else:
                    # No improvement for this point
                    pass

            if not improved_in_sweep:
                stagnation_count += 1
                if stagnation_count >= max_stagnation:
                    # Global exploration jump
                    x = lo + range_ * np.random.rand(self.dim)
                    f = problem(x)
                    evals += 1
                    add_to_archive(x, f)
                    stagnation_count = 0
                    # Reset max_stagnation to allow more refinement cycles
                    max_stagnation = 10

        return archive_x
