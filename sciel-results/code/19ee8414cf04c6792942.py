import time
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Evaluation counter
        self.evals = 0
        self.max_budget = self.budget
        
        def eval_func(x):
            if self.evals >= self.max_budget:
                return np.array([1e6, 1e6, 1e6])
            self.evals += 1
            try:
                f = np.array(problem(x), dtype=float)
                if not np.all(np.isfinite(f)):
                    f = np.array([1e6, 1e6, 1e6])
            except:
                f = np.array([1e6, 1e6, 1e6])
            return f

        # Phase 1: Latin Hypercube Sampling
        n_lhs = 100
        if n_lhs > self.budget:
            n_lhs = self.budget
            
        # Generate LHS points
        # Simple LHS implementation
        lhs_points = np.zeros((n_lhs, self.dim))
        for i in range(n_lhs):
            for j in range(self.dim):
                # Random permutation for each dimension
                # To be efficient, we can generate permutations once
                pass
                
        # More efficient LHS generation
        perms = [np.random.permutation(n_lhs) for _ in range(self.dim)]
        for i in range(n_lhs):
            for j in range(self.dim):
                u = (perms[j][i] + np.random.random()) / n_lhs
                lhs_points[i, j] = lo[j] + u * (hi[j] - lo[j])

        archive = []
        archive_f = []
        
        for i in range(n_lhs):
            if time.time() > deadline:
                break
            x = lhs_points[i]
            f = eval_func(x)
            archive.append(x)
            archive_f.append(f)

        # Filter dominated points from archive
        def is_dominated(f, Fs):
            # Returns True if f is dominated by any in Fs
            if len(Fs) == 0:
                return False
            Fs = np.array(Fs)
            le = np.all(Fs <= f, axis=1)
            lt = np.any(Fs < f, axis=1)
            return bool(np.any(le & lt))

        # Keep non-dominated
        new_archive_x = []
        new_archive_f = []
        for i in range(len(archive_f)):
            # Check if dominated by other archive points
            others_f = [archive_f[j] for j in range(len(archive_f)) if j != i]
            if not is_dominated(archive_f[i], others_f):
                new_archive_x.append(archive_x[i] if 'archive_x' in locals() else archive[i])
                new_archive_f.append(archive_f[i])
        
        archive_x = new_archive_x
        archive_f = new_archive_f

        # Phase 2: Local Refinement
        # Select top K points based on sum of objectives (proxy for proximity to ideal)
        # Ideally, we want points that are close to the Pareto front.
        # Minimizing sum of objectives is a good heuristic.
        if archive_f:
            sums = np.sum(np.array(archive_f), axis=1)
            # Get indices of best 10
            k_refine = min(10, len(archive_f))
            best_indices = np.argsort(sums)[:k_refine]
            
            remaining_budget = self.budget - self.evals
            evals_per_point = max(1, remaining_budget // k_refine)
            
            for idx in best_indices:
                if time.time() > deadline:
                    break
                if self.evals >= self.budget:
                    break
                
                x0 = archive_x[idx]
                
                # Define objective for local search: Sum of objectives
                def local_obj(x):
                    if self.evals >= self.budget:
                        return 1e6
                    f = eval_func(x)
                    return np.sum(f)
                
                # Use Nelder-Mead
                try:
                    res = minimize(local_obj, x0, method='Nelder-Mead', 
                                   options={'maxiter': 10, 'maxfev': evals_per_point})
                    if res.success or res.fun < 1e5:
                        # Add result to archive
                        new_f = eval_func(res.x) # Note: this consumes an eval if not cached, but minimize already called it. 
                        # Actually, minimize calls local_obj which calls eval_func. 
                        # We should be careful not to double count or exceed budget.
                        # The eval inside minimize is counted. 
                        # We don't need to re-evaluate if we trust res.x, but we need the f vector.
                        # Let's re-evaluate only if we haven't hit budget.
                        if self.evals < self.budget:
                             # We need to get the f vector. 
                             # We can just use the last f from minimize? No, minimize returns scalar.
                             # We must call problem again to get vector? 
                             # That costs 1 eval. 
                             # To save evals, we could modify local_obj to store last f.
                             pass
                             
                        # Simpler: Just add res.x and evaluate it once more if budget allows
                        if self.evals < self.budget:
                            f_new = eval_func(res.x)
                            # Check if dominated
                            if not is_dominated(f_new, archive_f):
                                # Remove dominated points
                                new_ax = []
                                new_af = []
                                for ax, af in zip(archive_x, archive_f):
                                    if not (np.all(f_new <= af) and np.any(f_new < af)):
                                        new_ax.append(ax)
                                        new_af.append(af)
                                new_ax.append(res.x)
                                new_af.append(f_new)
                                archive_x = new_ax
                                archive_f = new_af
                except:
                    pass

        # Return the archive (ignored, but must be valid)
        if archive_x:
            return np.array(archive_x)
        return np.array([lo + 0.5 * (hi - lo)])
