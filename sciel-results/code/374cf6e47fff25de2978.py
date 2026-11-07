import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Archive of non-dominated points
        archive_F = []
        archive_X = []
        evals = 0
        
        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)
        
        def add_to_archive(x, f):
            nonlocal archive_X, archive_F, evals
            # Check if x is dominated by any in archive
            for i, f_arch in enumerate(archive_F):
                if dominates(f_arch, f):
                    return False
            # Check if x dominates any in archive
            new_archive_X = []
            new_archive_F = []
            for j, f_arch in enumerate(archive_F):
                if not dominates(f, f_arch):
                    new_archive_X.append(archive_X[j])
                    new_archive_F.append(f_arch)
            new_archive_X.append(x)
            new_archive_F.append(f)
            archive_X = new_archive_X
            archive_F = new_archive_F
            return True
        
        # Tchebycheff scalarization with weight vectors
        # Generate 30 uniformly distributed weight vectors in 3D
        n_weights = 30
        weights = []
        # Use a grid on the simplex
        for i in range(n_weights + 1):
            for j in range(n_weights + 1 - i):
                k = n_weights - i - j
                w = np.array([i, j, k], dtype=np.float64)
                if w.sum() == n_weights:
                    weights.append(w / n_weights)
        # Take first 30
        weights = np.array(weights[:30])
        
        # For each weight vector, solve a sub-problem
        # Budget allocation: use ~15 evals per sub-problem
        evals_per_subproblem = max(3, budget // (2 * len(weights)))
        
        for w in weights:
            if evals >= budget or time.time() > deadline:
                break
            
            # Tchebycheff scalarization: min max_i w_i * (f_i - z_i)
            # Use reference point z = [1, 1, 1] (typical for WFG7)
            z = np.array([1.0, 1.0, 1.0])
            
            def tcheb(x):
                nonlocal evals
                f = problem(x)
                evals += 1
                return np.max(w * (f - z))
            
            # Initialize at center of search space
            x0 = (lo + hi) / 2.0
            
            # Simple local search: coordinate descent with adaptive step
            best_x = x0.copy()
            best_f = problem(best_x)
            evals += 1
            add_to_archive(best_x, best_f)
            
            # Coordinate descent
            step_size = 0.1 * (hi - lo)
            for coord in range(dim):
                if evals >= budget or time.time() > deadline:
                    break
                # Try positive and negative steps
                for direction in [1, -1]:
                    trial_x = best_x.copy()
                    trial_x[coord] += direction * step_size[coord]
                    trial_x = np.clip(trial_x, lo, hi)
                    trial_f = problem(trial_x)
                    evals += 1
                    if np.all(trial_f <= best_f) and np.any(trial_f < best_f):
                        best_x = trial_x
                        best_f = trial_f
                        add_to_archive(best_x, best_f)
                    elif np.all(trial_f <= best_f):
                        # Not strictly better, but not worse - keep exploring
                        pass
                # Adaptive step: shrink if no improvement
                step_size[coord] *= 0.5
            
            # Also try a few random perturbations around best_x
            for _ in range(2):
                if evals >= budget or time.time() > deadline:
                    break
                perturbation = 0.05 * (hi - lo) * np.random.standard_normal(dim)
                trial_x = np.clip(best_x + perturbation, lo, hi)
                trial_f = problem(trial_x)
                evals += 1
                add_to_archive(trial_x, trial_f)
        
        return archive_X
