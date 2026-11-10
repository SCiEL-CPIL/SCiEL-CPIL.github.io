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
        deadline = 310.0  # Safety margin
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        # Use a fixed seed for reproducibility if needed, but task says np.random.seed(seed) is called by evaluator.
        # We should rely on the global state or create our own rng.
        # The task says: "For seeds 0 and 1 the evaluator calls np.random.seed(seed)... and calls it once."
        # This implies we should use np.random functions which will be seeded.
        
        # However, using np.random.default_rng is better practice, but if the evaluator seeds np.random, 
        # we might want to use np.random to respect that seeding.
        # Let's use np.random to be safe with the evaluator's seeding.
        
        # Initialization
        pop_size = 30
        # LHS
        lh_samples = np.zeros((pop_size, dim))
        for d in range(dim):
            strata = np.linspace(0, 1, pop_size + 1)
            for i in range(pop_size):
                lh_samples[i, d] = strata[i] + (strata[i + 1] - strata[i]) * np.random.random()
        
        X_init = lo + (hi - lo) * lh_samples
        
        # Add corners
        corners = np.array(
            [[hi[j] if (i >> j) & 1 else lo[j] for j in range(dim)] for i in range(32)]
        )
        # Replace first 6 LHS points with corners to ensure boundary coverage
        corner_indices = [0, 1, 2, 4, 8, 16]
        for i, ci in enumerate(corner_indices):
            X_init[i] = corners[ci]
            
        X = X_init.copy()
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Archive
        archive_X = []
        archive_F = []
        
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)
            
        def update_archive(x, f):
            nonlocal archive_X, archive_F
            # Check if dominated
            for af in archive_F:
                if dominates(af, f):
                    return
            # Remove dominated
            new_X = []
            new_F = []
            for ax, af in zip(archive_X, archive_F):
                if not dominates(f, af):
                    new_X.append(ax)
                    new_F.append(af)
            new_X.append(x.copy())
            new_F.append(f.copy())
            archive_X = new_X
            archive_F = new_F
            
            # Truncate if too large
            if len(archive_X) > 100:
                # Keep points with highest crowding distance
                cd = crowding_distance(np.array(archive_F))
                # Sort by cd descending
                order = np.argsort(-cd)
                keep_idx = order[:100]
                archive_X = [archive_X[i] for i in keep_idx]
                archive_F = [archive_F[i] for i in keep_idx]

        def crowding_distance(arch_F):
            n = len(arch_F)
            if n <= 2:
                return np.full(n, np.inf)
            cd = np.zeros(n)
            for m in range(n_obj):
                obj = arch_F[:, m]
                order = np.argsort(obj)
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                span = obj[order[-1]] - obj[order[0]]
                if span > 0:
                    for i in range(1, n - 1):
                        cd[order[i]] += (obj[order[i + 1]] - obj[order[i - 1]]) / span
            return cd

        # Initial archive
        for i in range(len(X)):
            update_archive(X[i], F[i])
            
        # DE Parameters
        F_start = 0.9
        F_end = 0.5
        CR = 0.8
        
        # Main Loop
        while used < budget:
            if time.time() - start_time > deadline:
                break
                
            # Adaptive F
            progress = used / budget
            F_factor = F_start + (F_end - F_start) * progress
            
            # Select parent from archive
            if len(archive_X) > 1:
                cd = crowding_distance(np.array(archive_F))
                i1 = np.random.randint(len(archive_X))
                i2 = np.random.randint(len(archive_X))
                parent_idx = i1 if cd[i1] >= cd[i2] else i2
                parent = archive_X[parent_idx]
            else:
                parent = X[np.random.randint(pop_size)]
                
            # DE Mutation
            idx1, idx2, idx3 = np.random.choice(pop_size, size=3, replace=False)
            mutant = X[idx1] + F_factor * (X[idx2] - X[idx3])
            
            # Crossover
            mask = np.random.random(dim) < CR
            trial = np.where(mask, mutant, parent)
            trial = np.clip(trial, lo, hi)
            
            ft = problem(trial)
            used += 1
            
            update_archive(trial, ft)
            
            # Update Population
            # Find worst point in population
            # Simple heuristic: point dominated by most others
            dom_count = np.zeros(pop_size, dtype=int)
            for i in range(pop_size):
                for j in range(pop_size):
                    if i != j and dominates(F[i], F[j]):
                        dom_count[j] += 1
                        
            worst_idx = np.argmax(dom_count)
            if dom_count[worst_idx] == 0:
                # All non-dominated, use sum of normalized objectives
                # Normalize F
                f_min = F.min(axis=0)
                f_max = F.max(axis=0)
                f_range = f_max - f_min
                f_range[f_range == 0] = 1.0
                F_norm = (F - f_min) / f_range
                worst_idx = np.argmax(F_norm.sum(axis=1))
                
            if not dominates(F[worst_idx], ft):
                X[worst_idx] = trial
                F[worst_idx] = ft
            else:
                # Random replacement to maintain diversity
                if np.random.random() < 0.1:
                    X[worst_idx] = trial
                    F[worst_idx] = ft

        # Polish Phase
        # Reserve last 30 evaluations for polish
        remaining = budget - used
        if remaining > 10 and len(archive_X) > 0 and time.time() - start_time < deadline - 20:
            # Select top 10 points from archive by crowding distance
            arch_F_arr = np.array(archive_F)
            cd = crowding_distance(arch_F_arr)
            order = np.argsort(-cd)
            n_polish = min(10, len(archive_X), remaining // 3)
            
            for pi in order[:n_polish]:
                if used >= budget - 1 or time.time() - start_time > deadline - 5:
                    break
                
                x0 = archive_X[pi].copy()
                
                # Weights for scalarization
                # Use range of entire archive
                obj_min = arch_F_arr.min(axis=0)
                obj_max = arch_F_arr.max(axis=0)
                obj_range = obj_max - obj_min
                obj_range[obj_range == 0] = 1.0
                weights = 1.0 / obj_range
                
                eval_count = [0]
                def scalar_obj(x):
                    fx = problem(np.clip(x, lo, hi))
                    eval_count[0] += 1
                    return np.sum(weights * fx)
                
                try:
                    res = minimize(
                        scalar_obj, x0,
                        method='Nelder-Mead',
                        options={'maxiter': 20, 'xatol': 1e-4, 'fatol': 1e-4}
                    )
                    # Check if we have budget for the final evaluation of res.x
                    # Nelder-Mead uses many evaluations. We need to track them.
                    # The problem is that minimize doesn't return the number of evaluations directly in a simple way for all methods, 
                    # but we can count them in the wrapper.
                    # However, we must ensure we don't exceed budget.
                    # If eval_count[0] + 1 <= remaining:
                    if used + eval_count[0] <= budget:
                        f_res = problem(np.clip(res.x, lo, hi))
                        used += eval_count[0] + 1
                        update_archive(res.x, f_res)
                    else:
                        # We ran out of budget during minimize
                        pass
                except Exception:
                    pass

        return X
