import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin from 330s limit
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Reference point for HV calculation (used for selection, not just scoring)
        ref_point = np.array([3000.0, 0.06])
        
        # 1. Halton Initialization
        def halton_sequence(n, dim):
            points = np.zeros((n, dim))
            for d in range(dim):
                base = d + 2  # Start with primes 2, 3, 5, 7
                points[:, d] = self._halton_sequence(n, base)
            return points
        
        def _halton_sequence(self, n, base):
            result = np.zeros(n)
            f = 1.0 / base
            for i in range(1, n + 1):
                idx = i
                while idx > 0:
                    result[i-1] += f * (idx % base)
                    idx //= base
                    f /= base
            return result
        
        n_init = min(50, budget // 4)
        # Map Halton [0,1] to [lo, hi]
        X_init = lo + range_ * halton_sequence(n_init, dim)
        
        # Evaluate initial population
        F_init = np.array([problem(x) for x in X_init])
        used = n_init
        
        # Store all evaluated points to track non-dominated front
        # We keep a list of (x, f) pairs
        archive = [(x, f) for x, f in zip(X_init, F_init)]
        
        def is_dominated(f_new, F_archive):
            # f_new: (n_obj,)
            # F_archive: list of (n_obj,)
            for f_arch in F_archive:
                if np.all(f_arch <= f_new) and np.any(f_arch < f_new):
                    return True
            return False
            
        def update_archive(archive, x_new, f_new):
            # Check if new point is dominated
            if is_dominated(f_new, [f for _, f in archive]):
                return archive
            
            # Remove points dominated by new point
            new_archive = []
            for x_old, f_old in archive:
                if np.all(f_new <= f_old) and np.any(f_new < f_old):
                    continue # Old point is dominated, discard
                new_archive.append((x_old, f_old))
            
            new_archive.append((x_new, f_new))
            return new_archive

        # 2. Main Evolutionary Loop
        # Target: use ~75% of budget for exploration
        n_evo = int(budget * 0.7)
        
        for i in range(n_evo):
            if used >= budget:
                break
            if time.time() > deadline:
                break
                
            # Get current non-dominated front
            F_list = [f for _, f in archive]
            X_list = [x for x, _ in archive]
            
            # Identify non-dominated indices
            nd_indices = []
            for i_idx, f_i in enumerate(F_list):
                dominated = False
                for j_idx, f_j in enumerate(F_list):
                    if i_idx != j_idx:
                        if np.all(f_j <= f_i) and np.any(f_j < f_i):
                            dominated = True
                            break
                if not dominated:
                    nd_indices.append(i_idx)
            
            if not nd_indices:
                nd_indices = [0]
                
            # Select parent: Random from non-dominated
            parent_idx = np.random.choice(nd_indices)
            parent_x = X_list[parent_idx]
            
            # Mutation: Gaussian with decaying sigma
            # Start with 0.2 * range, decay to 0.01 * range
            progress = i / max(1, n_evo)
            sigma = (0.2 - 0.19 * progress) * range_
            
            child_x = parent_x + sigma * np.random.standard_normal(dim)
            child_x = np.clip(child_x, lo, hi)
            
            child_f = problem(child_x)
            used += 1
            
            archive = update_archive(archive, child_x, child_f)
            
        # 3. Local Polish Phase
        # Use remaining budget to refine the best points
        remaining_budget = budget - used
        
        if remaining_budget > 0:
            # Sort archive by sum of objectives (simple heuristic for "best")
            # Or by HV contribution. Sum is cheaper.
            scored_archive = sorted(archive, key=lambda item: np.sum(item[1]))
            
            # Pick top K points
            n_polish_targets = min(5, len(scored_archive))
            targets = scored_archive[:n_polish_targets]
            
            evals_per_target = remaining_budget // max(1, n_polish_targets)
            
            for x_target, f_target in targets:
                for k in range(evals_per_target):
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Small local step
                    local_sigma = 0.01 * range_
                    child_x = x_target + local_sigma * np.random.standard_normal(dim)
                    child_x = np.clip(child_x, lo, hi)
                    
                    child_f = problem(child_x)
                    used += 1
                    
                    archive = update_archive(archive, child_x, child_f)
                    
                    # Optional: If child is better, update target for next step?
                    # Since we record all, we don't strictly need to, but it helps focus.
                    # However, modifying 'targets' list while iterating is tricky.
                    # We just generate random neighbors of the original top points.
                    # This is sufficient for "polish".

        return archive

    # Helper for Halton, defined inside to avoid self-reference issues if class is reloaded
    # Actually, defining it inside __call__ or as a static method is better.
    # Let's move Halton generation to a static method or local function to be safe.
    
    @staticmethod
    def _halton(n, base):
        result = np.zeros(n)
        f = 1.0 / base
        for i in range(1, n + 1):
            idx = i
            while idx > 0:
                result[i-1] += f * (idx % base)
                idx //= base
                f /= base
        return result

    # Re-defining __call__ to use the static method correctly
    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # 1. Halton Initialization
        n_init = min(50, budget // 4)
        X_init = np.zeros((n_init, dim))
        for d in range(dim):
            X_init[:, d] = self._halton(n_init, d + 2)
        X_init = lo + range_ * X_init
        
        F_init = np.array([problem(x) for x in X_init])
        used = n_init
        
        archive = [(x, f) for x, f in zip(X_init, F_init)]
        
        def is_dominated(f_new, F_archive):
            for f_arch in F_archive:
                if np.all(f_arch <= f_new) and np.any(f_arch < f_new):
                    return True
            return False
            
        def update_archive(archive, x_new, f_new):
            if is_dominated(f_new, [f for _, f in archive]):
                return archive
            
            new_archive = []
            for x_old, f_old in archive:
                if np.all(f_new <= f_old) and np.any(f_new < f_old):
                    continue
                new_archive.append((x_old, f_old))
            
            new_archive.append((x_new, f_new))
            return new_archive

        # 2. Main Evolutionary Loop
        n_evo = int(budget * 0.7)
        
        for i in range(n_evo):
            if used >= budget:
                break
            if time.time() > deadline:
                break
                
            F_list = [f for _, f in archive]
            X_list = [x for x, _ in archive]
            
            nd_indices = []
            for i_idx in range(len(F_list)):
                dominated = False
                for j_idx in range(len(F_list)):
                    if i_idx != j_idx:
                        if np.all(F_list[j_idx] <= F_list[i_idx]) and np.any(F_list[j_idx] < F_list[i_idx]):
                            dominated = True
                            break
                if not dominated:
                    nd_indices.append(i_idx)
            
            if not nd_indices:
                nd_indices = [0]
                
            parent_idx = np.random.choice(nd_indices)
            parent_x = X_list[parent_idx]
            
            progress = i / max(1, n_evo)
            sigma = (0.2 - 0.19 * progress) * range_
            
            child_x = parent_x + sigma * np.random.standard_normal(dim)
            child_x = np.clip(child_x, lo, hi)
            
            child_f = problem(child_x)
            used += 1
            
            archive = update_archive(archive, child_x, child_f)
            
        # 3. Local Polish Phase
        remaining_budget = budget - used
        
        if remaining_budget > 0:
            scored_archive = sorted(archive, key=lambda item: np.sum(item[1]))
            n_polish_targets = min(5, len(scored_archive))
            targets = scored_archive[:n_polish_targets]
            
            evals_per_target = remaining_budget // max(1, n_polish_targets)
            
            for x_target, f_target in targets:
                for k in range(evals_per_target):
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    local_sigma = 0.01 * range_
                    child_x = x_target + local_sigma * np.random.standard_normal(dim)
                    child_x = np.clip(child_x, lo, hi)
                    
                    child_f = problem(child_x)
                    used += 1
                    
                    archive = update_archive(archive, child_x, child_f)

        return archive
