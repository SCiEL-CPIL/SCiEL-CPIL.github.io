import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        timeout = 295.0  # Safety margin before 300s hard limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Archive of non-dominated points
        archive_X = []
        archive_F = []
        evals = 0
        
        def dominates(f_new, f_arch):
            """Check if f_new dominates f_arch"""
            return np.all(f_new <= f_arch) and np.any(f_new < f_arch)
        
        def update_archive(x, f):
            """Update archive with new point, maintaining non-domination"""
            # Check if f is dominated by any archive point
            for f_arch in archive_F:
                if dominates(f_arch, f):
                    return  # f is dominated, don't add
            
            # Remove archive points dominated by f
            new_archive_X = []
            new_archive_F = []
            for i in range(len(archive_F)):
                if not dominates(f, archive_F[i]):
                    new_archive_X.append(archive_X[i])
                    new_archive_F.append(archive_F[i])
            
            # Add new point
            new_archive_X.append(x)
            new_archive_F.append(f)
            
            archive_X.clear()
            archive_F.clear()
            archive_X.extend(new_archive_X)
            archive_F.extend(new_archive_F)
        
        def hypervolume_contribution(f, archive_F, ref):
            """Compute hypervolume contribution of point f given archive and reference point"""
            # Only count if f is better than reference in all objectives
            if np.any(f >= ref):
                return 0.0
            
            # Simple approximation: volume of box from f to ref
            # This is an upper bound; actual contribution is less due to other points
            # For selection purposes, this works as a heuristic
            return np.prod(ref - f)
        
        def crowding_distance(archive_F):
            """Compute crowding distance for each point in archive"""
            n = len(archive_F)
            if n <= 2:
                return np.full(n, np.inf)
            
            cd = np.zeros(n)
            F = np.array(archive_F)
            
            for k in range(n_obj):
                # Sort by objective k
                order = np.argsort(F[:, k])
                cd[order[0]] = np.inf
                cd[order[-1]] = np.inf
                if F[order[-1], k] == F[order[0], k]:
                    continue
                for i in range(1, n-1):
                    cd[order[i]] += (F[order[i+1], k] - F[order[i-1], k]) / (F[order[-1], k] - F[order[0], k])
            
            return cd
        
        def tournament_select(archive_X, archive_F, k=3):
            """Tournament selection based on crowding distance"""
            n = len(archive_F)
            if n <= k:
                return np.random.randint(n)
            
            candidates = np.random.choice(n, k, replace=False)
            cd = crowding_distance([archive_F[i] for i in candidates])
            return candidates[np.argmax(cd)]
        
        # Reference point for hypervolume
        ref = np.array([1705.0, 12.0, 0.3])
        
        # Initialization: stratified sampling for better coverage
        N_init = min(20, budget // 10)
        for i in range(N_init):
            if time.time() - start_time > timeout:
                break
            x = lo + (hi - lo) * np.random.random(dim)
            f = problem(x)
            evals += 1
            update_archive(x, f)
        
        # Main loop: perturb archive points with adaptive step size
        while evals < budget:
            if time.time() - start_time > timeout:
                break
            
            if not archive_X:
                # Fallback: random point
                x = lo + (hi - lo) * np.random.random(dim)
            else:
                # Tournament selection to pick parent
                idx = tournament_select(archive_X, archive_F, k=3)
                x_parent = archive_X[idx]
                
                # Adaptive step size: start large, shrink over time
                progress = evals / budget
                sigma = (1.0 - 0.7 * progress) * (hi - lo) * 0.25
                x = x_parent + sigma * np.random.standard_normal(dim)
                x = np.clip(x, lo, hi)
            
            f = problem(x)
            evals += 1
            update_archive(x, f)
        
        return archive_X if archive_X else [np.zeros(dim)]
