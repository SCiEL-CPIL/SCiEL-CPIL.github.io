import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)
        
        n_obj = self.n_obj
        dim = self.dim
        
        def halton(dim, n, skip=1):
            points = np.zeros((n, dim))
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            for d in range(dim):
                p = primes[d % len(primes)]
                for i in range(n):
                    x = 0.0
                    f = 1.0 / p
                    idx = i + skip
                    while idx > 0:
                        idx, rem = divmod(idx, p)
                        x += f * rem
                        f /= p
                    points[i, d] = x
            return points

        # Phase 1: Space-filling initialization
        n_init = min(30, self.budget // 3)
        H = halton(dim, n_init)
        X = lo + (hi - lo) * H
        F = np.array([problem(x) for x in X])
        used = n_init
        
        # Maintain archive of non-dominated points
        archive_X = [X[i] for i in range(n_init)]
        archive_F = [F[i] for i in range(n_init)]
        
        # Remove dominated points from initial archive
        def prune_archive(archive_F):
            n = len(archive_F)
            if n <= 1:
                return list(range(n))
            keep = [True] * n
            for i in range(n):
                if not keep[i]:
                    continue
                for j in range(n):
                    if i == j or not keep[j]:
                        continue
                    if np.all(archive_F[j] <= archive_F[i]) and np.any(archive_F[j] < archive_F[i]):
                        keep[i] = False
                        break
            return [i for i in range(n) if keep[i]]
        
        keep_idx = prune_archive(archive_F)
        archive_X = [archive_X[i] for i in keep_idx]
        archive_F = [archive_F[i] for i in keep_idx]
        
        # Phase 2: Farthest-point sampling in objective space
        # At each step, select the point in the current archive that has the maximum
        # minimum Euclidean distance to all other points in the archive (in objective space),
        # then perturb it slightly to explore nearby regions.
        
        # To make this more effective, we'll use a combination:
        # 1. Find the "most isolated" point in the archive (max min-distance to others)
        # 2. Generate a candidate near that point
        # 3. Accept if it improves the archive
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            if len(archive_F) == 0:
                break
            
            # Compute pairwise distances in objective space
            n_arch = len(archive_F)
            F_arr = np.array(archive_F)
            
            if n_arch == 1:
                # Just perturb the single point
                x_center = archive_X[0]
                f_center = archive_F[0]
            else:
                # For each point, compute its minimum distance to all other points
                # Use vectorized computation
                diff = F_arr[:, np.newaxis, :] - F_arr[np.newaxis, :, :]
                dist = np.sqrt(np.sum(diff ** 2, axis=2))
                np.fill_diagonal(dist, np.inf)
                min_dists = np.min(dist, axis=1)
                
                # Select the point with maximum minimum distance (most isolated)
                # Add some randomness to break ties and explore different regions
                # Use softmax-like selection with temperature
                temp = 0.1 * np.max(min_dists) + 1e-10
                probs = np.exp(min_dists / temp)
                probs /= probs.sum()
                
                # Sample with some randomness
                idx = rng.choice(n_arch, p=probs)
                x_center = archive_X[idx]
                f_center = archive_F[idx]
            
            # Generate a candidate near the selected center
            # Step size decays as we use more budget
            progress = used / self.budget
            step_size = 0.1 * (1.0 - 0.5 * progress)
            
            noise = step_size * (hi - lo) * rng.standard_normal(dim)
            x_new = np.clip(x_center + noise, lo, hi)
            
            f_new = problem(x_new)
            used += 1
            
            # Check if f_new is non-dominated and improves the archive
            # First, check if it's dominated by any point in the archive
            is_dominated = False
            for f_old in archive_F:
                if np.all(f_old <= f_new) and np.any(f_old < f_new):
                    is_dominated = True
                    break
            
            if not is_dominated:
                # Remove points dominated by f_new
                new_archive_X = []
                new_archive_F = []
                for i in range(len(archive_X)):
                    if not (np.all(f_new <= archive_F[i]) and np.any(f_new < archive_F[i])):
                        new_archive_X.append(archive_X[i])
                        new_archive_F.append(archive_F[i])
                
                new_archive_X.append(x_new)
                new_archive_F.append(f_new)
                archive_X = new_archive_X
                archive_F = new_archive_F
                
                # Prune if archive gets too large
                if len(archive_F) > 50:
                    keep_idx = prune_archive(archive_F)
                    archive_X = [archive_X[i] for i in keep_idx]
                    archive_F = [archive_F[i] for i in keep_idx]
        
        return archive_X
