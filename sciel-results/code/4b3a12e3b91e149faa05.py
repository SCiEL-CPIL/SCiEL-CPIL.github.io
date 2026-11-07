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
        
        # Phase 2: Weighted-sum local search (structural change from farthest-point sampling)
        # Use multiple weight vectors to explore different trade-offs systematically
        # Each weight vector defines a scalarized sub-problem; we do local search around
        # the best-known solution for each weight
        
        # Generate weight vectors on the simplex
        n_weights = 8
        weights = []
        for w1 in np.linspace(0.05, 0.9, 3):
            for w2 in np.linspace(0.05, 0.9 - w1, 3):
                w3 = 1.0 - w1 - w2
                if w3 >= 0.05:
                    weights.append(np.array([w1, w2, w3]))
        # Add uniform weight
        weights.append(np.array([1.0/3, 1.0/3, 1.0/3]))
        
        # For each weight, maintain best solution found
        best_for_weight = {}
        for w in weights:
            # Initialize with the archive point that minimizes the weighted sum
            F_arr = np.array(archive_F)
            weighted_sums = F_arr @ w
            best_idx = np.argmin(weighted_sums)
            best_for_weight[tuple(w)] = (archive_X[best_idx].copy(), archive_F[best_idx].copy())
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Select a weight to explore (cycle through with some randomness)
            weight_list = list(weights)
            w = weight_list[int(rng.random() * len(weight_list))]
            
            x_best, f_best = best_for_weight[tuple(w)]
            
            # Local search: generate several candidates around x_best
            n_candidates = 3
            best_candidate = None
            best_candidate_score = np.inf
            
            for _ in range(n_candidates):
                # Adaptive step size based on how close we are to the deadline
                progress = used / self.budget
                step_size = 0.05 * (1.0 - 0.3 * progress)
                
                noise = step_size * (hi - lo) * rng.standard_normal(dim)
                x_new = np.clip(x_best + noise, lo, hi)
                
                f_new = problem(x_new)
                used += 1
                
                if used >= self.budget or time.time() > deadline:
                    break
                
                # Score by weighted sum (lower is better)
                score = f_new @ w
                
                # Check if this is better than current best for this weight
                if score < best_for_weight[tuple(w)][1] @ w:
                    best_for_weight[tuple(w)] = (x_new.copy(), f_new.copy())
                
                # Also check if it's non-dominated and should go in archive
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
            
            # If we've explored all candidates for this weight, move to next
            # The while loop will continue until budget is exhausted
        
        return archive_X
