import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0 # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        N = 30
        
        # Initialize
        X = lo + (hi - lo) * np.random.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        
        archive_X = X.copy()
        archive_F = F.copy()
        
        used = N
        
        # DE Parameters
        F_mut = 0.8
        CR = 0.9
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Prune archive before selection to ensure we are sampling from current front
            n_points = len(archive_F)
            if n_points == 0:
                break
            dominated = np.zeros(n_points, dtype=bool)
            for i in range(n_points):
                if dominated[i]: continue
                for j in range(n_points):
                    if i == j or dominated[j]: continue
                    if np.all(archive_F[i] <= archive_F[j]) and np.any(archive_F[i] < archive_F[j]):
                        dominated[j] = True
                    if np.all(archive_F[j] <= archive_F[i]) and np.any(archive_F[j] < archive_F[i]):
                        dominated[i] = True
                        break
            nd_indices = np.where(~dominated)[0]
            
            if len(nd_indices) > N:
                # Prune by crowding distance
                nd_F = archive_F[nd_indices]
                m = len(nd_indices)
                dist = np.zeros(m)
                for k in range(self.n_obj):
                    order = np.argsort(nd_F[:, k])
                    dist[order[0]] = np.inf
                    dist[order[-1]] = np.inf
                    if m > 2:
                        range_k = nd_F[order[-1], k] - nd_F[order[0], k]
                        if range_k > 1e-10:
                            for i in range(1, m-1):
                                dist[order[i]] += (nd_F[order[i+1], k] - nd_F[order[i-1], k]) / range_k
                sorted_nd = nd_indices[np.argsort(dist)[::-1]]
                keep_indices = sorted_nd[:N]
            else:
                # Keep all non-dominated
                keep_indices = nd_indices
            
            archive_X = archive_X[keep_indices]
            archive_F = archive_F[keep_indices]
            
            # If archive is too small for DE, expand with random points or perturbations?
            # For now, if < 3, we can't do standard DE.
            if len(archive_X) < 3:
                # Fallback: Random perturbation of a random archive point
                idx = np.random.randint(0, len(archive_X))
                trial = archive_X[idx].copy()
                trial += (hi - lo) * 0.1 * np.random.randn(self.dim)
                trial = np.clip(trial, lo, hi)
            else:
                # Select target
                idx_target = np.random.randint(0, len(archive_X))
                x_target = archive_X[idx_target]
                
                # Select 3 distinct parents
                parents_idx = np.random.choice(len(archive_X), 3, replace=False)
                x_r1 = archive_X[parents_idx[0]]
                x_r2 = archive_X[parents_idx[1]]
                x_r3 = archive_X[parents_idx[2]]
                
                # Mutation
                v = x_r1 + F_mut * (x_r2 - x_r3)
                
                # Crossover
                j_rand = np.random.randint(0, self.dim)
                trial = np.copy(x_target)
                for j in range(self.dim):
                    if np.random.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Clip
                trial = np.clip(trial, lo, hi)
            
            # Evaluate
            f_trial = problem(trial)
            used += 1
            
            # Add to archive
            archive_X = np.vstack([archive_X, trial])
            archive_F = np.vstack([archive_F, f_trial])

        return archive_X
