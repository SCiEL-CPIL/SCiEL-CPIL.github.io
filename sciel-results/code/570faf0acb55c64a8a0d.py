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
        deadline = start_time + 290.0 # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Evaluation counter
        self.eval_count = 0
        
        def counting_problem(x):
            self.eval_count += 1
            return problem(x)
            
        # Phase 1: Global Exploration with DE
        N = 20
        
        # Initialize
        X = lo + (hi - lo) * np.random.random((N, self.dim))
        F = np.array([counting_problem(x) for x in X])
        
        archive_X = X.copy()
        archive_F = F.copy()
        
        # DE Parameters
        F_mut = 0.8
        CR = 0.9
        
        # Run DE for a fixed number of steps
        # We want to use 100 evals for DE (20 init + 80 steps)
        de_steps = 80
        
        for _ in range(de_steps):
            if self.eval_count >= self.budget or time.time() > deadline:
                break
            
            # Select target
            idx_target = np.random.randint(0, len(archive_X))
            x_target = archive_X[idx_target]
            
            # Select 3 parents for DE
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
            f_trial = counting_problem(trial)
            
            # Add to archive
            archive_X = np.vstack([archive_X, trial])
            archive_F = np.vstack([archive_F, f_trial])
            
            # Prune Archive to size N
            n_points = len(archive_F)
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
                # Keep top N by crowding distance
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
                keep_indices = nd_indices
                # Fill with dominated if needed?
                if len(keep_indices) < N:
                    dominated_indices = np.where(dominated)[0]
                    if len(dominated_indices) > 0:
                        need = N - len(keep_indices)
                        # Pick random dominated
                        keep_dominated = np.random.choice(dominated_indices, size=min(need, len(dominated_indices)), replace=False)
                        keep_indices = np.concatenate([keep_indices, keep_dominated])

            archive_X = archive_X[keep_indices]
            archive_F = archive_F[keep_indices]

        # Phase 2: Local Exploitation with Nelder-Mead
        # Select top 5 points from archive for local search
        if self.eval_count < self.budget:
            # Calculate crowding distance for the entire archive to pick best 5
            n_points = len(archive_F)
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
            
            if len(nd_indices) > 0:
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
                                
                # Sort by dist descending
                sorted_nd = nd_indices[np.argsort(dist)[::-1]]
                # Pick top 5
                top_5_indices = sorted_nd[:min(5, len(sorted_nd))]
                
                for idx in top_5_indices:
                    if self.eval_count >= self.budget or time.time() > deadline:
                        break
                    
                    anchor = archive_X[idx]
                    # Generate random weight vector
                    weights = np.random.dirichlet(np.ones(self.n_obj))
                    
                    def nm_obj(x, w=weights):
                        x_clipped = np.clip(x, lo, hi)
                        fx = counting_problem(x_clipped)
                        return np.dot(w, fx)
                    
                    # Run Nelder-Mead
                    # maxiter 10 should take ~10-20 evals
                    try:
                        res = minimize(nm_obj, anchor, method='Nelder-Mead', options={'maxiter': 10, 'xatol': 1e-6, 'fatol': 1e-6})
                    except Exception:
                        pass
                        
        return archive_X
