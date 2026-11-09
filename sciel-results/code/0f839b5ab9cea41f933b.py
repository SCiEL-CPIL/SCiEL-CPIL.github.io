import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # MOPSO Parameters
        N = 20  # Swarm size
        w = 0.7  # Inertia weight
        c1 = 1.5  # Cognitive coefficient
        c2 = 1.5  # Social coefficient
        max_archive_size = 50
        
        # Initialize Position and Velocity
        X = lo + np.random.rand(N, self.dim) * span
        V = np.random.rand(N, self.dim) * 0.5 * span
        
        # Evaluate initial population
        F = np.array([problem(x) for x in X])
        evals = N
        
        # Initialize Personal Bests
        P_best = X.copy()
        F_best = F.copy()
        
        # Initialize Global Best Archive
        archive_X = X.copy()
        archive_F = F.copy()
        
        while evals < self.budget:
            r1 = np.random.rand(N, 1)
            r2 = np.random.rand(N, 1)
            
            # Select g_best from Archive
            nd_idx = self._find_non_dominated(archive_F)
            if len(nd_idx) > 0:
                # Pick one with max crowding distance from the non-dominated set in archive
                cdists = self._crowding_distance(archive_F[nd_idx])
                best_local_idx = np.argmax(cdists)
                g_best = archive_X[nd_idx[best_local_idx]]
            else:
                # Fallback: pick best based on sum of objectives
                norm_F = archive_F / (archive_F.max(axis=0) - archive_F.min(axis=0) + 1e-8)
                sums = norm_F.sum(axis=1)
                g_best = archive_X[np.argmin(sums)]
            
            # Update Velocity
            V = w * V + c1 * r1 * (P_best - X) + c2 * r2 * (g_best - X)
            
            # Clamp Velocity
            V_max = 0.5 * span
            V = np.clip(V, -V_max, V_max)
            
            # Update Position
            X_new = X + V
            X_new = np.clip(X_new, lo, hi)
            
            # Evaluate New Positions
            F_new = np.array([problem(x) for x in X_new])
            evals += N
            
            # Update Personal Bests
            for i in range(N):
                if self._dominates(F_new[i], F_best[i]):
                    P_best[i] = X_new[i]
                    F_best[i] = F_new[i]
            
            # Update Archive
            combined_X = np.vstack([archive_X, X_new])
            combined_F = np.vstack([archive_F, F_new])
            
            # Find non-dominated set
            nd_idx = self._find_non_dominated(combined_F)
            
            # If archive size exceeds limit, prune
            if len(nd_idx) > max_archive_size:
                # Calculate crowding distance for non-dominated set
                cdists = self._crowding_distance(combined_F[nd_idx])
                # Sort by crowding distance, keep best max_archive_size
                sorted_idx = np.argsort(-cdists)
                keep_local = sorted_idx[:max_archive_size]
                nd_idx = nd_idx[keep_local]
                
            archive_X = combined_X[nd_idx]
            archive_F = combined_F[nd_idx]
            
            # Move to next iteration
            X = X_new
            F = F_new
            
        return X

    def _dominates(self, a, b):
        return np.all(a <= b) and np.any(a < b)

    def _find_non_dominated(self, F):
        N = len(F)
        dominated_count = np.zeros(N, dtype=int)
        
        for i in range(N):
            for j in range(i+1, N):
                if self._dominates(F[i], F[j]):
                    dominated_count[j] += 1
                elif self._dominates(F[j], F[i]):
                    dominated_count[i] += 1
                    
        return [i for i in range(N) if dominated_count[i] == 0]

    def _crowding_distance(self, F):
        N = len(F)
        if N <= 2:
            return np.full(N, np.inf)
            
        cdists = np.zeros(N)
        for k in range(self.n_obj):
            sorted_idx = np.argsort(F[:, k])
            cdists[sorted_idx[0]] = np.inf
            cdists[sorted_idx[-1]] = np.inf
            range_val = F[sorted_idx[-1], k] - F[sorted_idx[0], k]
            if range_val == 0:
                continue
            for i in range(1, N-1):
                cdists[sorted_idx[i]] += (F[sorted_idx[i+1], k] - F[sorted_idx[i-1], k]) / range_val
        return cdists
