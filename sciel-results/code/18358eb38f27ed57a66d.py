import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hypervolume_contribution(self, point, archive, ref_point):
        """Calculate hypervolume contribution of a point given an archive."""
        if len(archive) == 0:
            return np.prod(np.maximum(0, ref_point - point))
        
        # Check if point is dominated by any point in archive
        for p in archive:
            if np.all(p <= point) and np.any(p < point):
                return 0.0
        
        # Calculate HV contribution using inclusion-exclusion for 3 objectives
        # For 3D, we can use the standard formula
        a = point
        b = ref_point
        
        # The contribution is the volume of the box [a, b] minus the parts covered by archive points
        # that are not dominated by a
        
        # For simplicity and speed, use a simple approximation:
        # Find the region where this point is the minimum in each objective
        # This is complex, so use a simpler approach: compute HV of {point} + archive 
        # minus HV of archive
        
        # Actually, let's use a direct computation for 3 objectives
        # Sort archive points by first objective
        if len(archive) == 0:
            return np.prod(np.maximum(0, b - a))
        
        # For 3D, the contribution of point a is:
        # Volume of box [a, b] - sum of volumes of boxes [max(a, p_i), b] for p_i that are not dominated by a
        # But this requires careful handling of overlaps
        
        # Simpler: just compute the HV difference
        # HV(archive ∪ {a}) - HV(archive)
        
        # For performance, use a simplified approach:
        # Find the "shadow" region of a that is not covered by any archive point
        # In 3D, this can be computed by finding for each pair of objectives
        # the maximum extent not blocked by archive points
        
        # Practical approach: binary search or direct integration
        # Given time constraints, use a conservative estimate
        base_vol = np.prod(np.maximum(0, b - a))
        
        # Subtract volumes covered by archive points that dominate parts of a's box
        covered = 0.0
        for p in archive:
            if np.all(p >= a) and np.any(p > a):
                # p is in a's box and not dominated by a
                # The intersection of [a,b] and [p,b] is [max(a,p), b]
                inter = np.maximum(a, p)
                inter_vol = np.prod(np.maximum(0, b - inter))
                covered += inter_vol
        
        # This overcounts overlaps, but for selection purposes it's a reasonable heuristic
        return max(0, base_vol - covered * 0.5)

    def _non_dominated_sort(self, F):
        """Perform non-dominated sorting."""
        n = len(F)
        m = F.shape[1]
        ranks = np.zeros(n, dtype=int)
        fronts = []
        remaining = list(range(n))
        
        while remaining:
            front = []
            for i in remaining:
                dominated = False
                for j in remaining:
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    front.append(i)
            if not front:
                front = [remaining[0]]
            fronts.append(front)
            for i in front:
                ranks[i] = len(fronts) - 1
            remaining = [i for i in remaining if i not in front]
        
        return ranks, fronts

    def _crowding_distance(self, F, fronts):
        """Calculate crowding distance for each front."""
        n = len(F)
        m = F.shape[1]
        dist = np.zeros(n)
        
        for f in fronts:
            if len(f) == 0:
                continue
            if len(f) == 1:
                dist[f[0]] = np.inf
                continue
            
            for k in range(m):
                sorted_f = sorted(f, key=lambda i: F[i][k])
                dist[sorted_f[0]] = np.inf
                dist[sorted_f[-1]] = np.inf
                min_val = F[sorted_f[0]][k]
                max_val = F[sorted_f[-1]][k]
                range_val = max_val - min_val
                if range_val < 1e-14:
                    continue
                for i in range(1, len(sorted_f) - 1):
                    idx = sorted_f[i]
                    dist[idx] += (F[sorted_f[i + 1]][k] - F[sorted_f[i - 1]][k]) / range_val
        
        return dist

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        
        # Population size
        N = 30
        
        # Reference point for hypervolume
        ref_point = np.array([1.1, 1.1, 25.0])
        
        # Initialize population using Latin Hypercube Sampling
        X = np.empty((N, n))
        for j in range(n):
            perm = rng.permutation(N)
            X[:, j] = lo[j] + (hi[j] - lo[j]) * (perm + rng.random(N)) / N
        
        F = np.array([problem(x) for x in X])
        used = N
        
        # DE parameters
        CR = 0.5  # Crossover rate
        F_init = 0.8  # Initial mutation scale
        F_min = 0.3
        F_max = 1.0
        
        # Adaptive F: start high for exploration, decrease for exploitation
        total_gens = max(1, (self.budget - used) // N)
        
        gen = 0
        while used < self.budget:
            gen += 1
            # Adaptive mutation scale
            progress = gen / max(1, total_gens)
            F_mut = F_init - (F_init - F_min) * progress
            F_mut = np.clip(F_mut, F_min, F_max)
            
            # Generate offspring using DE/rand/1/bin
            Y = np.empty_like(X)
            F_Y = np.empty_like(F)
            
            batch_size = min(N, self.budget - used)
            if batch_size == 0:
                break
            
            for i in range(batch_size):
                # Select three distinct random indices
                idxs = rng.choice(N, size=3, replace=False)
                a, b, c = idxs
                
                # Mutation: v = X_a + F * (X_b - X_c)
                v = X[a] + F_mut * (X[b] - X[c])
                
                # Crossover: binomial crossover
                j_rand = rng.integers(n)
                cross_mask = rng.random(n) < CR
                cross_mask[j_rand] = True  # Ensure at least one gene from v
                
                # Create offspring
                child = np.where(cross_mask, v, X[i])
                child = np.clip(child, lo, hi)
                
                Y[i] = child
                F_Y[i] = problem(child)
                used += 1
            
            # Combine and select
            R = np.vstack([X, Y])
            F_R = np.vstack([F, F_Y])
            
            # Non-dominated sorting
            ranks, fronts = self._non_dominated_sort(F_R)
            dist = self._crowding_distance(F_R, fronts)
            
            # Select N best by rank, then crowding distance
            order = np.lexsort((-dist, ranks))
            X = R[order[:N]]
            F = F_R[order[:N]]
            
            # Update reference point to be slightly better than worst in archive
            # Keep fixed reference point as specified
        
        return X
