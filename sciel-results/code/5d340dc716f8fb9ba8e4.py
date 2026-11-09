"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-aware evolutionary multi-objective search for DTLZ7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 290s safety margin within 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj

        # Population size
        pop_size = 40
        used = 0

        def dominates(a, b):
            """Check if a dominates b (a is better in all, strictly better in at least one)."""
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_sort(F):
            """Fast non-dominated sorting. Returns list of fronts (lists of indices)."""
            n_pts = len(F)
            S = [[] for _ in range(n_pts)]  # S[i] = indices dominated by i
            n_dom = np.zeros(n_pts, dtype=int)
            rank = np.zeros(n_pts, dtype=int)
            fronts = []

            for i in range(n_pts):
                for j in range(i + 1, n_pts):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1

            # First front
            front = [i for i in range(n_pts) if n_dom[i] == 0]
            rank[front] = 0
            k = 0
            while front:
                fronts.append(front)
                next_front = []
                for i in front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                front = next_front
                k += 1

            return fronts

        def crowding_distance(F, indices):
            """Compute crowding distance for a set of indices."""
            n = len(indices)
            if n <= 2:
                return np.full(n, np.inf)

            cd = np.zeros(n)
            for k in range(m):
                f_k = F[indices, k]
                min_k = np.min(f_k)
                max_k = np.max(f_k)
                if max_k - min_k < 1e-15:
                    cd[:] = np.inf
                    continue
                idx_sorted = np.argsort(f_k)
                cd[idx_sorted[0]] = np.inf
                cd[idx_sorted[-1]] = np.inf
                for i in range(1, n - 1):
                    cd[idx_sorted[i]] += (f_k[idx_sorted[i + 1]] - f_k[idx_sorted[i - 1]]) / (max_k - min_k)
            return cd

        # Initialize population with hypervolume-aware sampling
        # DTLZ7: g(x) = sum(x_i^0.5) for i in [n-m, n), h(u,g) = (1+g) * prod(1 - u_i/(1+g) - (u_i/(1+g))^cos(10*pi*u_i/(1+g)+pi)/2)
        # For 3 objectives, 22 vars: 19 decision vars, 3 objective vars
        # Key insight: for DTLZ7, setting g=0 (i.e., x[19:22] = 0) and varying x[19:22] gives the Pareto front
        
        # Actually for DTLZ7: f_1 = u_0, f_2 = u_1*(1+g), f_3 = u_2*(1+g) where u_i are the last m variables
        # and g = sum(x_i^0.5) for i in [n-m, n) ... wait let me reconsider.
        
        # DTLZ7 with M objectives:
        # g(x) = sum_{i=n-M+1}^{n} x_i^0.5  (where x is 0-indexed, so last M variables)
        # h(u, g) = (1+g) * prod_{i=1}^{M-1} (1 - u_i/(1+g) - (u_i/(1+g))^cos(10*pi*u_i/(1+g)+pi)/2)
        # f_1 = u_0
        # f_i = u_i * h(u, g) for i = 2, ..., M
        # where u_i = x_{n-M+i-1} (0-indexed: x[n-M+i-1])
        
        # For M=3, n=22: u_0 = x[19], u_1 = x[20], u_2 = x[21], g = sum(x[19:22]^0.5)
        # Wait, that's not right. Let me re-read.
        
        # Standard DTLZ7: 
        # g(x) = sum_{i=n-M+1}^{n} x_i^0.5  (using 1-indexed, so variables n-M+1 to n)
        # In 0-indexed: g = sum(x[n-M:i] for i in range(n-M, n)) = sum(x[i]^0.5 for i in range(n-M, n))
        # For M=3, n=22: g = x[19]^0.5 + x[20]^0.5 + x[21]^0.5
        # u_i = x_{n-M+i} for i=0,...,M-1 (0-indexed)
        # So u_0 = x[19], u_1 = x[20], u_2 = x[21]
        # h(u,g) = (1+g) * prod_{i=1}^{M-1} (1 - u_i/(1+g) - (u_i/(1+g))^cos(10*pi*u_i/(1+g)+pi)/2)
        # f_1 = u_0
        # f_i = u_i * h(u,g) for i=2,...,M
        
        # To get points on/near Pareto front: set g=0 by making x[19:22]=0, but then u_i=0 too.
        # Actually for DTLZ7, the Pareto front is where g=0, which means x[19:22]=0.
        # But then u_i = x[19:22] = 0, giving f=(0,0,0).
        
        # I think I'm confusing myself. Let me just use a good evolutionary approach.
        
        # Better initialization: mix of random and front-aware points
        X = np.random.uniform(lo, hi, size=(pop_size, n))
        
        # Add some structured points near the known Pareto front region
        # For DTLZ7, good points have small g, so x[n-m:] should be small
        for i in range(pop_size // 2):
            x = np.random.uniform(lo, hi, size=n)
            # Make last m variables small to reduce g
            x[n-m:] = np.random.uniform(0, 0.1, size=m)
            X[i] = x
        
        F = np.array([problem(x) for x in X])
        used = pop_size

        # Evolutionary loop
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Non-dominated sort
            fronts = non_dominated_sort(F)
            
            # Select parents from first front using crowding distance
            front0 = fronts[0]
            if len(front0) > 1:
                cd = crowding_distance(F, front0)
                # Select 2 parents with highest crowding distance
                parent_idx = np.argsort(cd)[-2:]
                p1 = X[front0[parent_idx[0]]]
                p2 = X[front0[parent_idx[1]]]
            else:
                p1 = X[front0[0]] if len(front0) > 0 else X[0]
                p2 = X[front0[0]] if len(front0) > 0 else X[1]

            # Crossover and mutation
            child = (p1 + p2) / 2.0
            # Adaptive mutation rate
            progress = used / self.budget
            mut_rate = 0.3 * (1 - progress * 0.5)
            child += mut_rate * (hi - lo) * rng.standard_normal(n)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Replace worst point based on dominance
            # Find if child dominates any existing point
            replaced = False
            for i in range(len(F)):
                if dominates(fc, F[i]):
                    X[i] = child
                    F[i] = fc
                    replaced = True
                    break
            
            if not replaced:
                # Replace point with worst crowding distance in its front
                # Find front of each point
                # Simple approach: replace random point if no dominance
                # Better: replace point that is dominated
                dominated_idx = []
                for i in range(len(F)):
                    for j in range(len(F)):
                        if i != j and dominates(F[j], F[i]):
                            dominated_idx.append(i)
                            break
                if dominated_idx:
                    idx = dominated_idx[rng.integers(len(dominated_idx))]
                else:
                    # Replace point with worst sum of objectives
                    idx = int(np.argmax(np.sum(F, axis=1)))
                X[idx] = child
                F[idx] = fc

        return X
# EVOLVE-BLOCK-END
