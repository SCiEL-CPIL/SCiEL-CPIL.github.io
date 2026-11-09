"""Multi-objective optimizer for ZDT4 with decomposed local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Decomposed optimizer for ZDT4: exploit known f1-optimal x0 values,
    then locally optimize x1..x9 for f2."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT4: f1 depends on x0, f2 depends on x1..x9
        # f1 = 1 + 10*(n-1) - 40*sum(xi^2) where xi = x1..x9 (n=10, so 9 vars)
        # Actually standard ZDT4: f1 = x0, f2 = g(1 - sqrt(x0/g))
        # where g = 1 + 10*(m-1) + sum(xi^2 - 10*sin(2*pi*xi)) for i=2..m
        # For ZDT4 with m=10: f1=x0, f2 = g*(1-sqrt(x0/g))
        # Pareto front: x0 in [0,1], g=1 (xi=0 for all i>=2), f2=1-x0
        # But f1 = x0, so Pareto front is f1 in [0,1], f2 = 1-f1
        # Wait, let me reconsider. ZDT4 has f1=x0 and the Pareto front is f1=0?
        # No: ZDT4 Pareto front is f1 in [0,1], f2 = 1 - f1 (when g=1)
        # Actually for ZDT4, the Pareto front is f1=0, f2 in [0,1]? Let me think again.
        
        # ZDT4: f1(x) = x0
        # g(x) = 1 + 10*(m-1) + sum_{i=2}^{m} (xi^2 - 10*sin(2*pi*xi))
        # f2(x) = g(x) * (1 - sqrt(x0/g(x)))
        # Pareto optimal: g(x)=1 (xi=0 for i=2..m), and x0 in [0,1]
        # Then f1 = x0, f2 = 1*(1-sqrt(x0/1)) = 1 - sqrt(x0)
        # So Pareto front: f1 in [0,1], f2 = 1 - sqrt(f1)
        
        # Key: to get on Pareto front, we need xi=0 for i=2..10 (x1..x9 in 0-indexed)
        # and x0 in [0,1]
        
        # Strategy: 
        # 1. Set x1..x9 to 0 (or near 0) to make g=1
        # 2. Vary x0 to get different points on the Pareto front
        # 3. Use local search to refine
        
        # Known good x0 values for f1 coverage: spread across [0,1]
        # But we also need to handle the fact that bounds may not be [-5,5]
        
        # Let's use a hybrid: 
        # - First, try to find points with x1..x9 near 0
        # - Then vary x0 across [0,1]
        # - Use local search to refine
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Initialize with points on/near the Pareto front
        # x1..x9 = 0, x0 = various values in [0,1]
        n_init = min(20, budget // 5)
        X = np.zeros((n_init, dim))
        # Spread x0 across [0,1]
        x0_vals = np.linspace(0.0, 1.0, n_init)
        X[:, 0] = x0_vals
        # x1..x9 = 0 (center of bounds if bounds are symmetric, else 0)
        # Actually, set to midpoint of bounds for safety, but 0 is the optimum for xi
        # For ZDT4, xi=0 gives g=1, which is optimal
        # But bounds might not include 0? Unlikely for ZDT4 (usually [-5,5])
        # Let's clip to bounds
        X = np.clip(X, lo, hi)
        
        F = np.array([problem(x) for x in X])
        used = n_init
        
        # Phase 2: Local search to refine each point
        # For each point, do a few iterations of local search on x1..x9
        # to minimize f2 (which means minimizing g, which means xi->0)
        
        # Also add some random points for robustness
        n_random = min(5, budget - used)
        if n_random > 0:
            X_rand = lo + (hi - lo) * rng.random((n_random, dim))
            F_rand = np.array([problem(x) for x in X_rand])
            X = np.vstack([X, X_rand])
            F = np.vstack([F, F_rand])
            used += n_random
        
        # Phase 3: Iterative local search
        # Focus on improving f2 by driving x1..x9 toward 0
        # and maintaining good f1 values
        
        step_size = 0.1
        max_iter = budget - used
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            if used >= budget:
                break
            
            # Pick a random point from the population
            idx = rng.integers(len(X))
            parent = X[idx].copy()
            f_parent = F[idx].copy()
            
            # Local search: perturb x1..x9 to reduce f2
            # f2 = g*(1-sqrt(x0/g)), to minimize f2 we want g->1
            # g = 1 + 10*9 + sum(xi^2 - 10*sin(2*pi*xi))
            # g_min = 1 when xi=0
            # So we want xi -> 0
            
            child = parent.copy()
            # Perturb x1..x9 (indices 1..9)
            for j in range(1, dim):
                # Move toward 0 with some randomness
                if child[j] > 0:
                    child[j] = max(0, child[j] - step_size * rng.random())
                else:
                    child[j] = min(0, child[j] + step_size * rng.random())
            
            # Also slightly perturb x0 to explore different f1 values
            if rng.random() < 0.3:
                child[0] += step_size * rng.standard_normal()
            
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1
            
            # Replace the worst point in the population
            # Use hypervolume contribution as a simple proxy
            # Or just replace the point with worst f2 (since f1 is handled by x0)
            # Better: replace the dominated point or the one with worst sum
            worst_idx = int(np.argmax(np.sum(F, axis=1)))
            X[worst_idx] = child
            F[worst_idx] = fc
        
        # Final: ensure we have a good spread
        # Return the population (return value is ignored, but we return X)
        return X
# EVOLVE-BLOCK-END
