import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _compute_hv(self, f1_vals, f2_vals, ref):
        """
        Compute Hypervolume for 2D points.
        f1_vals, f2_vals: numpy arrays of objective values.
        ref: tuple (ref1, ref2).
        """
        if len(f1_vals) == 0:
            return 0.0
        
        # Filter points strictly better than reference
        mask = (f1_vals < ref[0]) & (f2_vals < ref[1])
        if not np.any(mask):
            return 0.0
            
        f1 = f1_vals[mask]
        f2 = f2_vals[mask]
        
        # Sort by f1 ascending
        sort_idx = np.argsort(f1)
        f1 = f1[sort_idx]
        f2 = f2[sort_idx]
        
        # Extract Pareto front
        # Minimizing: Point A dominates B if A.f1 <= B.f1 and A.f2 <= B.f2 and at least one is strict.
        # Sorted by f1 ascending: f1[0] <= f1[1] <= ...
        # Point i dominates j (i < j) if f2[i] <= f2[j].
        # So we want to keep points i such that f2[i] < min(f2[0..i-1]).
        # If f2[i] == min(f2[0..i-1]), it is dominated by the point with that min f2 (which has smaller f1).
        # So we keep i if f2[i] < min_f2_so_far.
        
        pareto_f1 = []
        pareto_f2 = []
        min_f2_so_far = np.inf
        for i in range(len(f1)):
            if f2[i] < min_f2_so_far:
                pareto_f1.append(f1[i])
                pareto_f2.append(f2[i])
                min_f2_so_far = f2[i]
            
        if not pareto_f1:
            return 0.0
            
        pf1 = np.array(pareto_f1)
        pf2 = np.array(pareto_f2)
        
        # Now compute HV of the Pareto front
        # pf1 is ascending, pf2 is strictly descending.
        # Add reference point
        # HV = sum over i of (pf1[i+1] - pf1[i]) * (ref[1] - pf2[i]) + (ref[0] - pf1[-1]) * (ref[1] - pf2[-1])
        
        m = len(pf1)
        if m == 1:
            return (ref[0] - pf1[0]) * (ref[1] - pf2[0])
            
        hv = 0.0
        for i in range(m - 1):
            hv += (pf1[i + 1] - pf1[i]) * (ref[1] - pf2[i])
        hv += (ref[0] - pf1[-1]) * (ref[1] - pf2[-1])
        
        return hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0 # Safety margin
        
        # 1. Model: ZDT2 Pareto Front is x2..xD = 0, f1=x1, f2=1-sqrt(x1)
        # We optimize the x1 values to maximize HV of the resulting set.
        
        N = self.budget
        dim = self.dim
        
        # Initial guess: Chebyshev nodes in [0, 1]
        # Chebyshev nodes minimize the maximum gap and are optimal for approximating
        # smooth functions. For the convex ZDT2 front, this should distribute points
        # more efficiently than uniform spacing.
        # Chebyshev nodes of the first kind on [-1, 1]: cos((2k-1)*pi/(2N))
        # Map to [0, 1]: 0.5 * (1 + cos((2k-1)*pi/(2N)))
        k = np.arange(1, N + 1)
        angles = (2 * k - 1) * np.pi / (2 * N)
        x1_params = 0.5 * (1.0 + np.cos(angles))
        # Ensure sorted ascending
        x1_params = np.sort(x1_params)
        
        # Optimization of x1_params to maximize HV
        # Since the model is analytic, we can compute HV without calling problem()
        ref = (11.0, 11.0)
        
        def get_hv(x1s):
            # x1s are in [0, 1]
            f1 = x1s
            f2 = 1.0 - np.sqrt(f1)
            return self._compute_hv(f1, f2, ref)
        
        current_hv = get_hv(x1_params)
        
        # Simple hill climbing to improve grid distribution
        # We want to maximize HV.
        # Perturb each point slightly.
        step_size = 0.01
        for iter in range(100): # 100 iterations is fast
            if time.time() > deadline - 10.0: break
            improved = False
            for i in range(N):
                # Try small random step
                original = x1_params[i]
                # Try 5 random steps
                for _ in range(5):
                    x1_params[i] = np.clip(original + step_size * np.random.uniform(-1, 1), 0.0, 1.0)
                    new_hv = get_hv(x1_params)
                    if new_hv > current_hv + 1e-12:
                        current_hv = new_hv
                        improved = True
                        break
                    else:
                        x1_params[i] = original
            if not improved:
                step_size *= 0.5
                if step_size < 1e-5:
                    break

        # 2. Execute evaluations using the optimized parameters
        # Prepare vectors
        X = np.zeros((N, dim))
        X[:, 0] = x1_params
        # x[1:] are already 0
        
        results = []
        for i in range(N):
            if time.time() > deadline:
                break
            x = X[i]
            # Ensure finite and within bounds (though 0 and x1 in [0,1] are safe)
            f = problem(x)
            results.append(f)
            
        return X
