import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        
        # Global state
        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0
        
        def eval_func(x):
            nonlocal calls, best_f, best_x
            if time.time() - start_time > 290: # Safety margin
                return float('inf')
            if calls >= budget:
                return float('inf')
            
            x = np.asarray(x, dtype=np.float64)
            fx = func(x)
            calls += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global Sampling
        # Use 10% of budget for global search
        n_global = max(10, budget // 10)
        rng = np.random.default_rng(42)
        
        candidates = []
        
        # Uniform sampling
        for _ in range(n_global // 2):
            x = rng.uniform(-5, 5, dim)
            fx = eval_func(x)
            candidates.append((fx, x))
            
        # Stratified/Latin Hypercube-like sampling
        # Divide space into bins
        n_per_dim = int(np.ceil((n_global // 2) ** (1.0 / dim)))
        # This might be too many points if n_global is large and dim is small
        # For dim=5, n_global=50, n_per_dim = 2. 2^5 = 32 points.
        # Let's just use a grid of points with jitter
        grid_size = 2
        points_per_dim = grid_size ** dim
        if points_per_dim > n_global // 2:
             # Just use uniform for the rest
             for _ in range(n_global // 2):
                 x = rng.uniform(-5, 5, dim)
                 fx = eval_func(x)
                 candidates.append((fx, x))
        else:
             # Generate grid points
             coords = np.linspace(-5, 5, grid_size)
             for i in range(grid_size):
                 for j in range(grid_size):
                     for k in range(grid_size):
                         for l in range(grid_size):
                             for m in range(grid_size):
                                 x = np.array([i, j, k, l, m]) # Wait, coords are values
                                 x = np.array([coords[i], coords[j], coords[k], coords[l], coords[m]])
                                 # Add jitter
                                 x += rng.normal(0, 0.5, dim)
                                 x = np.clip(x, -5, 5)
                                 fx = eval_func(x)
                                 candidates.append((fx, x))
                                 if calls >= budget:
                                     break


        if not candidates:
            return best_x, best_f

        # Sort candidates
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Local Optimization with Nelder-Mead
        # Select top K candidates
        K = min(5, len(candidates))
        
        # Allocate remaining budget to K starts
        remaining = budget - calls
        if remaining <= 0:
            return best_x, best_f
            
        # We will run Nelder-Mead sequentially
        # Allocate more budget to better candidates
        weights = np.array([K - i + 1 for i in range(K)], dtype=float)
        weights /= weights.sum()
        budgets = (remaining * weights).astype(int)
        
        # Ensure at least some budget for each
        # Adjust if necessary
        for i in range(K):
            if budgets[i] < 10:
                # Steal from the last one
                deficit = 10 - budgets[i]
                if budgets[-1] > deficit:
                    budgets[-1] -= deficit
                    budgets[i] = 10
        
        for i in range(K):
            if calls >= budget:
                break
                
            x0 = candidates[i][1].copy()
            f0 = candidates[i][0]
            
            # Run Nelder-Mead
            nm_budget = budgets[i]
            nm_x, nm_f = self._nelder_mead(func, eval_func, x0, f0, nm_budget, dim, start_time)
            
            # Update global best (already done in eval_func, but just in case)
            if nm_f < best_f:
                best_f = nm_f
                best_x = nm_x.copy()
                
        return best_x, best_f

    def _nelder_mead(self, func, eval_func, x0, f0, budget, dim, start_time):
        """
        Custom Nelder-Mead implementation with adaptive reset.
        """
        # Initialize simplex
        # x0 is the best point
        # Other points are x0 + delta * e_i
        delta = 0.5
        simplex = [x0.copy()]
        for i in range(dim):
            x = x0.copy()
            x[i] += delta
            simplex.append(x)
        
        # Evaluate simplex
        simplex_f = [f0]
        for i in range(1, dim + 1):
            fx = eval_func(simplex[i])
            simplex_f.append(fx)
            
        best_x = x0.copy()
        best_f = f0
        
        # Main loop
        while calls < budget and (time.time() - start_time) < 290:
            # Sort simplex by fitness
            idx = np.argsort(simplex_f)
            simplex = [simplex[i] for i in idx]
            simplex_f = [simplex_f[i] for i in idx]
            
            # Update best
            if simplex_f[0] < best_f:
                best_f = simplex_f[0]
                best_x = simplex[0].copy()
                
            # Check for convergence/stagnation
            # If the difference between best and worst is small, or simplex is small
            f_range = simplex_f[-1] - simplex_f[0]
            x_range = np.max(simplex, axis=0) - np.min(simplex, axis=0)
            x_size = np.linalg.norm(x_range)
            
            if f_range < 1e-8 or x_size < 1e-6:
                # Reset simplex
                # Expand around best_x
                delta = max(0.5, delta * 2)
                simplex = [best_x.copy()]
                for i in range(dim):
                    x = best_x.copy()
                    x[i] += delta
                    simplex.append(x)
                simplex_f = [eval_func(simplex[0])]
                for i in range(1, dim + 1):
                    simplex_f.append(eval_func(simplex[i]))
                continue
            
            # Centroid of the worst point excluded
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            alpha = 1.0
            x_reflect = centroid + alpha * (centroid - simplex[-1])
            x_reflect = np.clip(x_reflect, -5, 5)
            f_reflect = eval_func(x_reflect)
            
            if f_reflect < simplex_f[0]:
                # Expansion
                gamma = 2.0
                x_expand = centroid + gamma * (centroid - simplex[-1])
                x_expand = np.clip(x_expand, -5, 5)
                f_expand = eval_func(x_expand)
                
                if f_expand < f_reflect:
                    simplex[-1] = x_expand
                    simplex_f[-1] = f_expand
                else:
                    simplex[-1] = x_reflect
                    simplex_f[-1] = f_reflect
            elif f_reflect < simplex_f[-2]:
                # Accept reflection
                simplex[-1] = x_reflect
                simplex_f[-1] = f_reflect
            else:
                # Contraction
                beta = 0.5
                # Outer contraction
                x_contract = centroid + beta * (simplex[-1] - centroid)
                x_contract = np.clip(x_contract, -5, 5)
                f_contract = eval_func(x_contract)
                
                if f_contract < simplex_f[-1]:
                    simplex[-1] = x_contract
                    simplex_f[-1] = f_contract
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex_f[i] = eval_func(simplex[i])
        
        return best_x, best_f
