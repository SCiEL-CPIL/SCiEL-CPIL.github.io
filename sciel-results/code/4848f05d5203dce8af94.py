"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-objective optimizer for RE34 using a hybrid of stratified sampling and gradient-based local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        width = hi - lo
        
        # Reference point for scaling
        ref = np.array([1705.0, 12.0, 0.3])
        
        X_archive = []
        F_archive = []
        eval_count = 0
        
        def safe_problem(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                return [1e10, 1e10, 1e10]
            
            f = problem(x)
            eval_count += 1
            X_archive.append(x)
            F_archive.append(f)
            return f

        # 1. Initial Stratified Sampling (Latin Hypercube-like)
        # This provides better initial coverage than pure random, especially for hypervolume.
        n_init = 20
        if n_init > self.budget:
            n_init = self.budget
            
        # Generate LHS points
        # For simplicity and speed, we use random permutations for each dimension
        lhs_points = np.zeros((n_init, self.dim))
        for d in range(self.dim):
            # Create bins
            edges = np.linspace(0, 1, n_init + 1)
            # Randomly permute the bins
            perm = np.random.permutation(n_init)
            # Pick a random point within each bin
            points = (edges[perm] + np.random.rand(n_init)) / n_init
            # Actually, standard LHS: pick uniform in each bin
            points = (edges[perm] + np.random.rand(n_init)) / n_init
            # Wait, edges are 0..1/n_init. So point is in [i/n, (i+1)/n].
            # Correct formula: (perm + rand) / n_init
            points = (perm + np.random.rand(n_init)) / n_init
            lhs_points[:, d] = points
            
        X_init = lo + lhs_points * width
        
        for x in X_init:
            if eval_count >= self.budget:
                break
            safe_problem(x)
            
        if eval_count >= self.budget:
            return X_archive

        # 2. Generate Weight Vectors
        # We use a combination of extreme, mid-edge, and random weights.
        # Scaling is crucial. We use the reference point to normalize the objectives
        # so that the weighted sum is balanced.
        weights = []
        
        # Extreme points
        for i in range(self.n_obj):
            w = np.zeros(self.n_obj)
            w[i] = 1.0
            weights.append(w)
            
        # Mid-edge points
        for i in range(self.n_obj):
            for j in range(self.n_obj):
                if i < j:
                    w = np.zeros(self.n_obj)
                    w[i] = 0.5
                    w[j] = 0.5
                    weights.append(w)
                    
        # Center point
        weights.append(np.ones(self.n_obj) / self.n_obj)
        
        # Add random weights for diversity
        n_rand_weights = 10
        for _ in range(n_rand_weights):
            weights.append(np.random.dirichlet(np.ones(self.n_obj)))
            
        np.random.shuffle(weights)
        
        # 3. Refinement with L-BFGS-B
        remaining_budget = self.budget - eval_count
        n_weights = len(weights)
        
        # Reserve some budget for safety and potential final random fill
        safe_budget = remaining_budget - 5
        if safe_budget < 10:
             # If budget is too low, just do random search
             while eval_count < self.budget:
                 x = lo + (hi - lo) * np.random.rand(self.dim)
                 safe_problem(x)
             return X_archive

        # Allocate budget per weight
        # We want to ensure we don't run out of budget in the middle of an optimization
        # L-BFGS-B uses ~1-2 function calls per iteration.
        budget_per_weight = max(5, safe_budget // n_weights)
        
        for w in weights:
            if eval_count >= self.budget - 2: # Keep 2 evals in reserve
                break
            if time.time() > deadline:
                break
                
            # Effective weights for scaling
            # We divide by the reference point to normalize the magnitude of each objective
            w_eff = w / ref
            
            # Find best starting point in current archive for this direction
            if F_archive:
                F_arr = np.array(F_archive)
                # Use scaled objectives for comparison
                F_scaled = F_arr / ref
                scores = F_scaled @ w
                best_idx = np.argmin(scores)
                x0 = X_archive[best_idx].copy()
            else:
                x0 = (lo + hi) / 2.0
                
            def scalar_obj(x):
                f = safe_problem(x)
                # Use scaled objective for optimization
                return np.dot(w_eff, f)
                
            try:
                # Limit iterations to stay within budget
                # maxiter is number of iterations, not function calls.
                # Assume 2 function calls per iteration.
                max_iter = max(1, budget_per_weight // 2)
                
                res = minimize(scalar_obj, x0, method='L-BFGS-B', 
                               bounds=bounds, 
                               options={'maxiter': max_iter, 'maxls': 20})
                
            except Exception:
                pass

        # 4. Final Random Fill if budget remains
        # Use random sampling to fill any remaining budget.
        # This can sometimes find better points if the local searches got stuck.
        while eval_count < self.budget:
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * np.random.rand(self.dim)
            safe_problem(x)

        return X_archive
# EVOLVE-BLOCK-END
