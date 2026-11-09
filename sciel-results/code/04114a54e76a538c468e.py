"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def halton_sequence(num_samples, dim, skip=1):
    """Generate Halton sequence points in [0, 1)^dim."""
    points = np.zeros((num_samples, dim))
    for i in range(num_samples):
        n = i + skip
        for j in range(dim):
            base = j + 2  # Prime bases: 2, 3, 5, ...
            f = 1.0
            r = 0.0
            while n > 0:
                f /= base
                r += f * (n % base)
                n //= base
            points[i, j] = r
    return points

class Optimizer:
    """Multi-Start Nelder-Mead optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin for 330s limit
        
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        
        # Storage for all evaluated points to select local search seeds
        self.X_evals = []
        self.Y_evals = []
        
        def wrapped_func(x):
            if self.call_count >= self.budget:
                return self.best_f
            self.call_count += 1
            fx = func(x)
            if fx < self.best_f:
                self.best_f = fx
                self.best_x = x.copy()
            self.X_evals.append(x.copy())
            self.Y_evals.append(fx)
            return fx

        # Phase 1: Global Exploration
        # Use a mix of Halton (structured) and Random (stochastic) to cover [-5, 5]^dim
        # Budget allocation: ~20% for global, 80% for local
        n_global = int(self.budget * 0.2)
        if n_global < 10:
            n_global = 10
            
        # Generate Halton points
        halton_pts = halton_sequence(n_global // 2, self.dim, skip=1)
        halton_pts = halton_pts * 10.0 - 5.0
        
        # Generate Random points
        rng = np.random.default_rng(42)
        random_pts = rng.uniform(-5, 5, (n_global - n_global // 2, self.dim))
        
        # Combine
        global_pts = np.vstack([halton_pts, random_pts])
        
        # Evaluate Global Points
        for i in range(len(global_pts)):
            if time.time() > deadline:
                break
            if self.call_count >= self.budget:
                break
            x = global_pts[i]
            wrapped_func(x)
            
        # Phase 2: Multi-Start Local Refinement
        # Select top K candidates from Phase 1 to start Nelder-Mead
        if self.call_count < self.budget and len(self.Y_evals) > 0:
            # Sort by function value
            sorted_indices = np.argsort(self.Y_evals)
            top_k = 5
            seeds = []
            for idx in sorted_indices[:top_k]:
                if len(seeds) >= top_k:
                    break
                seeds.append(self.X_evals[idx])
            
            # Remaining budget
            remaining_budget = self.budget - self.call_count
            if remaining_budget <= 0:
                return self.best_x, self.best_f
                
            # Allocate budget to each local search
            # We run them sequentially. If one finishes early, the next gets more.
            # Or we can set a maxiter based on remaining_budget / len(seeds)
            
            # Strategy: Run Nelder-Mead on each seed.
            # To prevent one search from eating all budget, we can limit iterations per search
            # or just let them run until budget is exhausted or convergence.
            # Given NM is fast, it will likely converge or hit maxiter quickly.
            
            # We'll iterate through seeds.
            # We need a wrapper for NM that respects the global budget.
            
            for seed in seeds:
                if self.call_count >= self.budget:
                    break
                if time.time() > deadline - 5: # Leave 5s for safety
                    break
                    
                # Initial simplex size
                # Default NM uses a simplex with initial step 0.05 or 1.0 depending on version.
                # For [-5, 5], a step of 0.5 or 1.0 is reasonable.
                # scipy minimize Nelder-Mead options:
                # xatol, fatol, maxiter, adaptive, initial_simplex
                
                # We want to be aggressive.
                # Let's use adaptive=True to help with scaling.
                
                def nm_func(x):
                    if self.call_count >= self.budget:
                        raise ValueError("Budget exceeded")
                    if time.time() > deadline:
                        raise ValueError("Time exceeded")
                    self.call_count += 1
                    fx = func(x)
                    if fx < self.best_f:
                        self.best_f = fx
                        self.best_x = x.copy()
                    self.X_evals.append(x.copy())
                    self.Y_evals.append(fx)
                    return fx
                
                try:
                    # maxiter should be limited to avoid wasting time on one seed if many remain
                    # But if we have 5 seeds and 400 calls, 80 calls per seed is ~20 iterations (simplex has 6 vertices, so ~13 function calls per iter? No, NM is ~N+1 per iter).
                    # 500 calls / 5 seeds = 100 calls per seed.
                    # NM takes ~N+1 calls per iteration. N=5, so 6 calls/iter.
                    # 100 / 6 = 16 iterations.
                    # Let's set maxiter to 20 to be safe and allow some slack.
                    
                    res = minimize(
                        nm_func, 
                        seed, 
                        method='Nelder-Mead', 
                        options={
                            'maxiter': 20, 
                            'xatol': 1e-6, 
                            'fatol': 1e-6,
                            'adaptive': True
                        }
                    )
                except ValueError:
                    # Budget or time exceeded, break
                    break
                except Exception:
                    # Any other error, skip this seed
                    continue

        return self.best_x, self.best_f

# EVOLVE-BLOCK-END
