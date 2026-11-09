"""Stochastic local search optimizer for ZDT4."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for ZDT4 using stochastic local search.
    
    Strategy:
    1. Initialize a set of random points within bounds.
    2. For each point, perform local search by perturbing variables and
       accepting moves that improve the sum of objectives (or dominance).
    3. Keep track of the best points found.
    4. Use all budget for this search process.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        all_points = []
        
        # Check if we have time to proceed
        if time.time() > deadline:
            return all_points
        
        # Stochastic local search
        # We will use a portion of the budget for initialization and the rest for local search.
        # Let's allocate 10% for initial random points and 90% for local search steps.
        n_initial = max(1, budget // 10)
        n_search = budget - n_initial
        
        # Initialize random points
        current_points = []
        current_objs = []
        
        for i in range(n_initial):
            x = np.random.uniform(lo, hi)
            f = problem(x)
            f = np.asarray(f, dtype=float)
            current_points.append(x.copy())
            current_objs.append(f.copy())
            all_points.append(f.copy())
        
        if len(all_points) >= budget:
            return all_points
        
        # Local search phase
        # For each initial point, try to improve it
        for i in range(n_initial):
            if time.time() > deadline:
                break
            
            x = current_points[i].copy()
            f = current_objs[i].copy()
            best_x = x.copy()
            best_f = f.copy()
            
            # Try to improve this point
            for step in range(n_search // n_initial):
                if time.time() > deadline:
                    break
                
                # Perturb a random variable
                j = np.random.randint(0, dim)
                x_new = x.copy()
                
                # Gaussian perturbation
                x_new[j] += np.random.normal(0, (hi[j] - lo[j]) * 0.1)
                x_new[j] = np.clip(x_new[j], lo[j], hi[j])
                
                f_new = problem(x_new)
                f_new = np.asarray(f_new, dtype=float)
                
                # Accept if better in sum of objectives (simple scalarization)
                # Or if it dominates
                if np.sum(f_new) < np.sum(f):
                    x = x_new
                    f = f_new
                    best_x = x.copy()
                    best_f = f.copy()
                    all_points.append(f.copy())
                    
                    if len(all_points) >= budget:
                        return all_points
                
                # Also record the new point even if not better, to explore
                # But we must be careful not to exceed budget
                # Actually, let's only record improvements to keep quality high
                # Or record all to get a denser front?
                # The task says "Every returned objective vector is recorded"
                # So every call to problem() adds to all_points.
                # We need to be strategic.
        
        return all_points
# EVOLVE-BLOCK-END
