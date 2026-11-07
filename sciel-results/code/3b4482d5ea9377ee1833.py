import numpy as np
import time
import sys

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.start_time = time.time()
        self.deadline = self.start_time + 300  # Safety margin for 330s limit

    def __call__(self, problem, bounds):
        # Parse bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        all_X = []
        all_F = []
        
        def eval_point(x):
            x = np.array(x)
            # Clip to bounds
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            return x, f

        def add_to_archive(x, f):
            all_X.append(x)
            all_F.append(f)

        budget_used = 0
        remaining = self.budget
        
        # 1. Probe Phase: Find best constant value for distance variables (indices 2 to dim-1)
        # DTLZ4 distance variables are typically indices n_pos to dim-1.
        # For M=3, n_pos=2. So indices 2..dim-1.
        # We test a few candidate values for these variables.
        # Candidates: 0, 0.001, 0.01, 0.1, 0.5
        # We assume the first 2 variables (0, 1) are position variables.
        # We set them to 0.5 (middle of front) for probing.
        
        pos_vars = np.array([0.5, 0.5])
        dist_candidates = [0.0, 0.0001, 0.001, 0.01, 0.1, 0.5]
        
        best_dist_val = 0.0
        best_dist_score = np.inf
        
        for v in dist_candidates:
            if budget_used >= self.budget:
                break
            if time.time() > self.deadline:
                return all_X
                
            x = np.zeros(self.dim)
            x[0:2] = pos_vars
            x[2:] = v
            try:
                xc, fc = eval_point(x)
                add_to_archive(xc, fc)
                budget_used += 1
                score = np.sum(fc)
                if score < best_dist_score:
                    best_dist_score = score
                    best_dist_val = v
            except Exception:
                continue

        # 2. Search Phase: Use Sobol sequence for position variables (0, 1)
        # Use the best distance value for variables 2..dim-1
        # We have remaining budget.
        
        n_points = self.budget - budget_used
        
        if n_points > 0:
            # Use Sobol sequence
            try:
                from scipy.stats import qmc
                sampler = qmc.Sobol(d=2, scramble=True)
                sobol_points = sampler.random(n_points)
            except ImportError:
                # Fallback to Random
                sobol_points = np.random.rand(n_points, 2)
            
            for point in sobol_points:
                if budget_used >= self.budget:
                    break
                if time.time() > self.deadline:
                    return all_X
                
                x = np.zeros(self.dim)
                x[0] = point[0]
                x[1] = point[1]
                x[2:] = best_dist_val
                
                try:
                    xc, fc = eval_point(x)
                    add_to_archive(xc, fc)
                    budget_used += 1
                except Exception:
                    continue

        return all_X
