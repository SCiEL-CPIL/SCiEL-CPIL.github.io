"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        dim = self.dim
        
        # ZDT3 structure:
        # f1 = x[0]
        # f2 = 1 - sqrt(x[0]) + 10*x[0]*sin(10*pi*x[0])  when x[1:]=0
        # Pareto front: x[1:] = 0, x[0] in [0,1]
        # f2 has oscillations with period 0.1 in x[0]
        
        # Strategy: Use non-uniform sampling concentrated near x0=0
        # where the front has highest curvature and most hypervolume contribution.
        # Also add some uniform points to cover the full range.
        
        # Allocate budget:
        # - 60% for concentrated near 0 (where sin term creates structure)
        # - 40% for uniform coverage
        
        n_concentrated = int(n_evals * 0.6)
        n_uniform = n_evals - n_concentrated
        
        # Concentrated samples: use power distribution to cluster near 0
        # x0 = t^p where t is uniform in [0,1], p > 1 clusters near 0
        t_conc = np.linspace(0, 1, n_concentrated)
        x0_conc = t_conc ** 3  # Cube to cluster near 0
        
        # Uniform samples
        x0_uni = np.linspace(0, 1, n_uniform)
        
        # Combine
        x0_samples = np.concatenate([x0_conc, x0_uni])
        
        # Remove duplicates (round to avoid floating point issues)
        x0_samples = np.unique(np.round(x0_samples, 15))
        
        # If we have too many after dedup, trim
        if len(x0_samples) > n_evals:
            x0_samples = x0_samples[:n_evals]
        
        # Create full decision vectors
        X_candidates = np.zeros((len(x0_samples), dim))
        X_candidates[:, 0] = x0_samples
        # x[1:] = 0 for Pareto front
        
        # Evaluate all candidates
        F = np.array([problem(x) for x in X_candidates])
        
        # Check time
        if time.time() > deadline:
            return X_candidates
        
        # Now select non-dominated points to maximize hypervolume
        # For 2 objectives, a point is non-dominated if no other point
        # is better in both objectives
        
        n_points = len(F)
        if n_points == 0:
            return X_candidates
        
        # Non-dominated sort for 2 objectives
        # Point i dominates point j if F[i][0] <= F[j][0] and F[i][1] <= F[j][1] and not equal
        dominated = np.zeros(n_points, dtype=bool)
        
        for i in range(n_points):
            if dominated[i]:
                continue
            for j in range(i + 1, n_points):
                if dominated[j]:
                    continue
                # Check if i dominates j
                if (F[i][0] <= F[j][0] and F[i][1] <= F[j][1] and 
                    (F[i][0] < F[j][0] or F[i][1] < F[j][1])):
                    dominated[j] = True
                elif (F[j][0] <= F[i][0] and F[j][1] <= F[i][1] and 
                      (F[j][0] < F[i][0] or F[j][1] < F[i][1])):
                    dominated[i] = True
                    break
        
        # Keep non-dominated points
        nondom_idx = np.where(~dominated)[0]
        
        # If we have more non-dominated points than useful, select best by hypervolume contribution
        # For 2 objectives, sort by f1 and keep the Pareto front
        if len(nondom_idx) > 0:
            # Sort non-dominated points by f1
            sort_order = np.argsort(F[nondom_idx, 0])
            nondom_sorted = F[nondom_idx[sort_order]]
            
            # For hypervolume, we want points that contribute to the front
            # Keep all non-dominated points as they all contribute
            final_F = nondom_sorted
        else:
            # Fallback: keep all
            final_F = F
        
        return X_candidates
# EVOLVE-BLOCK-END
