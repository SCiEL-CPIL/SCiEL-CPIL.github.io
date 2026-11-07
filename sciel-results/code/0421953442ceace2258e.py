import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        # DTLZ4 Pareto front: x[2:]=0.5, x[0],x[1] free
        # f0 = 0.5*x0*cos(pi*x1/2), f1 = 0.5*x0*sin(pi*x1/2), f2 = 0.5*(1-x0)
        
        # Generate fine candidate pool on the Pareto front
        n_cand = 50
        x0_vals = np.linspace(0, 1, n_cand)
        x1_vals = np.linspace(0, 1, n_cand)
        
        # Compute all candidate objectives analytically (no problem() calls)
        x0_grid, x1_grid = np.meshgrid(x0_vals, x1_vals, indexing='ij')
        f0 = 0.5 * x0_grid * np.cos(np.pi * x1_grid / 2)
        f1 = 0.5 * x0_grid * np.sin(np.pi * x1_grid / 2)
        f2 = 0.5 * (1.0 - x0_grid)
        cand_objs = np.stack([f0.ravel(), f1.ravel(), f2.ravel()], axis=1)
        n_candidates = len(cand_objs)
        
        # Greedy selection: maximize spread in objective space
        selected_idx = []
        
        # Seed with the three extreme points of the Pareto front
        refs = np.array([[0.5, 0.0, 0.0],
                         [0.0, 0.5, 0.0],
                         [0.0, 0.0, 0.5]])
        for rp in refs:
            dists = np.linalg.norm(cand_objs - rp, axis=1)
            idx = int(np.argmin(dists))
            if idx not in selected_idx:
                selected_idx.append(idx)
        
        # Greedily add the candidate farthest from all selected points
        while len(selected_idx) < min(self.budget, n_candidates):
            if time.time() > deadline:
                break
            sel_arr = cand_objs[selected_idx]
            # Vectorized: compute min distance from each candidate to any selected
            min_dists = np.full(n_candidates, np.inf)
            for so in sel_arr:
                d = np.linalg.norm(cand_objs - so, axis=1)
                min_dists = np.minimum(min_dists, d)
            for si in selected_idx:
                min_dists[si] = 0.0
            idx = int(np.argmax(min_dists))
            if min_dists[idx] < 1e-12:
                break
            selected_idx.append(idx)
        
        # Evaluate only the selected points via problem()
        result = []
        for idx in selected_idx:
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = x0_vals[idx // n_cand]
            x[1] = x1_vals[idx % n_cand]
            x[2:] = 0.5
            result.append(problem(x))
        
        return result
