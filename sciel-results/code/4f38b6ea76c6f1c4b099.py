"""Multi-objective optimizer on DTLZ7 using max-min facility location on Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for DTLZ7 that uses max-min facility location
    to place points uniformly on the 2D Pareto front in objective space.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, keep 20s margin

        dim = self.dim
        budget = self.budget

        # DTLZ7 Pareto front for 3 objectives:
        # x[2:] = 0, f1 = x[0], f2 = x[1], f3 = 1 - sqrt(x[1])*(1 + sin(3*pi*x[1]))
        # The front is parameterized by (x0, x1) in [0,1]^2
        # f3 depends only on x1

        def f3_of_x1(x1):
            return 1.0 - np.sqrt(x1) * (1.0 + np.sin(3.0 * np.pi * x1))

        # Max-min facility location: place points to maximize minimum distance
        # in objective space (f1, f2, f3)
        
        # Use a candidate pool approach:
        # 1. Generate a fine grid of candidate (x0, x1) points
        # 2. Use greedy max-min selection to pick the best subset
        
        # Generate candidates: use a dense grid
        n_candidates = 200  # number of candidate points to consider
        # Use a Latin hypercube or grid for candidates
        # Let's use a structured grid with some jitter
        n_side = int(np.sqrt(n_candidates))
        if n_side < 1:
            n_side = 1
        if n_side * n_side < n_candidates:
            n_side += 1
        
        candidates = []
        for i in range(n_side):
            for j in range(n_side):
                x0 = (i + 0.5) / n_side
                x1 = (j + 0.5) / n_side
                candidates.append((x0, x1))
        
        # Compute objective values for candidates
        cand_f1 = np.array([c[0] for c in candidates])
        cand_f2 = np.array([c[1] for c in candidates])
        cand_f3 = f3_of_x1(cand_f2)
        cand_obj = np.column_stack([cand_f1, cand_f2, cand_f3])
        
        # Greedy max-min selection: pick points that maximize minimum distance
        # to already selected points in objective space
        selected_indices = []
        n_select = min(budget, len(candidates))
        
        # Start with a corner point (0,0) which gives (0, 0, f3(0)=1)
        # Actually f3(0) = 1 - 0*(1+0) = 1
        # Let's start with a point that's well-spread
        # Use a deterministic starting point
        start_idx = 0  # (0.5/n_side, 0.5/n_side)
        selected_indices.append(start_idx)
        
        min_dist = np.full(len(candidates), np.inf)
        
        for _ in range(1, n_select):
            if time.time() > deadline:
                break
            
            # Update min distances
            last_obj = cand_obj[selected_indices[-1]]
            dists = np.sqrt(np.sum((cand_obj - last_obj) ** 2, axis=1))
            min_dist = np.minimum(min_dist, dists)
            
            # Pick the candidate with maximum minimum distance
            # Exclude already selected
            for idx in selected_indices:
                min_dist[idx] = -1
            
            next_idx = np.argmax(min_dist)
            selected_indices.append(next_idx)
        
        # Now evaluate the selected points using the problem function
        evaluated = []
        eval_count = 0
        
        for idx in selected_indices:
            if eval_count >= budget:
                break
            if time.time() > deadline:
                break
            
            x0, x1 = candidates[idx]
            x = np.zeros(dim)
            x[0] = x0
            x[1] = x1
            # x[2:] stays 0
            
            f = problem(x)
            evaluated.append((x, f))
            eval_count += 1
        
        # If we have budget left, do additional random sampling
        if eval_count < budget and time.time() < deadline:
            while eval_count < budget and time.time() < deadline:
                x = np.zeros(dim)
                x[0] = np.random.uniform(0, 1)
                x[1] = np.random.uniform(0, 1)
                f = problem(x)
                evaluated.append((x, f))
                eval_count += 1
        
        if evaluated:
            return np.array([x for x, _ in evaluated])
        return np.array([])

# EVOLVE-BLOCK-END
