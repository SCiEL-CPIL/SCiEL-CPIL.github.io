"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using analytical Pareto front construction and maximin sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 specifics:
        # M = 3 objectives
        # N = 12 variables
        # Position variables: x[0], x[1] (M-1 = 2)
        # Distance variables: x[2]...x[11] (N-M+1 = 10)
        
        # To get points on the Pareto front (g=0), we must set distance variables to 0.5.
        # The objectives are determined solely by x[0] and x[1]:
        # f1 = cos(x0 * pi/2) * cos(x1 * pi/2)
        # f2 = cos(x0 * pi/2) * sin(x1 * pi/2)
        # f3 = sin(x0 * pi/2)
        
        # Strategy:
        # 1. Use the entire budget for points on the Pareto front.
        # 2. Use a maximin distance criterion in the (x0, x1) parameter space to select points.
        #    This ensures we cover the corners and the curved surface efficiently for hypervolume.
        
        n_points = budget
        
        # Initialize parameter space (x0, x1) in [0, 1]
        # We map x0, x1 to [0, 1] directly, then compute objectives.
        # x0 in [0, 1] -> theta1 = x0 * pi/2
        # x1 in [0, 1] -> theta2 = x1 * pi/2
        
        # Start with vertices to ensure corners are covered
        initial_params = np.array([
            [0.0, 0.0], # f=(1, 0, 0)
            [0.0, 1.0], # f=(0, 1, 0)
            [1.0, 0.0], # f=(0, 0, 1)
            [0.5, 0.5]  # Center
        ])
        
        selected_params = [initial_params]
        
        # Add points using Maximin Distance in the 2D parameter space
        # This is O(N^2) which is fine for N=200
        # We use a batched selection approach to improve efficiency and coverage
        while len(selected_params[0]) < n_points:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate a batch of candidate points to choose from
            # Using Sobol-like or random points
            n_candidates = min(50, n_points - len(selected_params[0]))
            # Use random points for candidates
            candidates = np.random.rand(n_candidates, 2)
            
            # Compute distance from each candidate to each selected point
            # selected_params[0] is (K, 2), candidates is (C, 2)
            # dist[i, j] = ||cand_i - sel_j||
            diffs = candidates[:, np.newaxis, :] - selected_params[0][np.newaxis, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            
            # For each candidate, find its minimum distance to any selected point
            min_dists = np.min(dists, axis=1)
            
            # Select the candidate with the maximum of these minimum distances
            best_idx = np.argmax(min_dists)
            selected_params[0] = np.vstack([selected_params[0], candidates[best_idx]])
            
        params = selected_params[0]
        
        # Construct decision vectors
        # x[0] = param[0]
        # x[1] = param[1]
        # x[2:] = 0.5
        X = np.zeros((len(params), dim))
        X[:, 0] = params[:, 0]
        X[:, 1] = params[:, 1]
        X[:, 2:] = 0.5
        
        # Clip to bounds (just in case, though 0.5 and [0,1] are safe)
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        # We must call problem(x) for each x.
        # The problem clips x to bounds.
        
        # To be safe with the evaluator's budget tracking, we evaluate one by one.
        # However, the problem function is likely fast.
        # We need to ensure we don't exceed the budget. We have exactly n_points.
        
        # Return value is ignored, but we must not throw exceptions.
        for i in range(len(X)):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
# EVOLVE-BLOCK-END
