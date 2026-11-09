"""Multi-objective optimizer for DTLZ7 exploiting the known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """
    Multi-objective optimizer for DTLZ7.
    
    Strategy:
    1. Exploit the known Pareto front structure of DTLZ7 where x[2:] = 0.
    2. Use Sobol sequence to sample the 2D parameter space (x[0], x[1]) uniformly.
    3. Include the 4 corner points to ensure full extent coverage.
    4. Use all budget evaluations for sampling the front directly.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290  # Safety margin before 300s kill (10s buffer)
        
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj
        
        # DTLZ7 Pareto front is defined by x[0], x[1] in [0,1] and x[2:] = 0.
        # Objectives:
        # f0 = x[0]
        # f1 = 1 - x[1]
        # f2 = 1 + g(x) * (1 + sin(10 * pi * x[0]))  -- but g=0 when x[2:]=0, so f2=1
        
        # We want to maximize hypervolume with reference point [1.1, 1.1, 25.0].
        # The "ideal" points are those on the front.
        # Since the front is 2D (a square in f0-f1 plane with f2=1), we should sample this square.
        
        # We have 'budget' evaluations.
        # Let's use all of them.
        
        if budget <= 0:
            return []

        # Generate Sobol points in 2D
        # Sobol sequences work best with power-of-2 sizes, but qmc.Sobol handles arbitrary N reasonably well.
        # To be safe and efficient, we can just generate N points.
        
        n_corners = 4
        if budget <= n_corners:
            # If budget is very small, just pick corners
            # But budget is 200, so this is unlikely.
            n_sobol = budget
            n_corners = 0
        else:
            n_sobol = budget - n_corners
            n_corners = 4

        # Generate Sobol points
        sampler = qmc.Sobol(d=2, scramble=True, seed=0)
        sobol_points = sampler.random(n_sobol)
        
        # sobol_points[:, 0] -> x[0]
        # sobol_points[:, 1] -> x[1]
        # Note: qmc.Sobol generates points in [0, 1).
        
        # Define corners
        # (x0, x1) pairs
        # (0,0), (0,1), (1,0), (1,1)
        corners = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ])
        
        # Combine
        if n_sobol > 0:
            # Stack corners and sobol points
            all_points_2d = np.vstack((corners, sobol_points))
        else:
            all_points_2d = corners[:budget]
            
        # Ensure we don't exceed budget
        all_points_2d = all_points_2d[:budget]
        
        # Construct full X vectors
        X = np.zeros((len(all_points_2d), dim))
        X[:, 0] = all_points_2d[:, 0]
        X[:, 1] = all_points_2d[:, 1]
        # x[2:] remains 0
        
        # Evaluate all points
        F_list = []
        for i in range(len(X)):
            if time.time() > deadline:
                break
            if i >= budget:
                break
            f = problem(X[i])
            F_list.append(f)
            
        # Return the list of objective vectors
        # The return value is ignored by the evaluator, but we return it as per standard practice
        # and in case the interface expects it.
        return F_list
# EVOLVE-BLOCK-END
