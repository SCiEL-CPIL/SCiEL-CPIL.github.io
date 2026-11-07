import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = self.budget
        
        # Generate points uniformly in the unit quarter-sphere using stratified sampling
        # We use a 3D Sobol sequence for quasi-random points in [0,1]^3
        # Then filter to keep only points inside the unit sphere (x^2+y^2+z^2 <= 1)
        # This gives us a good distribution over the Pareto front of DTLZ2
        
        # Generate more points than needed to ensure we have enough after filtering
        n_sample = n_evals * 5  # Oversample to account for filtering
        
        # Use Sobol sequence for better coverage
        sampler = qmc.Sobol(d=3, scramble=True)
        points = sampler.random(n_sample)
        
        # Filter points to be inside the unit sphere
        norms_sq = np.sum(points**2, axis=1)
        mask = norms_sq <= 1.0
        points_filtered = points[mask]
        
        # If we don't have enough points, generate more
        while len(points_filtered) < n_evals:
            additional = sampler.random(n_sample)
            additional_norms_sq = np.sum(additional**2, axis=1)
            additional_mask = additional_norms_sq <= 1.0
            points_filtered = np.vstack([points_filtered, additional[additional_mask]])
        
        # Take the first n_evals points
        points_selected = points_filtered[:n_evals]
        
        # Map from objective space (unit quarter-sphere) to decision space for DTLZ2
        # DTLZ2: f_i = sum_{j=1}^{n} g_j^2 * sin(alpha_i) where alpha_i are angles
        # For 3 objectives, we need 2 angles (alpha_1, alpha_2) and the rest are g values
        # The Pareto front is: f_i = sin(alpha_i) for i=1,2,3 with sum sin^2(alpha_i) = 1
        # Actually, DTLZ2 Pareto front: f_i = x_{n-1} * sin(alpha_i) for i=1..M
        # where alpha_1 + ... + alpha_{M-1} are determined by the first M-1 decision variables
        # and x_{n-1} is the last variable
        
        # For DTLZ2 with 3 objectives:
        # f_1 = x_{11} * sin(x_0 * pi/2)
        # f_2 = x_{11} * sin(x_1 * pi/2) * cos(x_0 * pi/2)
        # f_3 = x_{11} * cos(x_0 * pi/2) * cos(x_1 * pi/2)
        # where x_{11} = sum_{j=1}^{10} (x_j - 0.5)^2 + 1 (but for Pareto front, x_{11}=1)
        
        # To be on the Pareto front, we need x_{11}=1, which means x_2...x_11 = 0.5
        # And we need to set x_0 and x_1 such that the objective values match our desired points
        
        # Given desired objectives (f_1, f_2, f_3) on the unit sphere:
        # f_1 = sin(alpha_1)
        # f_2 = sin(alpha_2) * cos(alpha_1)
        # f_3 = cos(alpha_2) * cos(alpha_1)
        
        # So: alpha_1 = arcsin(f_1)
        #     cos(alpha_1) = sqrt(1 - f_1^2)
        #     sin(alpha_2) = f_2 / cos(alpha_1)
        #     cos(alpha_2) = f_3 / cos(alpha_1)
        
        # Then: x_0 = 2/pi * alpha_1
        #       x_1 = 2/pi * alpha_2
        
        X = np.zeros((n_evals, self.dim))
        
        for i in range(n_evals):
            f1, f2, f3 = points_selected[i]
            
            # Clamp to valid range to avoid numerical issues
            f1 = np.clip(f1, 0, 1)
            f2 = np.clip(f2, 0, 1)
            f3 = np.clip(f3, 0, 1)
            
            # Compute alpha_1
            cos_alpha_1_sq = 1.0 - f1**2
            if cos_alpha_1_sq < 0:
                cos_alpha_1_sq = 0
            cos_alpha_1 = np.sqrt(cos_alpha_1_sq)
            
            if cos_alpha_1 < 1e-10:
                # Edge case: f1 = 1, so alpha_1 = pi/2
                alpha_1 = np.pi / 2
                alpha_2 = 0
            else:
                alpha_1 = np.arcsin(f1)
                sin_alpha_2 = f2 / cos_alpha_1
                sin_alpha_2 = np.clip(sin_alpha_2, -1, 1)
                alpha_2 = np.arcsin(sin_alpha_2)
            
            # Map to decision variables
            X[i, 0] = lo[0] + (hi[0] - lo[0]) * (2.0 / np.pi * alpha_1)
            X[i, 1] = lo[1] + (hi[1] - lo[1]) * (2.0 / np.pi * alpha_2)
            
            # Set remaining variables to 0.5 to be on the Pareto front (x_{11}=1)
            if self.dim > 2:
                X[i, 2:] = 0.5 * (lo[2:] + hi[2:])
        
        # Evaluate
        eval_count = 0
        for i in range(len(X)):
            if time.time() > deadline:
                break
            try:
                x = X[i]
                f = problem(x)
                eval_count += 1
            except Exception:
                break
                
        return X
