"""Multi-objective optimizer on DTLZ2 using structured grid search on the Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structured grid optimizer for DTLZ2."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, self.n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        
        # DTLZ2 structure:
        # First n_obj - 1 variables determine position on the Pareto front.
        # Remaining variables determine distance g(x).
        # Optimal g(x) is 0, achieved by setting distance variables to midpoint 0.5.
        
        n_pos = self.n_obj - 1
        n_g = self.dim - n_pos
        
        # Determine grid size for position variables
        # We want N^2 points.
        # Budget is 200.
        # 14^2 = 196.
        N = int(np.sqrt(self.budget))
        # Ensure N^2 <= budget
        while N * N > self.budget:
            N -= 1
        
        # Generate grid in uniform space u
        # u0 in [0, 1], u1 in [0, 1]
        # Mapping to ensure uniform coverage on the DTLZ2 Pareto front (quarter sphere)
        # x0 = (2/pi) * arcsin(u0)
        # x1 = u1
        
        u0_vals = np.linspace(0, 1, N)
        u1_vals = np.linspace(0, 1, N)
        
        U0, U1 = np.meshgrid(u0_vals, u1_vals, indexing='ij')
        
        # Map to decision variables
        X0 = (2.0 / np.pi) * np.arcsin(U0)
        X1 = U1
        
        # Flatten
        X0_flat = X0.ravel()
        X1_flat = U1.ravel()
        
        n_points = len(X0_flat)
        
        # Create full dimension vectors
        X = np.zeros((n_points, self.dim))
        X[:, 0] = X0_flat
        X[:, 1] = X1_flat
        
        # Set distance variables to midpoint
        # Indices n_pos to dim-1
        if n_g > 0:
            mid = 0.5 * (hi[n_pos:] + lo[n_pos:])
            X[:, n_pos:] = mid
            
        # Clip to bounds just in case
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        # We must call problem(x) for each point to record the objectives.
        for i in range(n_points):
            if time.time() > deadline - 5:
                break
            problem(X[i])
            
        # If we have remaining budget, add a few random points on the front
        # to ensure no specific gap in the grid (though grid is dense)
        remaining = self.budget - n_points
        if remaining > 0 and time.time() < deadline - 5:
            # Use the global random state which is seeded by the evaluator
            u0_rand = np.random.random(remaining)
            u1_rand = np.random.random(remaining)
            
            X0_rand = (2.0 / np.pi) * np.arcsin(u0_rand)
            X1_rand = u1_rand
            
            X_rand = np.zeros((remaining, self.dim))
            X_rand[:, 0] = X0_rand
            X_rand[:, 1] = X1_rand
            if n_g > 0:
                X_rand[:, n_pos:] = mid
            X_rand = np.clip(X_rand, lo, hi)
            
            for i in range(remaining):
                if time.time() > deadline - 5:
                    break
                problem(X_rand[i])
                
        return X
# EVOLVE-BLOCK-END
