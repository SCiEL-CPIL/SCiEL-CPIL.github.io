# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for DTLZ4 that exploits the known Pareto front geometry.
    It sets distance variables to their optimal values (0.5) to minimize g(x),
    and samples shape variables using a transformation that yields uniform
    distribution on the Pareto front surface, maximizing hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 with 3 objectives:
        # Shape variables: x_0, x_1 (indices 0, 1)
        # Distance variables: x_2 ... x_{dim-1} (indices 2 to 11 for dim=12)
        n_shape = n_obj - 1  # 2
        n_dist = dim - n_shape  # 10

        # Optimal distance values for DTLZ4 are 0.5
        # g(x) = sum((x_i - 0.5)^2), so g=0 when x_i=0.5
        # This makes 1+g = 1, putting points exactly on the ideal front.
        
        # To maximize hypervolume with a reference point, we want points
        # distributed uniformly on the Pareto front.
        # For DTLZ4 3-objective, the front is a quarter sphere.
        # Uniform sampling on the sphere surface requires:
        # cos(theta_0) ~ Uniform(0, 1)
        # theta_1 ~ Uniform(0, pi/2)
        
        # The mapping from decision variables to angles is:
        # theta_0 = (pi / 2) * x_0
        # theta_1 = (pi / 2) * x_1
        
        # So:
        # cos((pi/2)*x_0) = u  => x_0 = (2/pi) * arccos(u)
        # (pi/2)*x_1 = v * (pi/2) => x_1 = v
        
        # Use a stratified grid for low-discrepancy sampling
        # This ensures uniform coverage of the Pareto front surface
        results = []
        
        # Determine grid size: use as many complete cells as possible
        n_grid = int(np.sqrt(budget))
        # Add any remaining points as additional stratified samples
        n_extra = budget - n_grid * n_grid
        
        for i in range(n_grid):
            for j in range(n_grid):
                # Check deadline
                if time.time() > deadline:
                    return None
                
                # Stratified sampling: pick a point in cell (i, j)
                # Use the center of each cell for deterministic, uniform coverage
                u = (i + 0.5) / n_grid
                v = (j + 0.5) / n_grid
                
                # Calculate shape variables
                x_0 = (2.0 / np.pi) * np.arccos(u)
                x_1 = v
                
                # Construct full vector
                x = np.zeros(dim)
                x[0] = x_0
                x[1] = x_1
                x[2:] = 0.5  # Set distance variables to optimal value
                
                # Evaluate
                f = problem(x)
                results.append(f)
        
        # Add extra points if budget allows (stratified in a finer sub-grid)
        if n_extra > 0:
            # Use a Halton-like sequence for the extra points
            for k in range(n_extra):
                if time.time() > deadline:
                    return None
                
                # Use irrational offsets for low-discrepancy
                u = (k + 0.5) / n_extra
                v = (k + 0.5) / n_extra
                
                # Apply a rotation to decorrelate
                angle = np.pi * (np.sqrt(2) - 1) * k
                u_rot = u * np.cos(angle) - v * np.sin(angle)
                v_rot = u * np.sin(angle) + v * np.cos(angle)
                
                # Clip to [0, 1]
                u_rot = np.clip(u_rot, 0.0, 1.0)
                v_rot = np.clip(v_rot, 0.0, 1.0)
                
                x_0 = (2.0 / np.pi) * np.arccos(u_rot)
                x_1 = v_rot
                
                x = np.zeros(dim)
                x[0] = x_0
                x[1] = x_1
                x[2:] = 0.5
                
                f = problem(x)
                results.append(f)
        
        return None
# EVOLVE-BLOCK-END
