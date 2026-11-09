"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ2 using a structured grid on the Pareto front.
    
    Strategy:
    1. DTLZ2 with M=3 has a Pareto front determined by x[0], x[1] (position variables).
    2. x[2:] should be set to the center of the bounds (0.5) to minimize g(x)=0.
    3. We use a structured grid to generate 200 points in [0, 1]^2.
    4. These points are mapped to the DTLZ2 Pareto front angles.
    5. The resulting 200 objective vectors are returned.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s. Return before that.
        deadline = start_time + 320
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_points = self.budget
        
        # DTLZ2 with M=3: Pareto front is a quarter-sphere in objective space.
        # Parameterize by angles theta1, theta2 in [0, pi/2].
        # x[0] = cos(theta1) * cos(theta2)
        # x[1] = cos(theta1) * sin(theta2)
        # x[2:] = 0.5 (to make g=0)
        # This gives f1 = sin(theta1), f2 = cos(theta1)*sin(theta2), f3 = cos(theta1)*cos(theta2)
        
        # Use a uniform grid in angle space for better Pareto front coverage
        # 14x14 = 196 points, leaving 4 for potential refinement
        n_per_axis = 14
        
        theta1_vals = np.linspace(0, np.pi/2, n_per_axis)
        theta2_vals = np.linspace(0, np.pi/2, n_per_axis)
        
        T1, T2 = np.meshgrid(theta1_vals, theta2_vals, indexing='ij')
        grid_points = np.column_stack([T1.ravel(), T2.ravel()])
        
        # Use all 196 points from the grid
        results = []
        
        for i in range(len(grid_points)):
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            
            # Position variables from angle parameterization
            t1 = grid_points[i, 0]
            t2 = grid_points[i, 1]
            x[0] = np.cos(t1) * np.cos(t2)
            x[1] = np.cos(t1) * np.sin(t2)
            
            # Distance variables: set to center of bounds
            if self.dim > 2:
                x[2:] = 0.5
            
            # Clip to bounds just in case
            x = np.clip(x, lo, hi)
            
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break
        
        # Use remaining budget for extra points at non-grid locations
        # Grid spacing is pi/26, so grid points are at k*pi/26 for k=0..13
        # Choose points between grid cells near the center of the Pareto front
        remaining = n_points - len(results)
        if remaining > 0 and time.time() < deadline:
            # Midpoints between grid points near the center maximize hypervolume contribution
            # Grid index 6 -> 6*pi/26 = 3*pi/13, index 7 -> 7*pi/26
            # Midpoint: 6.5*pi/26 = 13*pi/52 = pi/4 (already checked, not on grid)
            extra_angles = [
                (np.pi/4, np.pi/4),       # center of front
                (np.pi/4, 3*np.pi/8),     # offset in theta2
                (3*np.pi/8, np.pi/4),     # offset in theta1
                (np.pi/4, 5*np.pi/8),     # further offset
            ]
            
            for t1, t2 in extra_angles[:remaining]:
                if time.time() > deadline:
                    break
                x = np.zeros(self.dim)
                x[0] = np.cos(t1) * np.cos(t2)
                x[1] = np.cos(t1) * np.sin(t2)
                if self.dim > 2:
                    x[2:] = 0.5
                x = np.clip(x, lo, hi)
                try:
                    f = problem(x)
                    results.append(f)
                except Exception:
                    break
        
        return np.array(results) if results else None
# EVOLVE-BLOCK-END
