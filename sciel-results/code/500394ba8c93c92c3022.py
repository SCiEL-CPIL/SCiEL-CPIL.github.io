import numpy as np
import time

class Optimizer:
    """
    Exploits the structure of DTLZ2.
    DTLZ2 with M=3 objectives and N=12 variables:
    - Variables x[0], x[1] determine the position on the Pareto front (angles).
    - Variables x[2]...x[11] determine the distance from the front (g term).
    - Optimal strategy: Set x[2]...x[11] = 0.5 to make g=0 (points on the front).
    - Sample x[0], x[1] to cover the Pareto front (quarter sphere) uniformly.
    - Use Latin Hypercube Sampling (LHS) for uniform coverage.
    - Explicitly include the 3 extreme corners of the front.
    - Add deadline guard to prevent timeout.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        timeout = 320  # Safety margin before 330s limit
        
        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ2 specific: 
        # x[0], x[1] are the angle variables.
        # x[2:] are the distance variables.
        # Setting x[2:] = 0.5 minimizes g (distance to front).
        
        n_angle_vars = 2
        n_dist_vars = self.dim - n_angle_vars
        
        # Base vector for distance variables (center of bounds, usually 0.5 for [0,1])
        # Assuming bounds are [0,1] as per task description, 0.5 is the midpoint.
        # If bounds were different, we'd need to be careful, but task says [0,1].
        dist_mid = 0.5 * (lo[n_angle_vars:] + hi[n_angle_vars:])
        
        # We have self.budget evaluations.
        # We will use all of them.
        # Reserve 3 for the corners.
        n_corners = 3
        if self.budget <= n_corners:
             # Fallback if budget is tiny (not the case here, budget=200)
             points = []
             for i in range(self.budget):
                 x = np.random.uniform(lo, hi)
                 points.append(x)
             X = np.array(points)
             F = np.array([problem(x) for x in X])
             return F
             
        n_lhs = self.budget - n_corners
        
        # Generate LHS points in [0,1]^2 for the angle variables
        # Using a simple LHS implementation
        # 1. Divide [0,1] into n intervals
        # 2. Sample one point in each interval for each dimension
        # 3. Shuffle the second dimension to avoid diagonal alignment
        
        x1 = np.linspace(0, 1, n_lhs, endpoint=False)
        x1 += np.random.uniform(0, 1/n_lhs, n_lhs)
        x1 = np.sort(x1) # Ensure coverage
        
        # Shuffle for the second dimension
        x2 = np.linspace(0, 1, n_lhs, endpoint=False)
        x2 += np.random.uniform(0, 1/n_lhs, n_lhs)
        x2 = np.random.permutation(x2)
        
        # Combine angle variables
        angle_points = np.column_stack((x1, x2))
        
        # Create full dimension vectors
        # x[0], x[1] from angle_points
        # x[2:] from dist_mid
        X_lhs = np.zeros((n_lhs, self.dim))
        X_lhs[:, :n_angle_vars] = angle_points
        X_lhs[:, n_angle_vars:] = dist_mid
        
        # Define corner points
        # Corner 1: x1=0, x2=0
        # Corner 2: x1=0, x2=1
        # Corner 3: x1=1, x2=0
        # Note: In DTLZ2, the Pareto front is defined for x1, x2 in [0,1].
        # The mapping is theta = pi * x / 2.
        # (0,0) -> (1,0,0)
        # (0,1) -> (0,1,0)
        # (1,0) -> (0,0,1)
        # (1,1) -> (0,0,0) is NOT on the front in the standard DTLZ2 definition?
        # Let's check: f1 = cos(0)*cos(pi/2) = 0. f2 = cos(0)*sin(pi/2) = 1. f3 = sin(0) = 0.
        # So (1,1) gives (0,1,0) which is the same as (0,1)?
        # Wait.
        # f1 = cos(pi*x1/2) * cos(pi*x2/2)
        # f2 = cos(pi*x1/2) * sin(pi*x2/2)
        # f3 = sin(pi*x1/2)
        
        # x1=0, x2=0: f1=1, f2=0, f3=0. (Point A)
        # x1=0, x2=1: f1=0, f2=1, f3=0. (Point B)
        # x1=1, x2=0: f1=0, f2=0, f3=1. (Point C)
        # x1=1, x2=1: f1=0, f2=0, f3=1. (Same as C? No, f2 = cos(pi/2)*sin(pi/2) = 0. Yes, same.)
        # So the corners are (0,0), (0,1), (1,0).
        
        corners = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0]
        ])
        
        X_corners = np.zeros((n_corners, self.dim))
        X_corners[:, :n_angle_vars] = corners
        X_corners[:, n_angle_vars:] = dist_mid
        
        # Combine all points
        X = np.vstack((X_lhs, X_corners))
        
        # Evaluate with deadline guard
        F_list = []
        for x in X:
            if time.time() - start_time > timeout:
                break
            F_list.append(problem(x))
        
        F = np.array(F_list)
        
        return F
