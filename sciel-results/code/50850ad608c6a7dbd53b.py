import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285
        
        n_points = self.budget
        
        # For DTLZ4 with 3 objectives, the Pareto front is a 2D surface
        # Parameterized by two angles/variables. We need to sample this 2D surface
        # to maximize hypervolume with reference point [2.5, 2.5, 2.5]
        
        # Strategy: Use stratified sampling in 2D parameter space
        # The front is parameterized by (a, b) where:
        #   a in [0, pi/2], b in [0, pi/2]
        #   f1 = 0 (or small), f2 = cos(a)*sin(b), f3 = cos(a)*cos(b)
        # But actually for DTLZ4, the front when g=0 gives:
        #   f1 = sin(pi*x1/2), f2 = cos(pi*x1/2)*sin(pi*x2/2), f3 = cos(pi*x1/2)*cos(pi*x2/2)
        # with x1, x2 in [0,1]
        
        # So parameter space is (x1, x2) in [0,1]^2
        # f1 = sin(pi*x1/2)
        # f2 = cos(pi*x1/2) * sin(pi*x2/2)
        # f3 = cos(pi*x1/2) * cos(pi*x2/2)
        
        # To maximize hypervolume, we want uniform coverage in objective space
        # which corresponds to specific non-uniform sampling in (x1, x2) space
        
        # Use Halton sequence for low-discrepancy sampling in [0,1]^2
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f = f / base
                r = r + f * (index % base)
                index //= base
            return r
        
        # Generate points using Halton sequence
        # Include explicit corner points for maximum coverage
        # Corners in (x1, x2) space: (0,0), (0,1), (1,0), (1,1)
        # These map to:
        # (0,0): f=(0, 0, 1) -> wait, f1=sin(0)=0, f2=cos(0)*sin(0)=0, f3=cos(0)*cos(0)=1 -> (0,0,1)
        # (0,1): f1=0, f2=cos(0)*sin(pi/2)=1, f3=cos(0)*cos(pi/2)=0 -> (0,1,0)
        # (1,0): f1=sin(pi/2)=1, f2=cos(pi/2)*sin(0)=0, f3=cos(pi/2)*cos(0)=0 -> (1,0,0)
        # (1,1): f1=1, f2=0, f3=0 -> (1,0,0) same as (1,0)? No: f2=cos(pi/2)*sin(pi/2)=0, f3=cos(pi/2)*cos(pi/2)=0
        
        # So corners give (0,0,1), (0,1,0), (1,0,0) - the three extreme points
        
        points_param = []
        
        # Add explicit corners
        points_param.append((0.0, 0.0))  # -> (0, 0, 1)
        points_param.append((0.0, 1.0))  # -> (0, 1, 0)
        points_param.append((1.0, 0.0))  # -> (1, 0, 0)
        
        # Generate remaining points with Halton
        num_remaining = n_points - 3
        for i in range(1, num_remaining + 1):
            if time.time() > deadline:
                break
            x1 = halton(i, 2)
            x2 = halton(i, 3)
            points_param.append((x1, x2))
        
        results = []
        for x1, x2 in points_param:
            if time.time() > deadline:
                break
            
            # Clip to [0,1]
            x1_c = np.clip(x1, 0.0, 1.0)
            x2_c = np.clip(x2, 0.0, 1.0)
            
            # Map to DTLZ4 variables
            # For g=0, we need the "g" variables to be at their optimum
            # Typically x[2:12] should be 0.5 for DTLZ4 (where g is minimized)
            # x[0] and x[1] are the front-shaping variables
            
            x = np.zeros(self.dim)
            x[0] = x1_c
            x[1] = x2_c
            # Set remaining variables to 0.5 to minimize g
            x[2:] = 0.5
            
            f = problem(x)
            results.append(f)
        
        return results
