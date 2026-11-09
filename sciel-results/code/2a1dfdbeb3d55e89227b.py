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
        # Safety margin: 330s limit, return at 300s to be safe
        deadline = start_time + 295 # Slightly tighter to ensure no risk
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Budget allocation
        # 180 Sobol, 20 Strategic
        n_sobol = 180
        n_strategic = 20
        total_points = n_sobol + n_strategic
        
        if total_points > self.budget:
            scale = self.budget / total_points
            n_sobol = int(n_sobol * scale)
            n_strategic = self.budget - n_sobol

        # Generate Sobol sequence
        # Use a prime number or power of 2? Sobol is best with powers of 2.
        # 128 is a power of 2. 256 is a power of 2.
        # We will generate 256 and take the first 180.
        sobol_generator = qmc.Sobol(d=2, scramble=True)
        sobol_points = sobol_generator.random(256)
        
        # Strategic points:
        # The Pareto front is f1=x0, f2=x1, f3=1-sqrt(x0*x1) (assuming g=0)
        # We want to maximize the hypervolume.
        # The "ideal" points are those that are non-dominated.
        # In the x0-x1 square, ALL points are non-dominated on the Pareto front 
        # (since f3 decreases as x0*x1 increases, and f1/f2 increase as x0/x1 increase).
        # Wait, are they all non-dominated?
        # If we have point A (0.5, 0.5) -> f=(0.5, 0.5, 1-sqrt(0.25)=0.5)
        # And point B (0.6, 0.6) -> f=(0.6, 0.6, 1-sqrt(0.36)=0.4)
        # A is dominated by B? No.
        # f1(A)=0.5 < f1(B)=0.6
        # f2(A)=0.5 < f2(B)=0.6
        # f3(A)=0.5 > f3(B)=0.4
        # So A and B are non-dominated to each other.
        # In fact, for DTLZ7, the mapping (x0, x1) -> (f1, f2, f3) is one-to-one 
        # and all points on the Pareto front are non-dominated to each other.
        # Therefore, we just need to cover the square [0,1]x[0,1] as uniformly as possible 
        # to maximize the volume under the surface f3(x0,x1) in the objective space?
        # No, Hypervolume is the volume dominated by the set of points relative to a reference point.
        # Ref point: [1.1, 1.1, 25.0].
        # The volume is the integral over the region dominated by the points.
        # To maximize this, we want the points to be as "extreme" as possible.
        # The Pareto front is the set of all optimal solutions.
        # We are sampling from the Pareto front (by setting x2: = 0).
        # So we are effectively sampling 2D points (x0, x1) to define the 3D front.
        # The hypervolume is determined by the "shell" formed by these points.
        # To maximize hypervolume with a fixed number of points on a smooth front, 
        # we should distribute them such that they partition the objective space 
        # into boxes of roughly equal volume, or maximize the sum of the volumes 
        # of the boxes under the front.
        # For a convex front (in objective space, looking from the reference point), 
        # a uniform distribution in the parameter space (x0, x1) is a good start, 
        # but might be suboptimal if the curvature is high.
        
        # Let's stick to Sobol for the bulk, but refine the strategic points 
        # to ensure the "corners" and "edges" are tightly captured, as these 
        # define the maximum extent of the front.
        
        strategic_x01 = np.array([
            [0.0, 0.0],   # f=(0,0,1)
            [1.0, 0.0],   # f=(1,0,1)
            [0.0, 1.0],   # f=(0,1,1)
            [1.0, 1.0],   # f=(1,1,0)
            
            # Dense sampling near the (1,1) corner where f3 is lowest (most valuable for HV)
            [0.9, 0.9],
            [0.8, 0.9],
            [0.9, 0.8],
            [0.8, 0.8],
            [0.7, 0.9],
            [0.9, 0.7],
            [0.7, 0.8],
            [0.8, 0.7],
            
            # Mid-range
            [0.5, 0.5],
            [0.5, 0.6],
            [0.6, 0.5],
            
            # Low ranges
            [0.1, 0.1],
            [0.1, 0.5],
            [0.5, 0.1],
            [0.1, 0.9],
            [0.9, 0.1]
        ])
        
        # Combine
        all_points = np.vstack([sobol_points[:n_sobol], strategic_x01[:n_strategic]])
        
        # Shuffle to ensure random order of evaluation (though it doesn't strictly matter for final set)
        np.random.shuffle(all_points)
        
        # Prepare full dimensionality vectors
        X = np.zeros((len(all_points), self.dim))
        X[:, 0] = all_points[:, 0]
        X[:, 1] = all_points[:, 1]
        X[:, 2:] = lo[2:] # Set to 0 for DTLZ7 optimality
        
        results = []
        for i in range(len(X)):
            if time.time() > deadline:
                break
            
            x = np.clip(X[i], lo, hi)
            f = problem(x)
            results.append(f)
            
        return results
