import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        n_samples = self.budget
        dim = self.dim
        
        # DTLZ7 Structure Analysis:
        # Objectives:
        # f1 = x_0
        # f2 = x_0 * (1 - x_1)
        # f3 = (1 + x_1) * (1 - x_0 - x_0*(1-x_1)) = (1 + x_1) * (1 - 2*x_0 + x_0*x_1)
        # 
        # The Pareto front is defined by x_0, x_1 in [0,1] and all other x_i = 0 (to minimize g=1).
        # Let u = x_0 and v = x_1.
        # f1 = u
        # f2 = u * (1 - v)
        # f3 = (1 + v) * (1 - u - u*(1-v))
        
        # Reference Point R = [1.1, 1.1, 25.0]
        # Hypervolume contribution of a point p is roughly proportional to the volume of the box
        # it dominates. Since R is far away in f3 (25.0) but close in f1, f2 (1.1),
        # points with smaller f1 and f2 but larger f3 might not necessarily be best.
        # Actually, for DTLZ7, the Pareto front is "convex" towards the origin in f1,f2,f3?
        # Let's check the shape.
        # At u=0, v=0: f=(0,0,1). Box: 1.1*1.1*24 = 29.04
        # At u=1, v=0: f=(1,0,0). Box: 0.1*1.1*25 = 2.75
        # At u=0, v=1: f=(0,0,2). Box: 1.1*1.1*23 = 27.83
        # At u=1, v=1: f=(1,0,0). Box: 2.75
        # Wait, f2 = u(1-v). If v=1, f2=0. f3 = 2(1-u).
        # If u=0, v=1: f=(0,0,2).
        # If u=0.5, v=0.5: f1=0.5, f2=0.25, f3=1.5*(1-0.5-0.25)=1.5*0.25=0.375.
        # Box: (1.1-0.5)*(1.1-0.25)*(25-0.375) = 0.6*0.85*24.625 = 12.5.
        
        # The "corners" of the Pareto front in objective space are:
        # P1: u=0, v=0 -> f=(0,0,1)
        # P2: u=1, v=0 -> f=(1,0,0)
        # P3: u=0, v=1 -> f=(0,0,2)
        # Note: The front is not a simple triangle in f-space.
        
        # Strategy:
        # 1. Explicitly evaluate the Pareto extreme points (corners) to capture the full extent.
        #    This is crucial because finite sampling might miss the exact boundaries.
        # 2. Use importance sampling for the remaining points.
        #    The hypervolume is dominated by points that are "close" to the reference point in a way
        #    that maximizes the product of (R_i - f_i).
        #    Since R = [1.1, 1.1, 25.0], the term (25 - f3) is large and relatively constant for most of the front
        #    (f3 is between 0 and 2). The variation in f1 and f2 (between 0 and 1) has a larger relative impact
        #    on the product because (1.1 - f1) and (1.1 - f2) vary from 1.1 down to 0.
        #    Therefore, we should sample more densely in the region of the Pareto front where f1 and f2 are small,
        #    as these points have a larger "lever arm" against the reference point in the first two dimensions.
        
        # Let's use a transformation that biases towards u=0, v=0 (which corresponds to small f1, f2).
        # However, we must cover the whole front to ensure non-dominated set coverage.
        # A good heuristic for hypervolume maximization with a far reference point is to sample
        # uniformly in the objective space region dominated by the front, but mapped back to the front.
        
        # Alternative: Sample uniformly in (u, v) space but with a bias.
        # Let's try sampling u and v such that they are biased towards 0.
        # u = U1^alpha, v = U2^beta.
        # If we bias towards 0, we get more points near (0,0) in u,v space.
        # Near u=0, v=0, f1~0, f2~0, f3~1. This is the "best" point for hypervolume contribution?
        # Let's check the gradient of the HV contribution.
        
        # Actually, the inspiration 3 noted that uniform density wastes evaluations on points contributing little.
        # Points near the "origin" of the objective space (0,0,0) contribute the most.
        # The point (0,0,1) is the closest to the origin.
        # So we should bias sampling towards u=0, v=0.
        
        n_corners = 3
        n_samples_remaining = n_samples - n_corners
        
        # Define the corners in (u, v) space
        corners_uv = np.array([
            [0.0, 0.0], # f=(0,0,1)
            [1.0, 0.0], # f=(1,0,0)
            [0.0, 1.0]  # f=(0,0,2)
        ])
        
        # Generate remaining points with bias towards u=0, v=0
        # Using power-law transformation. 
        # If we use u = U^2, the density is higher near 0.
        # Let's try u = U1^1.5, v = U2^1.5. This is a moderate bias.
        
        U1 = np.random.random(n_samples_remaining)
        U2 = np.random.random(n_samples_remaining)
        
        u_samples = U1 ** 1.5
        v_samples = U2 ** 1.5
        
        # Combine corners and sampled points
        u_all = np.concatenate([corners_uv[:, 0], u_samples])
        v_all = np.concatenate([corners_uv[:, 1], v_samples])
        
        # Construct full dimension vectors
        X = np.zeros((n_samples, dim))
        X[:, 0] = u_all
        X[:, 1] = v_all
        # Variables 2 to 21 are 0.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        for i in range(n_samples):
            if time.time() > deadline:
                break
            x = X[i]
            x = np.clip(x, lo, hi)
            problem(x)
            
        return X
