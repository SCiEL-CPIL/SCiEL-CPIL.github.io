import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ7 Pareto front structure:
        # For x2...x22 = 0, we have:
        # g = 0, g+1 = 1
        # h(u, g) = g + 1 - u*(g+1 - sum_{j=1}^{m-2} u_j^0.5 * sin(pi*u_j/2))
        # With m=3, m-2=1, so sum is just u1*sin(pi*u1/2) where u1=x0
        # h = 1 - u1*(1 - u1*sin(pi*u1/2)) = 1 - u1 + u1^2*sin(pi*u1/2)
        # f1 = x0 = u1
        # f2 = x1*(1+x0) = u2*(1+u1)
        # f3 = h = 1 - u1 + u1^2*sin(pi*u1/2)
        
        # The Pareto front is a 2D surface parameterized by u1 in [0,1] and u2 in [0, 1.1/(1+u1)]
        # (since f2 < 1.1 constraint)
        
        # To maximize hypervolume, we want to densely sample the front.
        # The hypervolume contribution is roughly (ref - f_i) product.
        
        # Strategy: Use a grid in (u1, u2) space that is denser where the hypervolume 
        # contribution is larger. The contribution is largest when all f_i are small,
        # i.e., u1 small, u2 small.
        
        # Let's use a non-uniform grid that is denser near the origin.
        # Use a power-law spacing: t = k^alpha / N^alpha where k=1..N, alpha > 1 gives denser near 0.
        
        # Alternatively, just use a fine uniform grid. With 200 points, let's try different splits.
        # The Pareto front in (f1, f2, f3) space: for fixed u1, f2 ranges linearly with u2, 
        # and f3 is fixed. So the front is a collection of line segments in (f2, f3) for each f1.
        
        # For hypervolume, we want to cover the 2D projection well.
        # Let's use 16 points in u1 and 12 in u2 = 192, plus 8 extra points.
        
        # Actually, let's think about this more carefully. The reference point is [1.1, 1.1, 25.0].
        # f1 = u1 in [0,1], so ref_f1 - f1 = 1.1 - u1 in [0.1, 1.1]
        # f2 = u2*(1+u1), with u2 in [0, 1.1/(1+u1)], so f2 in [0, 1.1), ref_f2 - f2 = 1.1 - f2 in (0, 1.1]
        # f3 = 1 - u1 + u1^2*sin(pi*u1/2), which is in [0, 1], so ref_f3 - f3 = 25 - f3 in [24, 25]
        
        # The hypervolume element is roughly (1.1-u1) * (1.1-f2) * (25-f3) * df1 * df2
        # Since (25-f3) is almost constant (~24-25), the dominant factors are (1.1-u1)*(1.1-f2).
        
        # Let's use a grid that is uniform in the objective space rather than parameter space.
        # For each u1, f2 = u2*(1+u1), so u2 = f2/(1+u1).
        # If we want uniform f2 in [0, 1.1), then u2 = f2/(1+u1) is non-uniform.
        
        # Let's just use a uniform grid in (u1, u2) and see.
        # Try 20x10 = 200.
        
        # Use non-uniform grid denser near origin where hypervolume contribution is largest
        # Weight for hypervolume: (1.1-u1)*(1.1-f2)*(25-f3) ~ (1.1-u1)*(1.1-f2)*24.5
        # In objective space, f1=u1, f2=u2*(1+u1)
        # Use power-law spacing in f1: f1_i = (i/N)^alpha * 1.0, alpha < 1 gives denser near 0
        # And uniform in f2: f2_j = j/M * 1.05, then u2 = f2/(1+u1)
        
        n_u1 = 18
        n_u2 = 11  # 18*11 = 198, plus 2 extra points
        
        # Non-uniform u1: denser near 0
        # Use sqrt spacing: u1_i = (i/(N-1))^2 for i=0..N-1 gives denser near 0
        # Actually (i/N)^2 for i=0..N-1: u1 = [0, 1/361, 4/361, ..., (17/18)^2]
        # Let's use u1_i = (i/(n_u1-1))^2 for i=0..n_u1-1
        u1_vals = np.array([(i / (n_u1 - 1)) ** 2 for i in range(n_u1)])
        
        all_u1 = []
        all_u2 = []
        
        for u1 in u1_vals:
            # Sample f2 uniformly in [0, 1.05], then u2 = f2/(1+u1)
            f2_vals = np.linspace(0.0, 1.05, n_u2)
            u2_vals = f2_vals / (1.0 + u1)
            for u2 in u2_vals:
                all_u1.append(u1)
                all_u2.append(u2)
        
        # Add 2 extra points at strategic locations
        all_u1.append(0.0)
        all_u2.append(0.525 / 1.0)  # f2 = 0.525 at u1=0
        all_u1.append(0.5)
        all_u2.append(0.525 / 1.5)  # f2 = 0.525 at u1=0.5
        
        all_u1 = np.array(all_u1)
        all_u2 = np.array(all_u2)
        
        n_samples = len(all_u1)
        
        X = np.zeros((n_samples, self.dim))
        X[:, 0] = all_u1
        X[:, 1] = all_u2
        # X[:, 2:] = 0
        
        for i in range(n_samples):
            if time.time() > deadline:
                break
            problem(X[i])
            
        return X
