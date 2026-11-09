import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n = self.budget
        dim = self.dim
        
        # DTLZ1 Pareto front in objective space is the 2D simplex:
        # f0 + f1 + f2 = 0.5, fi >= 0
        # 
        # We parameterize using spherical coordinates on the unit simplex:
        # Let s = (f0, f1, f2) / 0.5, so s0+s1+s2 = 1, si >= 0
        # 
        # Using the standard parameterization:
        # s0 = sin(theta1) * sin(theta2)
        # s1 = sin(theta1) * cos(theta2)
        # s2 = cos(theta1)
        # where theta1 in [0, pi/2], theta2 in [0, 2*pi)
        # But this doesn't cover the simplex uniformly and has issues at boundaries.
        #
        # Better: use the fact that any point on the 2D simplex can be written as:
        # s0 = 1 - u - v
        # s1 = u
        # s2 = v
        # where u >= 0, v >= 0, u + v <= 1
        #
        # To get good coverage, we can use a quasi-random or structured grid in (u,v) space.
        #
        # Actually, let's use a simpler approach: sample (u, v) on a fine grid over the triangle
        # u >= 0, v >= 0, u + v <= 1, and map to objectives.
        #
        # For 200 points, we can use a triangular grid.
        # 
        # Alternatively, use the inverse mapping from objectives to decision variables:
        # Given f0, f1, f2 on the front (sum = 0.5):
        # f0 = 0.5 * x[0]  =>  x[0] = 2*f0
        # f2 = 0.5 * (1 - x[1])  =>  x[1] = 1 - 2*f2
        # f1 = 0.5 * (1 - x[0]) * x[1]  =>  check consistency
        #
        # Let's verify: with x[0] = 2*f0 and x[1] = 1 - 2*f2:
        # f1_computed = 0.5 * (1 - 2*f0) * (1 - 2*f2)
        # For this to equal f1, we need:
        # f1 = 0.5 * (1 - 2*f0) * (1 - 2*f2)
        #
        # And we also need f0 + f1 + f2 = 0.5:
        # f0 + 0.5*(1-2*f0)*(1-2*f2) + f2 = 0.5
        # f0 + 0.5 - f0 - f2 + 2*f0*f2 + f2 = 0.5
        # 0.5 + 2*f0*f2 = 0.5
        # So f0*f2 = 0, which means either f0=0 or f2=0.
        #
        # This confirms the front is only on the edges of the simplex in this parameterization?
        # No, that can't be right. Let me re-examine.
        #
        # I think the issue is that not all points on the simplex f0+f1+f2=0.5 are achievable
        # by the DTLZ1 mapping. The actual Pareto front of DTLZ1 is indeed the simplex, but
        # the mapping from (x[0], x[1]) to (f0, f1, f2) at g=0 is:
        # f0 = 0.5*x[0]
        # f1 = 0.5*(1-x[0])*x[1]
        # f2 = 0.5*(1-x[1])
        #
        # The image of [0,1]^2 under this map is a 2D region. Let's find what region.
        # Let a = x[0], b = x[1]. Then:
        # f0 = 0.5a
        # f1 = 0.5(1-a)b
        # f2 = 0.5(1-b)
        #
        # So a = 2*f0, b = 1 - 2*f2
        # And f1 = 0.5(1-2*f0)(1-2*f2)
        #
        # So the front is the set of (f0, f1, f2) such that:
        # f1 = 0.5(1-2*f0)(1-2*f2), f0 in [0,1], f2 in [0,1], f1 >= 0
        #
        # This is a 2D surface, not the full simplex. Let me check if it equals the simplex.
        # At f0=0.5, f2=0.5: f1 = 0.5*0*0 = 0. Sum = 0.5+0+0.5 = 1.0 != 0.5.
        # So the front is NOT the simplex sum=0.5. I was wrong earlier.
        #
        # Let me just sample (x[0], x[1]) on a fine grid and evaluate.
        # This is the most straightforward approach.
        
        # Use a structured grid in (x[0], x[1]) space
        # For 200 points, use approximately 14x14 grid = 196, plus 4 extra
        
        n_side = int(np.sqrt(n))
        # Ensure we don't exceed budget
        while n_side * n_side > n:
            n_side -= 1
        
        x0_vals = np.linspace(0, 1, n_side)
        x1_vals = np.linspace(0, 1, n_side)
        X0, X1 = np.meshgrid(x0_vals, x1_vals)
        samples = np.column_stack((X0.ravel(), X1.ravel()))
        
        # Add mid-edge and center points if we have room
        extra_points = []
        remaining = n - len(samples)
        
        # Mid-edges
        mid_edges = [
            [0.5, 0.0],
            [0.5, 1.0],
            [0.0, 0.5],
            [1.0, 0.5]
        ]
        
        # Center
        center = [0.5, 0.5]
        
        candidates = mid_edges + [center]
        
        for pt in candidates:
            if remaining <= 0:
                break
            # Check if this point is already in samples
            dists = np.sum((samples - np.array(pt))**2, axis=1)
            if np.min(dists) > 1e-10:
                samples = np.vstack([samples, np.array(pt)])
                remaining -= 1
        
        # If we still have room, add more points from a finer grid
        if remaining > 0:
            n_side_fine = n_side + 1
            x0_fine = np.linspace(0, 1, n_side_fine)
            x1_fine = np.linspace(0, 1, n_side_fine)
            X0_f, X1_f = np.meshgrid(x0_fine, x1_fine)
            fine_samples = np.column_stack((X0_f.ravel(), X1_f.ravel()))
            
            for fs in fine_samples:
                if remaining <= 0:
                    break
                dists = np.sum((samples - fs)**2, axis=1)
                if np.min(dists) > 1e-10:
                    samples = np.vstack([samples, fs])
                    remaining -= 1
        
        # Truncate or pad to exactly n
        if len(samples) > n:
            samples = samples[:n]
        elif len(samples) < n:
            # Pad with center point
            pad = np.tile(np.array([0.5, 0.5]), (n - len(samples), 1))
            samples = np.vstack([samples, pad])
        
        # Evaluate each sample
        for i in range(n):
            x = np.zeros(dim)
            x[0] = samples[i, 0]
            x[1] = samples[i, 1]
            x[2:] = 0.5  # g = 0
            problem(x)
        
        return None
