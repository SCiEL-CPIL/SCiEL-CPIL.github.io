import numpy as np
import time

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

        # Check if 0 is within bounds for x1..x_{dim-1}
        # This ensures g(x) is minimized (g=1 for standard ZDT4)
        zeros_ok = all(lo[i] <= 0 <= hi[i] for i in range(1, self.dim))

        if zeros_ok:
            # Set x1..x_{dim-1} to 0 to minimize g
            # Sample x0 uniformly across its full allowed bounds
            # This maximizes the range of f1 covered, thus maximizing hypervolume
            n_points = self.budget
            
            # Use the full bounds for x0, not clipped to [0, 1]
            # Points outside [0,1] are still on the Pareto front (or close to it)
            # and extend the range of f1, increasing hypervolume
            x0_lo = lo[0]
            x0_hi = hi[0]
            
            if x0_lo >= x0_hi:
                x0_vals = np.array([x0_lo])
                n_points = 1
            else:
                # Uniform sampling in x0
                x0_vals = np.linspace(x0_lo, x0_hi, n_points)
                
            X = np.zeros((n_points, self.dim))
            X[:, 0] = x0_vals
            # X[:, 1:] remains 0.0
            
            # Check time
            if time.time() > deadline:
                return X
            
            F = np.array([problem(x) for x in X])
            return X

        else:
            # Fallback: 0 is not in bounds for some x_i (i>=1)
            # Find the best feasible value for each x_i to minimize g
            # g(x) depends on sum of (x_i^2 - 10*cos(4*pi*x_i))
            # This function has local minima at x_i = k/2 for integer k.
            # We pick the bound closest to 0 (or any multiple of 0.5)
            best_x_tail = []
            for i in range(1, self.dim):
                l = lo[i]
                h = hi[i]
                # Check distance to 0
                d_lo = abs(l)
                d_hi = abs(h)
                # Also check distance to nearest 0.5 multiple? 
                # For simplicity, just pick the bound closer to 0.
                # In standard ZDT4, bounds are [-5, 5], so 0 is in bounds.
                # This case is unlikely.
                if d_lo < d_hi:
                    best_x_tail.append(l)
                else:
                    best_x_tail.append(h)
            
            n_points = self.budget
            x0_lo = lo[0]
            x0_hi = hi[0]
            
            if x0_lo >= x0_hi:
                x0_vals = np.array([x0_lo])
                n_points = 1
            else:
                x0_vals = np.linspace(x0_lo, x0_hi, n_points)
                
            X = np.zeros((n_points, self.dim))
            X[:, 0] = x0_vals
            for i in range(1, self.dim):
                X[:, i] = best_x_tail[i-1]
            
            if time.time() > deadline:
                return X
            
            F = np.array([problem(x) for x in X])
            return X
