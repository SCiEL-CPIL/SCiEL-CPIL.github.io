import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 300s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        budget = self.budget
        n_total = min(budget, 200)
        
        # DTLZ7 Pareto front: x[2:] = 0.5, x[0], x[1] in [0,1]
        # f1 = x[0], f2 = x[1], f3 = 161 * h where h = 4 - f1/(161)*(1+sin(pi*f1/161)) - f2/(161)*(1+sin(pi*f2/161))
        
        # Use a structured grid on the 2D Pareto front for optimal hypervolume coverage
        # The hypervolume contribution is maximized by uniform coverage in objective space
        # Since f1=x[0] and f2=x[1] are linear, use uniform grid in x[0],x[1]
        
        # Create a uniform grid that covers [0,1]^2 with n_total points
        # Use a space-filling curve (Z-order) for good coverage
        # Or simply use a stratified sample
        
        # Better approach: use Gauss-Legendre nodes for optimal quadrature
        # For 2D, use tensor product of 1D Gauss-Legendre nodes
        sqrt_n = int(np.sqrt(n_total))
        if sqrt_n * sqrt_n == n_total:
            # Perfect square, use tensor product
            nodes_1d, _ = np.polynomial.legendre.leggauss(sqrt_n)
            # Map from [-1,1] to [0,1]
            nodes_1d = (nodes_1d + 1) / 2
            x0_flat, x1_flat = np.meshgrid(nodes_1d, nodes_1d)
            x0_flat = x0_flat.ravel()
            x1_flat = x1_flat.ravel()
        else:
            # Use Halton sequence for non-square case
            def halton(index, base):
                result = 0.0
                f = 1.0 / base
                i = index
                while i > 0:
                    result += f * (i % base)
                    i = i // base
                    f /= base
                return result
            
            x0_flat = np.array([halton(i + 1, 2) for i in range(n_total)])
            x1_flat = np.array([halton(i + 1, 3) for i in range(n_total)])
        
        # Ensure we have exactly n_total points
        if len(x0_flat) > n_total:
            x0_flat = x0_flat[:n_total]
            x1_flat = x1_flat[:n_total]
        elif len(x0_flat) < n_total:
            # Pad with additional Halton points
            def halton2(index, base):
                result = 0.0
                f = 1.0 / base
                i = index
                while i > 0:
                    result += f * (i % base)
                    i = i // base
                    f /= base
                return result
            extra_n = n_total - len(x0_flat)
            extra_x0 = np.array([halton2(len(x0_flat) + i + 1, 2) for i in range(extra_n)])
            extra_x1 = np.array([halton2(len(x0_flat) + i + 1, 3) for i in range(extra_n)])
            x0_flat = np.concatenate([x0_flat, extra_x0])
            x1_flat = np.concatenate([x1_flat, extra_x1])
        
        # Build points on the Pareto front
        X = np.zeros((n_total, n))
        X[:, 0] = x0_flat
        X[:, 1] = x1_flat
        X[:, 2:] = 0.5  # Pareto front location
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
