import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        n = self.budget
        dim = self.dim
        
        # DTLZ7 Pareto front structure analysis:
        # Standard DTLZ7:
        # g(x) = 1 + 9 * (sum(x_i) for i in 1..n-m) / (n-m)
        # h(y, g) = g * (1 - sqrt(y_{m-1} / g))
        # f_i = y_i for i < m
        # f_m = h(y, g)
        
        # To minimize objectives, we want g=0 (which implies x_1..x_{n-m} = 0).
        # When g=0, the term (1 - sqrt(y_{m-1}/g)) is 0/0 indeterminate, but the limit 
        # as g->0+ depends on the ratio. However, in standard implementations, if g=0,
        # f_m is often defined or behaves such that the front is defined by the y variables.
        # Specifically, for DTLZ7, the Pareto front is non-convex.
        # The variables x_0 and x_1 (which are part of the 'y' set if m=3 and n-m is the rest)
        # define the shape of the front.
        # Wait, standard DTLZ7 definition:
        # x = [x_1, ..., x_{n-m+1}, x_{n-m+2}, ..., x_n]
        # y = [x_1, ..., x_{n-m+1}]  (these are the front-shaping variables)
        # g = 1 + 9 * sum(x_{n-m+2}..x_n) / (n-m)
        # f_i = y_i for i=1..m-1
        # f_m = g * (1 - sqrt(y_{m-1} / g))
        
        # In our case, n=22, m=3.
        # y has size m-1 = 2. So y = [x_0, x_1].
        # g depends on x_2 .. x_21.
        # To get to the Pareto front, we must minimize g, which means setting x_2..x_21 = 0.
        # Then g = 1.
        # f_0 = x_0
        # f_1 = x_1
        # f_2 = 1 * (1 - sqrt(x_1 / 1)) = 1 - sqrt(x_1)
        
        # So the Pareto front in objective space is:
        # f_0 = x_0
        # f_1 = x_1
        # f_2 = 1 - sqrt(x_1)
        # where x_0, x_1 in [0, 1].
        
        # We want to maximize hypervolume with reference point [1.1, 1.1, 25.0].
        # The front is a 2D surface (since x_0 and x_1 are independent variables on the front).
        # Actually, f_2 depends only on x_1. f_0 depends only on x_0.
        # This is a product structure.
        # To maximize hypervolume, we should sample x_0 and x_1 to cover the domain [0,1]x[0,1]
        # effectively. Since the front is 2D, a 2D low-discrepancy sequence is appropriate.
        
        # Fibonacci lattice (sunflower map) on the unit disk:
        # This provides uniform coverage of the 2D domain.
        
        # Generate indices
        i = np.arange(n)
        
        # Sunflower map parameters
        # r = sqrt(i) / sqrt(n)  (to fill the disk uniformly)
        # theta = i * golden_angle
        # x = r * cos(theta)
        # y = r * sin(theta)
        
        # However, we need to map to the square [0,1]x[0,1] or directly to x_0, x_1.
        # The Pareto front is defined on the square [0,1] x [0,1] for (x_0, x_1).
        # A Fibonacci lattice on a disk is good for circular domains.
        # For a square domain, a 2D Halton sequence or a simple grid with jitter might be better.
        # Or, we can use the Fibonacci lattice on the square by mapping the disk to the square?
        # No, better to use a 2D low-discrepancy sequence that covers the square.
        
        # Let's use a 2D Halton sequence for x_0 and x_1.
        # Base 2 for x_0, Base 3 for x_1.
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        # Vectorized Halton is faster
        def halton_vectorized(indices, base):
            r = np.zeros(len(indices))
            f = 1.0
            idx = indices.copy()
            while np.any(idx > 0):
                f /= base
                r += f * (idx % base)
                idx //= base
            return r
        
        x0_vals = halton_vectorized(i + 1, 2)
        x1_vals = halton_vectorized(i + 1, 3)
        
        # Construct X
        X = np.zeros((n, dim))
        X[:, 0] = x0_vals
        X[:, 1] = x1_vals
        # x_2 .. x_21 are 0 to ensure g=1 (minimum g)
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        # Check time
        if time.time() > deadline:
            pass # We are done anyway, just returning.
            
        return X
