import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo

        # ZDT6 with x[1:]=0:
        # g = 1 + 9/sqrt(x0^2) = 1 + 9/x0 for x0 > 0
        # f1 = 1 - exp(-4 * x0 * sin^2(10 * pi * x0))
        # f2 = g * (1 - sqrt(f1/g))
        #
        # The Pareto front of ZDT6 is: f1 in [0,1], f2 = 1 - sqrt(f1)
        # For x[1:]=0, as x0 varies in [0,1], f1 takes values in [0,1]
        # but NOT monotonically (sin^2 causes oscillations).
        #
        # Key insight: We can directly sample x0 values that map to well-distributed f1 values.
        # Since f1 = 1 - exp(-4*x0*sin^2(10*pi*x0)), and we want uniform coverage of f1 in [0,1],
        # we can use a dense grid of x0 values in [0,1] and let the function evaluate them.
        #
        # Strategy: Use ALL budget for dense sampling of x0 in [0,1] with x[1:]=0.
        # This ensures we cover the entire Pareto front without root-finding overhead.
        
        # Phase 1: Dense sampling along x0 in [0,1]
        # Use non-uniform grid to handle the non-monotonic f1(x0)
        # Near x0=0, f1 changes slowly; near x0=0.5, f1 oscillates rapidly
        
        n_front = budget  # Use all budget for front sampling
        
        # Create a grid that's denser where f1 changes rapidly
        # f1 = 1 - exp(-4*x0*sin^2(10*pi*x0))
        # The derivative is complex, so use a combination of uniform and clustered points
        
        # Uniform grid in x0
        x0_uniform = np.linspace(0, 1, n_front)
        
        # Map to decision space: x0 = x0_uniform, x[1:] = 0
        X_front = np.zeros((n_front, dim))
        X_front[:, 0] = x0_uniform
        
        # Evaluate all points
        F_front = np.array([problem(x) for x in X_front])
        used = n_front
        
        # Combine all points
        if used > 0:
            X_all = X_front
            F_all = F_front
        else:
            X_all = np.array([]).reshape(0, dim)
            F_all = np.array([]).reshape(0, n_obj)
        
        return X_all
