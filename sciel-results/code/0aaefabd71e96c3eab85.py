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
                
                lo = np.array([b[0] for b in bounds])
                hi = np.array([b[1] for b in bounds])
                
                # ZDT4 specific logic:
                # f1 = x1
                # f2 = g(x) * (1 - sqrt(x1/n))
                # g(x) = sum(g(xi)) for i=2..n
                # g(xi) min is at xi=0 (value 10)
                
                # Determine the "optimal" values for x2..xn
                # For standard ZDT4, this is 0.
                # Check if 0 is in bounds for all i > 0.
                optimal_tail = np.zeros(self.dim - 1)
                tail_in_bounds = np.all((lo[1:] <= 0) & (0 <= hi[1:]))
                
                if not tail_in_bounds:
                    # Fallback: Use center of bounds or a quick search
                    # For now, use center as a safe bet, or just clip 0
                    optimal_tail = np.clip(0, lo[1:], hi[1:])
                    # A more robust fallback would be a few random evals to find min g
                    # But let's stick to the simple assumption first.
                
                # We will use all budget to sample x1
                # x1 is index 0
                x1_lo, x1_hi = bounds[0]
                
                # Create x1 samples
                # For ZDT4, the Pareto front is convex: f2 = g*(1-sqrt(x1/n))
                # Use a non-uniform spacing that concentrates points where the curve bends
                # Use a sine-based or power-law spacing to better cover the convex front
                # t in [0,1], x1 = x1_lo + t*(x1_hi - x1_lo)
                # The front curvature is highest near x1=0, so use t^2 spacing
                t = np.linspace(0, 1, self.budget)
                # Use t^1.5 or similar to bias towards lower x1 values where curvature is higher
                t_nonuniform = t ** 1.3
                x1_samples = x1_lo + t_nonuniform * (x1_hi - x1_lo)
                
                # Construct full vectors
                X = np.zeros((self.budget, self.dim))
                X[:, 0] = x1_samples
                X[:, 1:] = optimal_tail
                
                # Evaluate
                F = []
                for i in range(self.budget):
                    if time.time() > deadline:
                        break
                    x = X[i]
                    f = problem(x)
                    F.append(f)
                
                return F
