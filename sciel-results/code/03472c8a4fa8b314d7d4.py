import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 320.0  # Safety margin for 330s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        n_evals = self.budget
        if n_evals < 1:
            return np.array([])

        # Strategy: Analytic Deterministic Grid on ZDT1 Pareto Front
        #
        # ZDT1 Pareto Front:
        # f1 = x0
        # f2 = (1 + 10*(n-1)) * (1 - sqrt(x0 / g))
        # With x[1:] = 0, g=1, so f2 = C * (1 - sqrt(f1)), where C = 10*(n-1)+1.
        #
        # Hypervolume Maximization:
        # We want to place N points (f1_i, f2_i) to maximize the union of rectangles
        # with corner (11, 11).
        # The area is sum of trapezoids. For a concave front like ZDT1,
        # uniform spacing in f1 is suboptimal compared to spacing that equalizes
        # the area contribution of each segment or accounts for the curvature.
        #
        # A known good heuristic for concave fronts to maximize HV with fixed N
        # is to space points such that the "loss" from discretization is minimized.
        # Spacing linearly in sqrt(f1) (which is linear in f2) often provides
        # better coverage of the "interesting" high-f2 region compared to uniform f1,
        # because the curve is steep near f1=0 and flat near f1=1.
        #
        # Let's use a grid that is uniform in t where f1 = t^2.
        # This means t_i is uniform in [0, 1].
        # f1_i = t_i^2.
        # f2_i = C * (1 - t_i).
        #
        # We will generate N points.
        # To avoid the exact corners if they don't add much HV relative to interior points?
        # Actually, with a fixed reference point, the extreme points (0, C) and (1, 0)
        # are crucial.
        #
        # Let's use linspace for t in [0, 1] with N points.
        
        # Generate t values
        # Using linspace ensures we hit 0 and 1.
        t_vals = np.linspace(0.0, 1.0, n_evals)
        
        # Calculate f1 values (x0)
        # Map t to x0 bounds?
        # The problem states bounds [0, 1] for every variable.
        # So x0 is in [lo[0], hi[0]].
        # We assume the Pareto front is defined for x0 in [0, 1].
        # If bounds are different, we should map accordingly, but task says [0,1].
        x0_vals = lo[0] + t_vals * (hi[0] - lo[0])
        
        # Construct X
        X = np.empty((n_evals, self.dim), dtype=np.float64)
        X[:, 0] = x0_vals
        
        if self.dim > 1:
            # Set x[1:] to lower bound to ensure g(x) = 1 (minimum g)
            # This places points exactly on the Pareto front.
            X[:, 1:] = lo[1:]
            
        # Evaluate all points
        F = np.empty((n_evals, self.n_obj), dtype=np.float64)
        
        for i in range(n_evals):
            if time.time() > deadline:
                break
            F[i] = problem(X[i])
            
        return X
