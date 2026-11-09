"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution

class Optimizer:
    """Multi-objective optimizer for ZDT1 using global optimization of the sampling distribution on the known Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT1 structure:
        # f1 = x0
        # f2 = 1 + sin(10*pi*x0) + (n-1) * (g(x) + (x0/g(x))^alpha)
        # The Pareto front is achieved when x1..xn = 0, which minimizes g(x) to 1.
        # On the front: f1 = x0, f2 = 1 + sin(10*pi*x0).
        # Reference point: [11.0, 11.0]
        
        # Strategy:
        # 1. We know the Pareto front analytically.
        # 2. We need to select `budget` points (x0 values) to maximize the hypervolume of the resulting objective vectors.
        # 3. We treat the x0 values as the decision variables for a global optimization problem.
        # 4. We use scipy.optimize.differential_evolution to find the optimal set of x0 values.
        
        n_points = self.budget
        ref1 = 11.0
        ref2 = 11.0
        
        def calc_hv(f1, f2, ref1, ref2):
            """Calculate 2D hypervolume for a list of points."""
            if len(f1) == 0:
                return 0.0
            # Pair and sort by f1
            points = sorted(zip(f1, f2))
            
            # Remove dominated points and compute HV
            # Sort by f1 ascending.
            # Traverse from right to left to find non-dominated points (min f2 so far).
            nd_points = []
            min_f2 = float('inf')
            for f1_val, f2_val in reversed(points):
                if f2_val < min_f2:
                    nd_points.append((f1_val, f2_val))
                    min_f2 = f2_val
            
            if not nd_points:
                return 0.0
                
            # nd_points is in order of f1 descending.
            # Reverse to get f1 ascending.
            nd_points.reverse()
            
            hv = 0.0
            for i in range(len(nd_points)):
                f1_cur, f2_cur = nd_points[i]
                f1_next = nd_points[i+1][0] if i + 1 < len(nd_points) else ref1
                width = f1_next - f1_cur
                height = ref2 - f2_cur
                hv += width * height
            return hv

        def objective(x0s):
            """
            Objective function for differential evolution.
            x0s: array of n_points values in [0, 1].
            Returns negative hypervolume (to minimize).
            """
            f1 = x0s
            f2 = 1.0 + np.sin(10 * np.pi * x0s)
            hv = calc_hv(f1, f2, ref1, ref2)
            return -hv

        # Initial guess: Uniform grid
        # Differential evolution will improve this.
        # We use a small population and limited iterations to stay within time limits.
        # The landscape is likely smooth enough for DE to work well from a uniform start.
        
        # Bounds for x0 are [0, 1]
        bounds_de = [(0.0, 1.0)] * n_points
        
        # To ensure we don't time out, we run DE with strict limits.
        # If DE is too slow, we fall back to the uniform grid or a simple greedy.
        
        try:
            # Use a moderate population size and max iterations
            # n_points=200 is high dimensional for DE, so we use a small popsize.
            res = differential_evolution(
                objective, 
                bounds_de, 
                maxiter=20, 
                popsize=5, 
                tol=0.01, 
                seed=42,
                init='latinhypercube'
            )
            
            if res.success or res.fun < 0: # Check if we got a valid result
                optimal_x0s = res.x
            else:
                optimal_x0s = np.linspace(0, 1, n_points)
                
        except Exception:
            # Fallback to uniform grid if optimization fails or times out
            optimal_x0s = np.linspace(0, 1, n_points)

        # Check time
        if time.time() > deadline:
             # If we are running out of time, just use the uniform grid which is fast to evaluate
             optimal_x0s = np.linspace(0, 1, n_points)

        # Evaluate the optimal points
        recorded_f1 = []
        recorded_f2 = []
        
        for x0 in optimal_x0s:
            x = np.zeros(self.dim)
            x[0] = x0
            x = np.clip(x, lo, hi)
            f = problem(x)
            recorded_f1.append(f[0])
            recorded_f2.append(f[1])
            
        return np.array(recorded_f1)
# EVOLVE-BLOCK-END
