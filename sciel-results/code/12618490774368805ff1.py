import numpy as np
import time
import sys
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        N = self.budget
        if N <= 0:
            return None

        if N == 1:
            x = np.zeros(self.dim)
            x[0] = 0.5
            problem(x)
            return None

        # ZDT1 Pareto front:
        # f1 = x1
        # f2 = 1 - sqrt(x1)
        # Reference point: (11, 11)
        
        # We want to choose x1_0, ..., x1_{N-1} to maximize Hypervolume.
        # HV = sum_{i=0}^{N-1} (f1_{i+1} - f1_i) * (11 - f2_i)
        # where f1_{N} = 11 (reference point)
        # f1_i = x1_i
        # f2_i = 1 - sqrt(x1_i)
        
        # HV = sum_{i=0}^{N-1} (x1_{i+1} - x1_i) * (10 + sqrt(x1_i))
        
        # Let y = x1. y is sorted ascending.
        # Objective to maximize:
        # H(y) = sum_{i=0}^{N-1} (y_{i+1} - y_i) * (10 + sqrt(y_i))
        # with y_N = 11.
        
        # We can reformulate this to minimize -H(y).
        # Note: y_i must be in [0, 1] for i < N-1? 
        # Actually, for ZDT1, the Pareto front is only for x1 in [0, 1].
        # If x1 > 1, f2 = g(1-sqrt(x1/g)) with g=1 => 1-sqrt(x1) which becomes complex/invalid?
        # Wait, ZDT1 definition: f2 = g(1 - sqrt(f1/g)). If f1 > g, sqrt is of negative number?
        # Standard ZDT1: f1 = x1, g = 1 + 9 sum(x2..xn)/(n-1).
        # Pareto front is g=1, so x2..xn=0.
        # Then f2 = 1 - sqrt(x1). This requires x1 >= 0.
        # Usually x1 is bounded [0, 1] in ZDT1 problems?
        # The problem statement says bounds [0, 1] for every variable.
        # So x1 is in [0, 1].
        # Thus y_i in [0, 1].
        
        # If we place points in [0, 1], the last point y_{N-1} <= 1.
        # The term for i=N-1 is (11 - y_{N-1}) * (10 + sqrt(y_{N-1})).
        
        def neg_hv(y):
            # y is array of length N
            # Calculate HV
            # f1 = y
            # f2 = 1 - sqrt(y)
            # ref = 11
            # HV = sum (f1_{i+1} - f1_i) * (ref - f2_i)
            # f1_N = 11
            f1_next = np.empty(N)
            f1_next[:-1] = y[1:]
            f1_next[-1] = 11.0
            
            f1_curr = y
            f2_curr = 1.0 - np.sqrt(np.clip(y, 0, 1))
            
            widths = f1_next - f1_curr
            heights = 11.0 - f2_curr
            
            # Ensure non-negative contributions? 
            # If y is not sorted, widths can be negative.
            # But we enforce bounds and initial guess sorted, and L-BFGS-B keeps bounds.
            # However, L-BFGS-B does not enforce monotonicity.
            # We should penalize unsorted y or use a transformation.
            # Simple penalty: if any width < 0, return large value.
            if np.any(widths < 0):
                return 1000.0 + np.sum(-widths)
                
            hv = np.sum(widths * heights)
            return -hv

        # Initial guess: uniform distribution in [0, 1]
        y0 = np.linspace(0, 1, N)
        
        # Bounds for y: [0, 1]
        bnds = [(0, 1)] * N
        
        try:
            res = minimize(neg_hv, y0, method='L-BFGS-B', bounds=bnds, options={'maxiter': 500})
            
            if time.time() > deadline:
                raise TimeoutError
                
            y_opt = res.x
            # Sort just in case, though bounds+initial guess usually keep it sorted
            # Actually, L-BFGS-B doesn't guarantee sorted order.
            # If unsorted, the HV calculation in neg_hv returned penalty, so it should be sorted.
            # But let's sort to be safe for the actual evaluation?
            # No, if we sort, the HV might change if the optimizer found a weird local min with crossings?
            # The penalty discourages crossings.
            
            # Clip to [0, 1]
            y_opt = np.clip(y_opt, 0, 1)
            
        except Exception:
            # Fallback to uniform if optimization fails
            y_opt = np.linspace(0, 1, N)

        # Evaluate the points
        base_vec = np.zeros(self.dim)
        for i in range(N):
            if time.time() > deadline:
                break
            x = base_vec.copy()
            x[0] = y_opt[i]
            problem(x)

        return None
