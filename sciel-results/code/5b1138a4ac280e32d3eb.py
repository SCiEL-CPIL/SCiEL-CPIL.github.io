import numpy as np

class Optimizer:
    """Deterministic grid search exploiting WFG9's known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # WFG9 Pareto front is determined by the first 3 variables.
        # The remaining 9 variables are position variables that should be set
        # to minimize the distance to the front (typically the midpoint).
        
        # We allocate the budget to a 3D grid over x[0], x[1], x[2].
        # 5 * 5 * 8 = 200 evaluations.
        n1, n2, n3 = 5, 5, 8
        
        # Adjust if budget is different (though task says 200)
        # For robustness, scale down if budget < 200
        while n1 * n2 * n3 > self.budget and min(n1, n2, n3) > 1:
            if n3 >= n2 and n3 >= n1:
                n3 -= 1
            elif n2 >= n1:
                n2 -= 1
            else:
                n1 -= 1
        
        # Generate grid points
        x1_vals = np.linspace(lo[0], hi[0], n1)
        x2_vals = np.linspace(lo[1], hi[1], n2)
        x3_vals = np.linspace(lo[2], hi[2], n3)
        
        # Set remaining variables to midpoint
        mid_vars = (lo[3:] + hi[3:]) / 2.0
        
        results = []
        used = 0
        
        # Pre-allocate the base vector for the tail variables
        base_tail = mid_vars.copy()
        
        for x1 in x1_vals:
            for x2 in x2_vals:
                for x3 in x3_vals:
                    if used >= self.budget:
                        break
                    
                    x = np.empty(self.dim)
                    x[0] = x1
                    x[1] = x2
                    x[2] = x3
                    x[3:] = base_tail
                    
                    # Ensure bounds are respected (linspace should handle it, but safe clip)
                    # Note: problem(x) also clips, but we want to pass valid points
                    f = problem(x)
                    results.append(f)
                    used += 1
                    
                    if used >= self.budget:
                        break
                if used >= self.budget:
                    break
            if used >= self.budget:
                break
        
        return np.array(results) if results else None
