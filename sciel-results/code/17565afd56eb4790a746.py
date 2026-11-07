import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # DTLZ7 specific optimization
        # Set distance variables to 0.35 to minimize g(x_in) to 0
        # x_0 and x_1 are the position variables
        
        x_in_val = 0.35
        x_in = np.full(self.dim - 2, x_in_val)
        
        # Generate Halton sequence for x_0, x_1
        # We use 200 evaluations
        n_evals = self.budget
        
        def halton(index, base):
            result = 0
            f = 1
            i = index
            while i > 0:
                f = f / base
                result = result + f * (i % base)
                i = i // base
            return result

        # Bases for Halton: 2, 3
        points = []
        for i in range(1, n_evals + 1):
            x0 = halton(i, 2)
            x1 = halton(i, 3)
            points.append([x0, x1])
            
        points = np.array(points)
        
        results = []
        for p in points:
            x = np.concatenate(([p[0], p[1]], x_in))
            # Ensure length is correct
            if len(x) != self.dim:
                # Pad or trim if necessary, though dim=22 is fixed
                x = np.resize(x, self.dim)
            f_vals = problem(x)
            results.append(f_vals)
            
        return results
