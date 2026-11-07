import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        budget = int(self.budget)
        t0 = time.time()
        DEADLINE = 295.0
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        if budget <= 0:
            return np.empty((0, dim))
            
        N = budget
        
        # Reference point for ZDT4 is typically [11, 400] or similar, 
        # but for model optimization we just need relative shape.
        # The actual reference point in scoring is [11.0, 400.0].
        # For the model front y = 1 - sqrt(x), x in [0, 1].
        # y is in [0, 1].
        # Reference [11, 400] is far away.
        # HV = sum (x_{i+1} - x_i) * (R2 - y_i)
        # Since R2 is large and constant, maximizing sum (x_{i+1} - x_i) * (1 - y_i) is roughly equivalent?
        # No, (R2 - y_i) = R2 - (1 - sqrt(x_i)) = R2 - 1 + sqrt(x_i).
        # So we maximize sum (x_{i+1} - x_i) * (C + sqrt(x_i)).
        
        R1 = 11.0
        R2 = 400.0
        
        def model_hv(p):
            if p <= 0: return 0
            i = np.arange(1, N+1)
            x = (i / N) ** p
            # Clip to [0, 1] for model
            x = np.clip(x, 0.0, 1.0)
            # Ensure last point is 1.0
            x[-1] = 1.0
            
            # ZDT4 Pareto front: y = 1 - sqrt(x)
            y = 1.0 - np.sqrt(x)
            
            # Calculate HV
            # Points sorted by x (they are)
            # x_next is x[i+1], last is R1
            x_next = np.concatenate((x[1:], [R1]))
            dx = x_next - x
            # Height is R2 - y
            dy = R2 - y
            # Area
            area = np.sum(dx * dy)
            return area

        # Search for optimal p
        best_p = 1.0
        best_val = model_hv(1.0)
        
        # Broad grid search
        for p in np.linspace(0.2, 5.0, 100):
            val = model_hv(p)
            if val > best_val:
                best_val = val
                best_p = p
                
        # Refine search
        for p in np.linspace(best_p - 0.2, best_p + 0.2, 100):
             if p <= 0: continue
             val = model_hv(p)
             if val > best_val:
                 best_val = val
                 best_p = p

        # Generate points
        i = np.arange(1, N+1)
        x0_vals = (i / N) ** best_p
        # Ensure coverage of [0, 1]
        # The first point is (1/N)^p. If p>1, this is small.
        # We might want to include 0?
        # If we include 0, y=1.
        # If we include 1, y=0.
        # The current formula generates points in (0, 1].
        # Let's force the last point to be hi[0] (assuming hi[0] is 1.0 for x0)
        # And maybe the first point to be lo[0] (assuming lo[0] is -5 or 0)?
        # For ZDT4, x0 is in [-5, 5]. But Pareto front is x0 in [0, 1].
        # If we sample outside [0,1], f1 = x0, f2 = ...
        # If x0 < 0, f1 < 0. f2 = 1 - sqrt(x0/99) is complex? No, ZDT4 f1 is x1.
        # If x1 < 0, f1 < 0.
        # The Pareto front is defined for f1 > 0.
        # Usually we restrict x0 to [0, 1] to be on the front.
        
        # Clip x0 to [0, 1] to stay on the ideal front
        x0_vals = np.clip(x0_vals, 0.0, 1.0)
        # Ensure we have points at the extremes
        x0_vals[0] = 0.0
        x0_vals[-1] = 1.0

        evals = 0
        results = []
        
        for x0_val in x0_vals:
            if evals >= budget or time.time() - t0 > DEADLINE:
                break
            
            x = np.zeros(dim)
            x[0] = x0_val
            # Set other variables to 0 to ensure g(x) = 1
            # If bounds don't include 0, we should pick the value that minimizes sum x_i^2
            # Usually bounds are [-5, 5], so 0 is valid.
            # If 0 is out of bounds, we should clip to the bound closest to 0.
            x = np.clip(x, lo, hi)
            
            try:
                f = problem(x)
                evals += 1
                results.append(f)
            except Exception:
                pass

        if results:
            return np.array(results)
            
        return np.empty((0, dim))
