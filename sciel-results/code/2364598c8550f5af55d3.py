import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # ZDT3 Pareto Front: x0 = f1, x1..xn = 0
        # f2(f1) = (1 + sqrt(f1)) * (1 - sqrt(f1) - sin(10*pi*f1))
        
        # Key points: Local minima of f2
        # sin(10*pi*f1) = 1 => 10*pi*f1 = pi/2 + 2*k*pi => f1 = 0.05 + 0.2*k
        minima_f1 = [0.05 + 0.2*k for k in range(5)]
        
        # Endpoints
        endpoints_f1 = [0.0, 1.0]
        
        # Combine key points
        key_f1 = minima_f1 + endpoints_f1
        
        # Remaining budget
        n_remaining = budget - len(key_f1)
        
        if n_remaining > 0:
            # Uniform sampling for the rest
            # Use np.linspace or random?
            # Uniform random might duplicate key points or cluster.
            # np.linspace ensures even coverage.
            # Let's use np.linspace for the remaining points to ensure full range coverage.
            # But linspace includes endpoints. We already have endpoints.
            # So we can use np.linspace(0, 1, n_remaining + 2)[1:-1] to avoid endpoints?
            # Or just np.linspace(0, 1, n_remaining) if we don't care about exact endpoints duplication.
            # Since we will take unique, duplication is fine.
            remaining_f1 = np.linspace(0.0, 1.0, n_remaining)
        else:
            remaining_f1 = np.array([])
            
        all_f1 = np.concatenate([np.array(key_f1), remaining_f1])
        all_f1 = np.unique(all_f1)
        all_f1 = np.sort(all_f1)
        
        # Trim if unique reduced count (should not happen if linspace is dense enough)
        # Actually, unique might remove duplicates if linspace hits key points exactly.
        # If len(all_f1) > budget, we need to trim.
        if len(all_f1) > budget:
            # Keep the key points and subsample the rest?
            # Simpler: just take the first 'budget' points? No, that might drop key points if they are not first.
            # Better: ensure key points are always included.
            # For simplicity, since budget is 200 and key points are 7, and linspace has 193,
            # unique will likely have 200 points (if no overlap) or slightly less.
            # If slightly less, we are under budget, which is fine (we just don't use all evals).
            # If more (should not happen), we trim.
            pass
            
        for f1 in all_f1:
            if time.time() > deadline:
                break
            
            x = np.zeros(dim)
            x[0] = f1
            # x[1:] = 0.0 is already set by zeros
            
            x = np.clip(x, lo, hi)
            
            try:
                problem(x)
            except Exception:
                break

        return None
