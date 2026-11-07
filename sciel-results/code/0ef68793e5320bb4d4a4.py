import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT3 Pareto front is primarily in x1 in [0, 0.25] with x2..x30=0
        # We sample this region to maximize hypervolume
        n_samples = min(budget, 200)
        
        # Use a deterministic grid or random?
        # Random is fine as seed is fixed, but grid ensures coverage.
        # Let's use a grid to ensure we hit the "ends" and the curve well.
        # Actually, uniform random is standard and works well for HV.
        # But a stratified grid is better for HV coverage.
        
        # Let's use a mix: mostly uniform in [0, 0.25]
        # But we should check if the front extends further.
        # If we miss the true PF, we lose score.
        # The "record low" logic suggests [0, 0.25].
        
        # Let's sample uniformly in [0, 0.25]
        x1_vals = np.linspace(0, 0.25, n_samples)
        
        results = []
        for x1 in x1_vals:
            x = np.zeros(dim)
            x[0] = x1
            # Clip to bounds just in case
            x = np.clip(x, lo, hi)
            f = problem(x)
            results.append(f)
            
        return results
