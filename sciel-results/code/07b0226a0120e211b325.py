"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Sobol quasi-random sampling for multi-objective optimization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Generate Sobol sequence points
        # scipy.stats.qmc.Sobol is efficient and deterministic
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
            # We need enough points. 2**k points. 
            # For dim=4, 2^10 = 1024 points is plenty for 200 budget.
            # Let's generate 1024 points to have a good pool, or just generate exactly what we need?
            # Sobol can generate any number, but power of 2 is standard.
            # Let's generate 200 points directly? Sobol is often better with powers of 2.
            # Let's generate 256 points (2^8) which is > 200.
            n_points = 256
            sobol_points = sampler.random(n_points)
        except ImportError:
            # Fallback to random if scipy qmc is not available (though it should be)
            rng = np.random.default_rng(42)
            sobol_points = rng.random((self.budget, self.dim))

        # Map to bounds
        X = lo + (hi - lo) * sobol_points
        
        # Evaluate points
        # We have budget 200. We have 256 points. We will evaluate the first 200.
        # Or we can be smarter: evaluate all 256 if time permits? No, budget is 200 evaluations.
        # "More than 200 evaluations ... makes the whole result invalid"
        # So we must strictly limit to 200 calls to problem().
        
        evaluated_X = []
        evaluated_F = []
        
        for i in range(self.budget):
            if time.time() > deadline - 5:
                break
                
            x = X[i]
            f = problem(x)
            evaluated_X.append(x)
            evaluated_F.append(f)
            
        return evaluated_X
# EVOLVE-BLOCK-END
