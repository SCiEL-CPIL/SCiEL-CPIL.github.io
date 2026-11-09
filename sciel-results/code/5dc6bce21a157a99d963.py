"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Quasi-random sampling optimizer using Sobol sequences for RE37."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        
        # Generate Sobol sequence points
        # We need 'budget' points. 
        # scipy.stats.qmc.Sobol generates points in [0, 1]^dim
        try:
            from scipy.stats.qmc import Sobol
            # Number of points must be a power of 2 for Sobol to be efficient, 
            # but we can just generate enough and take the first 'budget'
            # Or generate exactly 'budget' if the library allows (it does, though efficiency drops slightly)
            sampler = Sobol(d=self.dim, scramble=True, seed=42)
            # Generate a power of 2 >= budget for better properties, or just budget
            # For 200, next power of 2 is 256.
            n_gen = 1
            while n_gen < self.budget:
                n_gen *= 2
            
            # Sobol sample
            points = sampler.random(n_gen)
            # Take only the first 'budget' points
            points = points[:self.budget]
            
        except ImportError:
            # Fallback to Latin Hypercube or random if SciPy QMC is missing (unlikely per contract)
            rng = np.random.default_rng(42)
            points = rng.random((self.budget, self.dim))
        
        # Map from [0, 1] to [lo, hi]
        X = lo + (hi - lo) * points
        
        # Evaluate
        # We must ensure we don't exceed the budget or time.
        # The contract says "More than 200 evaluations ... makes the whole result invalid".
        # So we strictly evaluate exactly self.budget points.
        
        F_list = []
        for i in range(self.budget):
            if time.time() > deadline:
                break
            x = X[i]
            try:
                f = problem(x)
                F_list.append(f)
            except Exception:
                # If an exception occurs, we stop. The contract says "an exception later keeps the evaluations made".
                break
        
        # The return value is ignored, but we return something to be safe
        return np.array(F_list) if F_list else None
# EVOLVE-BLOCK-END
