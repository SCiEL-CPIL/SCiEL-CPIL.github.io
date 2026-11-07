"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Deterministic multi-objective search using Sobol sequences.
    Uses a space-filling design to ensure broad coverage of the 12D search space,
    maintaining a non-dominated front to maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 290.0  # Safety margin from 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        
        # Sobol sequence generation
        # We need up to 'budget' points.
        # scipy.stats.qmc.Sobol is efficient and available.
        from scipy.stats import qm
        
        # Generate Sobol points
        # qm.Sobol(d=dim, scramble=False)
        # We want deterministic, low-discrepancy.
        # To maximize hypervolume, we want points spread out.
        # Standard Sobol sequence provides this.
        
        # Note: The task says "Python standard library, NumPy and SciPy are available."
        # So we can use scipy.
        
        sobol = qm.Sobol(d=dim, scramble=False)
        # We need to generate enough points. 
        # Sobol requires n to be a power of 2 for the standard generator? 
        # Actually, scipy.stats.qm.Sobol.random(n) works for any n, but the sequence is defined for powers of 2.
        # It's better to generate a power of 2 >= budget and take the first 'budget' points.
        
        n_gen = 1
        while n_gen < self.budget:
            n_gen *= 2
            
        # Generate Sobol points in [0, 1]^dim
        # random(n_gen) returns (n_gen, dim)
        sobol_points = sobol.random(n_gen)
        
        # Take the first 'budget' points
        sobol_points = sobol_points[:self.budget]
        
        # Map to bounds [lo, hi]
        # x = lo + (hi - lo) * sobol_point
        X = lo + (hi - lo) * sobol_points
        
        # Evaluate all points
        # We must respect the budget. We have exactly 'budget' points.
        # We must also respect the time limit.
        
        F = np.zeros((self.budget, self.n_obj))
        
        # We will evaluate in order.
        # If time runs out, we stop and return what we have.
        # But the contract says "More than 200 evaluations... makes the whole result invalid".
        # So we must not exceed the budget.
        # We have exactly 'budget' points prepared.
        
        # However, if the problem evaluation is slow, we might hit the time limit.
        # In that case, we should stop evaluating and return the points we've evaluated so far.
        # But the return value is ignored, only recorded evaluations count.
        # So we just need to ensure we don't crash or exceed budget.
        
        # Let's evaluate one by one to allow early termination on time limit.
        # If we use a vectorized call, we might not be able to stop mid-batch if time runs out.
        # But problem(x) takes a single vector. So we must loop.
        
        for i in range(self.budget):
            if time.time() > deadline:
                # Break out if we are close to the limit
                # We have evaluated i points.
                break
            
            F[i] = problem(X[i])
            
        # The recorded vectors are those successfully evaluated.
        # The evaluator records every returned objective vector.
        # Since we call problem(x) in a loop, each call records a vector.
        
        # We don't need to return anything specific, but the signature says __call__ returns something.
        # The prompt says "the return value of __call__ is ignored".
        # So we can return None or the front.
        
        return None

# EVOLVE-BLOCK-END
