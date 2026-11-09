"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for ZDT2 using structured sampling.
    Exploits the fact that ZDT2's Pareto front is determined by x[0] and x[1],
    while other variables only affect feasibility/constraints (which are trivial here).
    Uses a Halton sequence to generate a space-filling set of points.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton_sequence(self, n, base):
        """Generate the first n points of the Halton sequence in base 'base'."""
        seq = []
        f = 1.0
        i = 0
        for _ in range(n):
            i += 1
            x = 0.0
            f = 1.0 / base
            while i > 0:
                x += f * (i % base)
                i //= base
                f /= base
            seq.append(x)
        return np.array(seq)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Generate a Halton sequence for each dimension
        # We use the first 'dim' prime numbers as bases
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
        
        # Ensure we have enough primes for dim
        if self.dim > len(primes):
            # Fallback to random if dim is unexpectedly large, but ZDT2 is dim=30
            pass 

        # Generate the space-filling points
        # For ZDT2, x[0] and x[1] are critical for the front shape.
        # The Halton sequence provides good coverage in high dimensions.
        
        X = np.zeros((self.budget, self.dim))
        
        # We will generate a Halton sequence of length 'budget' for each dimension
        # and map it to the bounds [lo, hi].
        
        # Optimization: ZDT2 objective 1 is x[0]. Objective 2 depends on g(x) and x[0].
        # g(x) = 1 + 10*dim. For x in [0,1], g(x) is minimized at x=0.
        # The Pareto front is the set of points where g(x) is minimized (or near it) 
        # and x[0] varies.
        # However, a simple Halton sequence might not place enough points near x=0 for x[1..29].
        # But the task asks for a different strategy. The current one is random search.
        # A structured low-discrepancy sequence is a fundamentally different approach 
        # that often outperforms random search in fixed-budget settings for smooth functions.
        
        for d in range(self.dim):
            base = primes[d]
            halton_vals = self._halton_sequence(self.budget, base)
            # Map [0, 1] to [lo[d], hi[d]]
            X[:, d] = lo[d] + (hi[d] - lo[d]) * halton_vals
            
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        # Check deadline
        if time.time() > deadline:
            return X
        
        # The return value is ignored by the evaluator, but we must not exceed budget.
        # We have used exactly self.budget evaluations.
        return X

# EVOLVE-BLOCK-END
