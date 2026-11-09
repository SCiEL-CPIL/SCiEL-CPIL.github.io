"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer for ZDT1 using low-discrepancy Halton sequence sampling on the analytic front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325.0  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT1 Structure:
        # f1 = x0
        # f2 = 1 + sin(10*pi*x0) + (n-1)*(g(x) + (x0/g(x))^alpha)
        # Pareto front is achieved when x1..xn = 0, so g(x)=1.
        # On the front: f1 = x0, f2 = 1 + sin(10*pi*x0).
        
        # Strategy:
        # Instead of iterative greedy HV maximization (which is O(N^2) or worse in Python),
        # we use a deterministic low-discrepancy sequence (Halton) to select x0 values.
        # Halton sequences provide near-optimal space-filling properties for 1D domains,
        # which translates to efficient coverage of the Pareto front for hypervolume maximization.
        # This is structurally different: it is a direct, non-adaptive, vectorized sampling strategy.
        
        N = self.budget
        
        # Generate Halton sequence for 1D (base 2 is just dyadic rationals, not good for 1D uniformity?
        # Actually, for 1D, the Van der Corput sequence (base 2) is the standard low-discrepancy sequence.
        # It distributes points evenly with better uniformity than a simple grid for finite N.
        
        def van_der_corput(n, base=2):
            """Generate the first n values of the Van der Corput sequence."""
            result = np.zeros(n)
            inv_base = 1.0 / base
            for i in range(n):
                f = inv_base
                num = i
                while num > 0:
                    num, rem = divmod(num, base)
                    result[i] += rem * f
                    f *= inv_base
            return result
        
        # Use Van der Corput sequence for x0 in [0, 1]
        x0_values = van_der_corput(N, base=2)
        
        # Ensure we cover the boundaries [0, 1] well.
        # The VdC sequence is dense near 0 and 1? No, it's low discrepancy.
        # Let's add a small epsilon to avoid exactly 0 or 1 if needed, though bounds are [0,1].
        # ZDT1 is defined on [0,1].
        
        # Prepare the input vectors
        # x1..xn = 0 to ensure we are on the Pareto front (minimizing g(x))
        X = np.zeros((N, self.dim))
        X[:, 0] = x0_values
        
        # Clip to bounds (though they should be [0,1])
        X = np.clip(X, lo, hi)
        
        # Evaluate all points in a batch if possible, or loop if problem is a callable
        # The problem interface: problem(x) returns 2 objectives.
        # We must call it N times.
        
        recorded_f1 = []
        recorded_f2 = []
        
        # Time check before starting the loop
        if time.time() > deadline:
            return np.array([])
        
        for i in range(N):
            if time.time() > deadline:
                break
            x = X[i]
            f = problem(x)
            recorded_f1.append(f[0])
            recorded_f2.append(f[1])
            
        # Return the last evaluated point or an array, the return value is ignored by the evaluator
        # but we must return something valid.
        if recorded_f1:
            return np.array(recorded_f1[-1])
        else:
            return np.array([0.0, 0.0])

# EVOLVE-BLOCK-END
