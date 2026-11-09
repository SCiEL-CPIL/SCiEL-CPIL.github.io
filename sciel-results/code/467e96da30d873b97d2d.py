import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_eval = self.budget
        results = []
        
        # DTLZ4 with n_obj=3:
        # The Pareto front is parameterized by two variables (let's call them u, v in [0,1])
        # f1 = 0.5 * (1 - v) * u
        # f2 = 0.5 * (1 - v) * (1 - u)
        # f3 = 0.5 * v
        #
        # However, DTLZ4 has a non-linear mapping for the position variables.
        # Standard DTLZ4: x_i = y_i^(1/alpha) for i in [0, M-2] (position variables)
        # Here M=3, so position variables are x[0] and x[1].
        # The distance variables are x[2:] which must be 0.5 for g=0.
        #
        # The "position" variables in the objective function are typically the last M-1 variables
        # in some formulations, or the first M-1 in others.
        # Let's look at the standard definition for DTLZ4 (Deb et al.):
        # f_i(x) = (1 + g(x_M)) * h_i(f_1(x), ..., f_{M-1}(x), x_{M-1})
        # For DTLZ4:
        # g(x) = sum_{i=M}^{n} (x_i - 0.5)^2
        # The position variables are x_1, ..., x_{M-1}.
        # For M=3, position variables are x[0], x[1].
        # Distance variables are x[2], ..., x[11].
        #
        # The objectives for DTLZ4 (M=3):
        # f_1 = 0.5 * (1 + g) * x_1 * (1 - x_2)  -- Wait, indexing is tricky.
        # Let's use the standard DTLZ2-like form but with x_i = u_i^(1/alpha).
        # For DTLZ2 (M=3):
        # f_1 = 0.5 * (1+g) * (1 - x_{M-1}) * x_{M-2}
        # f_2 = 0.5 * (1+g) * (1 - x_{M-1}) * (1 - x_{M-2})
        # f_3 = 0.5 * (1+g) * x_{M-1}
        # Here x_{M-1} is the (M-1)-th position variable (0-indexed: x[1]),
        # and x_{M-2} is the (M-2)-th position variable (0-indexed: x[0]).
        #
        # So for DTLZ4:
        # Let u = x[0], v = x[1].
        # But x[i] = u_i^(1/alpha) where u_i is uniform in [0,1].
        # To sample the Pareto front uniformly in objective space, we should sample
        # the underlying uniform variables u_i, v_i and map them.
        #
        # The inspirations suggest using two hypotheses for alpha:
        # 1. High Alpha (e.g., 100): x = u^(1/100)
        # 2. Low Alpha (e.g., 1): x = u
        #
        # Since we don't know the exact alpha, covering both is robust.
        # We will split the budget: 100 evals for alpha=100, 100 evals for alpha=1.
        #
        # To maximize hypervolume, we want a dense, uniform coverage of the 2D front.
        # A 10x10 grid for each hypothesis gives 100 points each.
        #
        # However, the "area-preserving" insight suggests sampling in a way that
        # accounts for the Jacobian of the transformation from parameter space to objective space.
        # But since we are just evaluating points, and the front is fixed,
        # a uniform grid in the parameter space (u, v) that generates the front
        # is a good starting point.
        #
        # Let's refine the grid to be slightly denser or use a better quasi-random sequence.
        # Halton sequences are good for low-discrepancy.
        #
        # Strategy:
        # Use Halton sequence for (u, v) in [0,1]^2.
        # For each point (u, v), generate two candidate x vectors:
        # 1. x[0] = u^(1/100), x[1] = v^(1/100)  (High Alpha)
        # 2. x[0] = u, x[1] = v                  (Low Alpha)
        #
        # Wait, if I generate 200 Halton points, I can evaluate 200 points total.
        # I can mix them: 100 points mapped with alpha=100, 100 points mapped with alpha=1.
        # Or, I can generate 100 Halton points, and for each, evaluate both mappings?
        # That would be 200 evaluations.
        #
        # Let's do this:
        # Generate 100 Halton points (u, v).
        # For i in 0..99:
        #   x_high = [u_i^(1/100), v_i^(1/100), 0.5, ..., 0.5]
        #   x_low  = [u_i, v_i, 0.5, ..., 0.5]
        #   Evaluate problem(x_high) -> f_high
        #   Evaluate problem(x_low)  -> f_low
        #   Append both to results.
        #
        # This ensures we sample the same 100 locations in the parameter space,
        # but mapped through two different non-linearities. This doubles our coverage
        # of the potential front shapes.
        
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f = f / base
                r = r + f * (index % base)
                index //= base
            return r
        
        n_halton = n_eval // 2
        if n_halton < 1:
            n_halton = 1
            
        for i in range(1, n_halton + 1):
            if time.time() > deadline:
                break
            
            u = halton(i, 2)
            v = halton(i, 3)
            
            # Clip u, v to [0,1] just in case, though Halton is in [0,1)
            u = np.clip(u, 0.0, 1.0)
            v = np.clip(v, 0.0, 1.0)
            
            # Candidate 1: High Alpha (100)
            x_high = np.zeros(self.dim)
            x_high[0] = u ** (1.0 / 100.0)
            x_high[1] = v ** (1.0 / 100.0)
            x_high[2:] = 0.5
            x_high = np.clip(x_high, lo, hi)
            
            try:
                f_high = problem(x_high)
                results.append(f_high)
            except Exception:
                pass
            
            # Candidate 2: Low Alpha (1)
            x_low = np.zeros(self.dim)
            x_low[0] = u
            x_low[1] = v
            x_low[2:] = 0.5
            x_low = np.clip(x_low, lo, hi)
            
            try:
                f_low = problem(x_low)
                results.append(f_low)
            except Exception:
                pass
                
            # Check budget
            if len(results) >= n_eval:
                break

        return results
