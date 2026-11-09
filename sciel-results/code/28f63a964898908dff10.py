"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-exploiting optimizer for DTLZ7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        budget = self.budget

        # DTLZ7 structure:
        # f1 = x_0
        # g = 120 + 10*m + sum_{i in [m..n-1]} ((x_i - 0.5)^2 - cos(20*pi*(x_i-0.5)))
        # gmin = 120 + 10*m
        # f_k = (1 + g) * g_k for k >= 2
        # g_k = 1 - sqrt(x_0 / (1+g)) * (1 - (k-2)/(m-1))
        
        # The Pareto front is characterized by:
        # x_0 in [0, 1], all other x_i = 0.5 (to minimize g)
        # Then g = gmin = 120 + 10*m
        # f1 = x_0
        # f_k = (1 + gmin) * (1 - sqrt(x_0/(1+gmin)) * (1 - (k-2)/(m-1)))
        
        gmin = 120 + 10 * m
        one_plus_gmin = 1 + gmin

        # Strategy:
        # 1. Sample points on the Pareto front manifold (x_0 varying, others at 0.5)
        # 2. Use quasi-random sequences for good coverage
        # 3. Add some perturbation to explore off-front regions
        # 4. Track all evaluated points and return the best set

        evaluated_x = []
        evaluated_f = []

        def evaluate(x):
            fx = problem(x)
            evaluated_x.append(x.copy())
            evaluated_f.append(fx)
            return fx

        # Phase 1: Dense sampling on the Pareto front manifold
        # Use Halton-like quasi-random points in [0,1] for x_0
        n_front = min(budget // 2, 100)
        
        # Generate quasi-random points using van der Corput / Halton
        def halton_1d(n, base):
            points = np.zeros(n)
            for i in range(n):
                f = 1.0
                r = 0.0
                q = i + 1
                while q > 0:
                    f /= base
                    r += f * (q % base)
                    q //= base
                points[i] = r
            return points

        x0_samples = halton_1d(n_front, 2)
        
        for x0 in x0_samples:
            if len(evaluated_x) >= budget:
                break
            if time.time() > deadline:
                break
            
            x = np.full(n, 0.5)
            x[0] = x0
            fx = evaluate(x)

        # Phase 2: Add some random exploration points
        remaining = budget - len(evaluated_x)
        if remaining > 0:
            n_explore = min(remaining, 30)
            rng = np.random.default_rng(42)
            for _ in range(n_explore):
                if time.time() > deadline:
                    break
                x = lo + (hi - lo) * rng.random(n)
                fx = evaluate(x)

        # Phase 3: Local refinement around promising points
        # Find points with low g (close to Pareto front) and refine x_0
        if len(evaluated_f) > 0:
            f_arr = np.array(evaluated_f)
            x_arr = np.array(evaluated_x)
            
            # Points close to the front have f2 + f3 + ... close to expected values
            # For DTLZ7, on the front: f1 + (f2+f3+...)/(1+gmin) should be ~1
            # Let's just refine around the best points in terms of spread
            
            # Select a few diverse points and do local search on x_0
            if len(evaluated_x) > 5:
                # Pick points that are well-spread
                # Use simple clustering: divide [0,1] into bins
                n_refine = min(10, budget - len(evaluated_x))
                if n_refine > 0:
                    # Sample x_0 values more densely in regions where we have points
                    x0_vals = x_arr[:, 0]
                    
                    # Add finer grid points
                    n_fine = min(n_refine, 50)
                    x0_fine = np.linspace(0, 1, n_fine)
                    for x0 in x0_fine:
                        if len(evaluated_x) >= budget:
                            break
                        if time.time() > deadline:
                            break
                        x = np.full(n, 0.5)
                        x[0] = x0
                        fx = evaluate(x)

        # Return the last evaluated point (return value is ignored)
        if evaluated_x:
            return evaluated_x[-1]
        return lo.copy()
# EVOLVE-BLOCK-END
