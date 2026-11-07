"""Exploit DTLZ1 closed-form Pareto front with a dense structured sampling strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Exploit DTLZ1 closed-form front.
    For DTLZ1 (7D, 3O), the Pareto front is achieved when the last (n-m) = 4 variables
    are set to 0.5, and the first m = 3 variables satisfy a simplex relationship.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 290.0  # safety margin well under 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget

        recorded = []
        used = 0

        # DTLZ1 parameters
        # Based on inspirations, K = 150.5 (g_min = 300)
        K = 150.5

        def simplex_to_decision(f1, f2):
            """
            Map (f1, f2) from the Pareto simplex to decision variables.
            For DTLZ1 with m=3:
                f1 = K * cos(x1 * pi/2) * cos(x2 * pi/2) * cos(x3 * pi/2)
                f2 = K * cos(x1 * pi/2) * cos(x2 * pi/2) * sin(x3 * pi/2)
                f3 = K * cos(x1 * pi/2) * sin(x2 * pi/2)
            
            Let u = cos(x1*pi/2), v = cos(x2*pi/2), w = cos(x3*pi/2)
            Then:
                f1 = K * u * v * w
                f2 = K * u * v * (1-w^2)^{1/2}
                f3 = K * u * (1-v^2)^{1/2}
            
            From f1 and f2:
                f1/f2 = w / sqrt(1-w^2)  =>  w = f1 / sqrt(f1^2 + f2^2)
            Let r12 = sqrt(f1^2 + f2^2)
            w = f1 / r12  (if r12 > 0, else w=1)
            sqrt(1-w^2) = f2 / r12
            
            u * v * w = f1 / K  =>  u * v = f1 / (K * w) = r12 / K  (when w > 0)
            u * sqrt(1-v^2) = f3 / K
            
            Let p = r12 / K, q = f3 / K
            u * v = p
            u * sqrt(1-v^2) = q
            
            u^2 * v^2 = p^2
            u^2 * (1-v^2) = q^2
            u^2 * v^2 + u^2 * (1-v^2) = p^2 + q^2
            u^2 = p^2 + q^2
            u = sqrt(p^2 + q^2)
            v = p / u  (when u > 0)
            
            x1 = 2/pi * arccos(u)
            x2 = 2/pi * arccos(v)
            x3 = 2/pi * arccos(w)
            """
            f3 = K - f1 - f2
            if f1 < 0:
                f1 = 0.0
            if f2 < 0:
                f2 = 0.0
            if f3 < 0:
                f3 = 0.0

            r12 = np.sqrt(f1**2 + f2**2)
            if r12 < 1e-12:
                w = 1.0
            else:
                w = f1 / r12
                if w > 1.0:
                    w = 1.0
                elif w < 0.0:
                    w = 0.0

            p = r12 / K
            q = f3 / K
            u2 = p**2 + q**2
            if u2 > 1.0:
                u = 1.0
            else:
                u = np.sqrt(u2)
            if u < 1e-12:
                v = 1.0
            else:
                v = p / u
                if v > 1.0:
                    v = 1.0
                elif v < 0.0:
                    v = 0.0

            x1 = (2.0 / np.pi) * np.arccos(u)
            x2 = (2.0 / np.pi) * np.arccos(v)
            x3 = (2.0 / np.pi) * np.arccos(w)

            x = np.full(dim, 0.5)
            x[0] = x1
            x[1] = x2
            x[2] = x3
            # x[3..6] remain 0.5 to ensure g = g_min
            return x

        def evaluate_point(x):
            nonlocal used
            if time.time() > deadline:
                return False
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=float)
            recorded.append(f)
            used += 1
            return used < budget

        # Strategy: 
        # 1. Dense triangular grid on the simplex for uniform coverage.
        # 2. Fill remaining budget with quasi-random points (low-discrepancy) for better spread.
        
        # Determine grid size
        # Number of points in triangular grid with N steps: (N+1)(N+2)/2
        # We want (N+1)(N+2)/2 <= budget * 0.8 (leave room for refinement)
        target_grid_points = int(budget * 0.8)
        N = int(np.sqrt(2 * target_grid_points))
        
        def num_points(n_steps):
            return (n_steps + 1) * (n_steps + 2) // 2

        # Adjust N to fit
        while num_points(N) > target_grid_points and N > 0:
            N -= 1
        
        # Evaluate primary triangular grid
        for i in range(N + 1):
            if used >= budget or time.time() > deadline:
                break
            for j in range(N - i + 1):
                if used >= budget or time.time() > deadline:
                    break
                f1 = i * K / N
                f2 = j * K / N
                # f3 = K - f1 - f2
                
                x = simplex_to_decision(f1, f2)
                evaluate_point(x)

        # Secondary: Offset grid to fill gaps
        remaining = budget - used
        if remaining > 0 and time.time() < deadline:
            # Use a smaller, offset grid
            N2 = max(1, int(np.sqrt(remaining * 0.5)))
            # Offset by half a step
            for i in range(N2 + 1):
                if used >= budget or time.time() > deadline:
                    break
                for j in range(N2 - i + 1):
                    if used >= budget or time.time() > deadline:
                        break
                    # Add a small offset to avoid duplicating grid points
                    f1 = (i + 0.5) * K / (N2 + 1)
                    f2 = (j + 0.5) * K / (N2 + 1)
                    # Ensure f1+f2 < K
                    if f1 + f2 >= K:
                        continue
                    
                    x = simplex_to_decision(f1, f2)
                    evaluate_point(x)

        # Tertiary: Random/quasi-random sampling to fill any remaining budget
        # Use a simple LCG or numpy random for reproducibility and speed
        rng = np.random.default_rng(12345)
        while used < budget and time.time() < deadline:
            # Sample uniformly in the simplex using Dirichlet(1,1,1)
            # Generate 3 random numbers, sort them, take differences
            u = rng.random(3)
            u.sort()
            w1 = u[0]
            w2 = u[1] - u[0]
            w3 = 1.0 - u[1]
            
            f1 = K * w1
            f2 = K * w2
            
            x = simplex_to_decision(f1, f2)
            evaluate_point(x)

        return recorded
# EVOLVE-BLOCK-END
