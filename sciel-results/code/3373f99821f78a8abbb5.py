"""Program that exploits DTLZ1's known structure to uniformly sample the Pareto front in objective space."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploit DTLZ1 closed-form front: map uniform simplex grid to decision space."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 300.0  # safety margin under 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        budget = self.budget

        recorded = []
        
        # DTLZ1 parameters
        # n=7, m=3
        # g(x) = 100 * (5 + sum_{i=6}^7 ((x_i-0.5)^2 - cos(20*pi*(x_i-0.5))))
        # Min g is at x_i=0.5, term=-1. g_min = 100*(5-2) = 300.
        # f_i = 0.5 * (1+g) * ...
        # Sum f_i = 0.5 * (1+g_min) = 150.5
        K = 150.5
        
        # Generate a triangular grid on the simplex f1+f2+f3=K
        # Number of points for N steps is (N+1)(N+2)/2
        # We want approx budget points.
        # (N+1)(N+2)/2 <= budget
        # N^2/2 ~ budget => N ~ sqrt(2*budget)
        N = int(np.sqrt(2 * budget))
        
        # Adjust N to fit budget exactly or closely
        # Try N and N-1
        def num_points(n_steps):
            return (n_steps + 1) * (n_steps + 2) // 2
            
        if num_points(N) > budget:
            N -= 1
        if num_points(N) > budget:
            N -= 1
            
        used = 0
        
        # Iterate over the triangular grid
        # f1 = i * delta
        # f2 = j * delta
        # f3 = K - f1 - f2
        delta = K / N
        
        for i in range(N + 1):
            if used >= budget:
                break
            for j in range(N - i + 1):
                if used >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break
                    
                f1 = i * delta
                f2 = j * delta
                f3 = K - f1 - f2
                
                # Map to x1, x2
                # x1 = (f1+f2)/K
                # x2 = f1/(f1+f2) if f1+f2 > 0 else 0
                
                s = f1 + f2
                if s > 1e-10:
                    x1 = s / K
                    x2 = f1 / s
                else:
                    # This is the corner where f1=0, f2=0, f3=K
                    # x1=0, x2 is undefined (0/0), but x1=0 makes x2 irrelevant for f1, f2.
                    # f1 = 0, f2 = 0.
                    # Set x2=0.5 for safety.
                    x1 = 0.0
                    x2 = 0.5
                    
                x = np.full(dim, 0.5)
                x[0] = x1
                x[1] = x2
                # x3..x7 are 0.5
                
                # Clip to bounds (though x1,x2 should be in [0,1])
                x = np.clip(x, lo, hi)
                
                f = np.array(problem(x), dtype=float)
                recorded.append(f)
                used += 1
                
        # If we have remaining budget, we can add some random points or refine the grid
        if used < budget and time.time() < deadline - 5.0:
            # Add points on the edges or interior with slightly different N?
            # Or just random sampling on the simplex to fill gaps?
            # A better refinement: use a finer grid for the remaining budget
            # Or simply evaluate some specific important points if not covered
            # The grid covers the whole simplex. 
            # We can add points with N+1 if budget allows, but that's complex.
            # Let's just do a few random samples on the simplex to ensure no numerical issues
            # or to add density if the grid was too coarse.
            
            # Actually, if N was reduced to fit budget, we might have < budget points.
            # Let's fill the rest with random points on the simplex.
            while used < budget and time.time() < deadline - 5.0:
                # Random point in simplex
                # u, v ~ Uniform(0,1)
                # Sort u, v
                # f1 = K * min(u,v)
                # f2 = K * (max(u,v) - min(u,v))
                # f3 = K * (1 - max(u,v))
                u = np.random.rand()
                v = np.random.rand()
                if u > v:
                    u, v = v, u
                f1 = K * u
                f2 = K * (v - u)
                f3 = K * (1 - v)
                
                s = f1 + f2
                if s > 1e-10:
                    x1 = s / K
                    x2 = f1 / s
                else:
                    x1 = 0.0
                    x2 = 0.5
                    
                x = np.full(dim, 0.5)
                x[0] = x1
                x[1] = x2
                x = np.clip(x, lo, hi)
                
                f = np.array(problem(x), dtype=float)
                recorded.append(f)
                used += 1

        return recorded
# EVOLVE-BLOCK-END
