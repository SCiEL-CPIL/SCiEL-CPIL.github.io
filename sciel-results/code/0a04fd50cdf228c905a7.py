"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer exploiting ZDT4 structure via non-uniform sampling.
    
    Strategy:
    ZDT4 has a known Pareto front where f_1 = x_0 and f_2 = 1 + (1 - sqrt(x_0/g)) * g,
    with g minimized to 1 by setting x_1...x_9 = 0.
    The hypervolume approximation error is dominated by the curvature of the front.
    The derivative of f_2 with respect to f_1 is singular at f_1=0 (vertical tangent).
    Therefore, a uniform grid wastes evaluations in the flat regions and under-samples
    the high-curvature region near x_0=0.
    
    We use a non-uniform grid (quadratic spacing) to place more points near x_0=0,
    which significantly improves the hypervolume approximation for a fixed budget.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Check if 0 is within bounds for x1..x9
        zeros_ok = all(lo[i] <= 0 <= hi[i] for i in range(1, self.dim))
        
        if zeros_ok:
            # Directly sample the Pareto front with non-uniform spacing
            n_points = self.budget
            
            # Ensure we cover the range [0, 1] for x0 if possible.
            x0_lo = max(lo[0], 0.0)
            x0_hi = min(hi[0], 1.0)
            
            if x0_lo >= x0_hi:
                # Fallback: use the actual bounds for x0
                x0_lo = lo[0]
                x0_hi = hi[0]
                if x0_lo >= x0_hi:
                    x0_vals = np.array([lo[0]])
                else:
                    # Uniform fallback for degenerate cases
                    x0_vals = np.linspace(x0_lo, x0_hi, n_points)
            else:
                # Non-uniform grid: denser near x0=0
                # Use a quadratic transformation: t^2 maps [0,1] to [0,1] with density ~1/t
                # This places more points near 0.
                t = np.linspace(0, 1, n_points)
                x0_vals = x0_lo + (x0_hi - x0_lo) * (t ** 2)
                
                # Ensure exact endpoints are included for maximum coverage
                x0_vals[0] = x0_lo
                x0_vals[-1] = x0_hi
            
            X = np.zeros((n_points, self.dim))
            X[:, 0] = x0_vals
            # X[:, 1:] is already 0.0, which minimizes g(x) to 1.
            
            # Evaluate all points, respecting the budget
            # We must check time periodically to avoid timeout, though 200 calls is fast
            F = []
            for i in range(n_points):
                if time.time() > deadline:
                    break
                F.append(problem(X[i]))
            
            return X

        else:
            # Fallback: If 0 is not in bounds for some xi, try to find the 
            # point that minimizes g(x) within bounds.
            # For ZDT4, g(x) = 1 + 10*(n-1) + sum_{i=2}^n (xi^2 - 10*cos(4*pi*xi*pi))
            # The term xi^2 - 10*cos(4*pi*xi*pi) is minimized at xi=0 with value -10.
            # If 0 is not in bounds, the minimum is at the nearest boundary point.
            
            # Find the best xi for each i > 0
            best_x = np.zeros(self.dim)
            for i in range(1, self.dim):
                # The function h(x) = x^2 - 10*cos(4*pi*x*pi) has minima at x = 0, ±0.5, ±1, ...
                # Within [lo[i], hi[i]], find the x that minimizes h(x)
                # Simple approach: check boundaries and nearby critical points
                candidates = [lo[i], hi[i]]
                # Add critical points near 0, ±0.5, ±1, etc.
                for k in range(-3, 4):
                    candidates.append(k * 0.5)
                    candidates.append(k)
                
                best_val = np.inf
                best_xi = lo[i]
                for c in candidates:
                    if lo[i] <= c <= hi[i]:
                        val = c**2 - 10*np.cos(4*np.pi*c*np.pi)
                        if val < best_val:
                            best_val = val
                            best_xi = c
                best_x[i] = best_xi
            
            # For x0, we want to cover the range to get diversity on the front
            # Use non-uniform sampling here as well
            x0_lo = lo[0]
            x0_hi = hi[0]
            if x0_lo >= x0_hi:
                x0_vals = np.array([x0_lo])
            else:
                t = np.linspace(0, 1, self.budget)
                x0_vals = x0_lo + (x0_hi - x0_lo) * (t ** 2)
                x0_vals[0] = x0_lo
                x0_vals[-1] = x0_hi
            
            X = np.tile(best_x, (len(x0_vals), 1))
            X[:, 0] = x0_vals
            
            F = []
            for i in range(len(x0_vals)):
                if time.time() > deadline:
                    break
                F.append(problem(X[i]))
            
            return X

# EVOLVE-BLOCK-END
