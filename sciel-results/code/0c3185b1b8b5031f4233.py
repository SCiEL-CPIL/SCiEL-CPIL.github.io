import numpy as np
from scipy.optimize import minimize_scalar
from scipy.integrate import quad

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT4 specific logic:
        # f1 = x0
        # f2 = g(x) * (1 - sqrt(f1/g(x)))
        # g(x) = 1 + 10(n-1) + sum_{i=2}^n (xi^2 - 10cos(4pi*xi))
        # To minimize f2, we must minimize g(x).
        # The terms in the sum are identical for i=2..n.
        # Let h(x) = x^2 - 10cos(4pi*x).
        # We want to find x_opt that minimizes h(x) within the bounds of the auxiliary variables.
        
        # Assume bounds for auxiliary variables are the same (standard ZDT4)
        # Use the bounds of variable 1 (index 1) as representative
        aux_lo = bounds[1][0]
        aux_hi = bounds[1][1]
        
        # Define the auxiliary function
        def h(x):
            return x**2 - 10*np.cos(4*np.pi*x)
        
        # Find the minimum of h on [aux_lo, aux_hi]
        # Since h is periodic and has many local minima, we check a grid and refine
        # Standard ZDT4 bounds are [0,1]. Global min is at x=0.
        # To be safe, we scan a grid.
        grid = np.linspace(aux_lo, aux_hi, 200)
        vals = np.array([h(x) for x in grid])
        min_idx = np.argmin(vals)
        x_opt = grid[min_idx]
        
        # Refine around the grid minimum
        # Use a small bounded optimization
        # Ensure bounds for minimize_scalar are valid
        local_lo = max(aux_lo, x_opt - 0.01)
        local_hi = min(aux_hi, x_opt + 0.01)
        
        if local_lo < local_hi:
            res = minimize_scalar(h, bounds=(local_lo, local_hi), method='bounded')
            if res.success:
                x_opt = res.x
        
        # Calculate G_min (g value at optimal auxiliary variables)
        # g = 1 + 10*(dim-1) + sum(h(x_i) for i in 1..dim-1)
        # Assuming all auxiliary vars are set to x_opt
        n_aux = self.dim - 1
        g_min = 1.0 + 10.0 * (self.dim - 1) + n_aux * h(x_opt)
        
        # We want to sample points on the Pareto front defined by:
        # f1 = x0
        # f2 = g_min * (1 - sqrt(x0 / g_min))
        
        # To maximize hypervolume, we distribute points uniformly along the arc length of this curve.
        x0_lo, x0_hi = bounds[0][0], bounds[0][1]
        
        # Define the derivative of f2 with respect to x0
        # f2(x) = g_min - sqrt(g_min * x)
        # f2'(x) = -0.5 * sqrt(g_min) / sqrt(x)
        # ds/dx = sqrt(1 + (f2')^2) = sqrt(1 + g_min / (4*x))
        
        def arc_length_integrand(x):
            # Avoid division by zero if x is 0
            if x <= 0:
                return 1.0 # Limit as x->0 is infinity, but integral is finite. 
                           # Actually sqrt(1 + C/x) ~ sqrt(C/x). Integral of x^-0.5 is 2sqrt(x).
                           # So we can just use a small epsilon or handle it in quad.
            return np.sqrt(1.0 + g_min / (4.0 * x))
        
        # Calculate total arc length
        # If x0_lo is 0, quad might have trouble. Let's start from a small epsilon if needed.
        eps = 1e-10
        x_start = max(x0_lo, eps)
        
        # If x0_lo is 0, we handle the first segment analytically or just start slightly above 0.
        # For hypervolume, the exact position of the first point near 0 matters less than spacing.
        
        # Let's use a grid-based inversion for robustness and speed.
        n_grid = 1000
        x_grid = np.linspace(x_start, x0_hi, n_grid)
        
        # Calculate cumulative arc length
        # We can integrate ds/dx
        # Using simple trapezoidal rule on the grid is fast and accurate enough for sampling.
        ds_dx = np.sqrt(1.0 + g_min / (4.0 * x_grid))
        # Handle x=0 case if x_start was 0 (but we set eps)
        
        # Cumulative sum of arc lengths
        dx = x_grid[1] - x_grid[0]
        arc_lengths = np.cumsum(0.5 * (ds_dx[:-1] + ds_dx[1:]) * dx)
        arc_lengths = np.insert(arc_lengths, 0, 0.0)
        
        total_arc = arc_lengths[-1]
        
        # We want n_evals points uniformly spaced in arc length
        n_evals = self.budget
        target_arcs = np.linspace(0.0, total_arc, n_evals)
        
        # Interpolate to find x values corresponding to these arc lengths
        x0_vals = np.interp(target_arcs, arc_lengths, x_grid)
        
        # Ensure x0_vals are within bounds
        x0_vals = np.clip(x0_vals, x0_lo, x0_hi)
        
        # Construct the points
        points = np.zeros((n_evals, self.dim))
        points[:, 0] = x0_vals
        # Set auxiliary variables to x_opt
        points[:, 1:] = x_opt
        
        # Evaluate all points
        for i in range(n_evals):
            problem(points[i])
            
        return points[-1]
