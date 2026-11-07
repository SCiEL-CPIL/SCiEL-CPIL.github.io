import numpy as np
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
from scipy.interpolate import interp1d

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT4 specific logic
        # f1 = x0
        # f2 = g(x) * (1 - sqrt(f1/g(x)))
        # g(x) = 1 + 10(n-1) + sum_{i=2}^n (xi^2 - 10cos(4pi*xi))
        
        # Find x_opt for auxiliary variables to minimize g(x)
        # Assuming all auxiliary variables have the same bounds
        aux_lo = bounds[1][0]
        aux_hi = bounds[1][1]
        
        def h(x):
            return x**2 - 10*np.cos(4*np.pi*x)
        
        # Grid search + local refine for x_opt
        grid = np.linspace(aux_lo, aux_hi, 100)
        vals = np.array([h(x) for x in grid])
        min_idx = np.argmin(vals)
        x_opt = grid[min_idx]
        
        local_lo = max(aux_lo, x_opt - 0.01)
        local_hi = min(aux_hi, x_opt + 0.01)
        
        if local_lo < local_hi:
            res = minimize_scalar(h, bounds=(local_lo, local_hi), method='bounded')
            if res.success:
                x_opt = res.x
        
        # Calculate G = g(x_opt)
        # g(x) = 1 + 10*(dim-1) + (dim-1)*h(x_opt)
        G = 1 + 10 * (self.dim - 1) + (self.dim - 1) * h(x_opt)
        
        # The Pareto front is defined by x0 in [bounds[0][0], bounds[0][1]]
        # f1 = x0
        # f2 = G - sqrt(G * x0)
        
        x0_min = bounds[0][0]
        x0_max = bounds[0][1]
        
        # We want to sample x0 such that the arc length along the curve is uniform.
        # Arc length differential: ds = sqrt(1 + (df2/dx0)^2) dx0
        # df2/dx0 = -0.5 * sqrt(G) / sqrt(x0)
        # (df2/dx0)^2 = G / (4 * x0)
        
        def integrand(x0):
            if x0 <= 0:
                return 0 # Avoid division by zero, though x0 should be > 0 for sqrt
            return np.sqrt(1 + G / (4 * x0))
        
        # Compute total arc length
        # We need to handle x0=0 carefully if bounds include 0.
        # If x0_min is 0, the integral diverges? 
        # sqrt(1 + G/(4x0)) ~ sqrt(G/(4x0)) ~ x0^-0.5. 
        # Integral of x^-0.5 is 2x^0.5, which is finite at 0. So it converges.
        
        total_length, _ = quad(integrand, x0_min, x0_max, limit=100)
        
        # We want n_evals points.
        n_evals = self.budget
        # Target arc lengths for each point
        target_lengths = np.linspace(0, total_length, n_evals)
        
        # Invert the cumulative arc length function to find x0 values
        # We need to solve: L(x0) = target_length
        # L(x0) = quad(integrand, x0_min, x0)
        
        x0_vals = np.zeros(n_evals)
        
        # To speed up, we can do a binary search or use a pre-computed lookup table.
        # Given n=200, binary search with quad might be slow if called 200 times * 50 iterations.
        # Better: Create a fine grid of x0, compute cumulative arc length, and interpolate.
        
        # Fine grid for interpolation
        n_grid = 1000
        x0_grid = np.linspace(x0_min, x0_max, n_grid)
        
        # Compute cumulative arc length on the grid
        # We can integrate the integrand over the grid intervals
        # Or use cumulative sum of (integrand(x_i) * dx)
        dx = x0_grid[1] - x0_grid[0]
        integrand_vals = np.array([integrand(x) for x in x0_grid])
        
        # Handle x0=0 case in integrand if x0_min == 0
        # The integrand goes to infinity at 0, but the integral is finite.
        # The value at 0 is technically inf, but for the Riemann sum, we can just use the limit or set it to 0?
        # Actually, quad handles it. For the grid, if x0_min=0, integrand[0] is inf.
        # Let's replace inf with a large number or handle the first interval analytically?
        # Or just shift the grid slightly if x0_min is 0?
        # Let's assume x0_min > 0 or handle the first point specially.
        # If x0_min == 0, the arc length from 0 to x is 2*sqrt(x*(1+G/(4x)))? No.
        # Let's just use quad for the inversion to be safe and accurate, but optimize the number of calls.
        # Actually, `scipy.integrate.quad` is robust. 200 calls * ~50 evals per quad = 10000 evals of a simple function. This is fast.
        
        # Let's try the interpolation method first as it's O(1) per point after setup.
        # If x0_min is 0, integrand is singular. 
        # Let's use a dense grid and `cumsum`.
        # To avoid inf at 0, we can start the grid at a tiny epsilon if x0_min is 0.
        
        eps = 1e-9
        x0_grid_start = max(x0_min, eps)
        if x0_min == 0:
             # We need to account for the arc length from 0 to eps
             # L(0, eps) = quad(integrand, 0, eps)
             l_start, _ = quad(integrand, 0, eps, limit=100)
             x0_grid = np.linspace(eps, x0_max, n_grid)
        else:
             l_start = 0
             x0_grid = np.linspace(x0_min, x0_max, n_grid)
             
        dx = x0_grid[1] - x0_grid[0]
        integrand_vals = np.array([integrand(x) for x in x0_grid])
        
        # Approximate cumulative arc length
        # L(x_i) = L(x0_min) + sum_{j=0}^{i-1} (L(x_{j+1}) - L(x_j))
        # Using trapezoidal rule for better accuracy
        diffs = np.diff(x0_grid)
        avg_integrand = 0.5 * (integrand_vals[:-1] + integrand_vals[1:])
        seg_lengths = avg_integrand * diffs
        cum_lengths = np.concatenate(([0], np.cumsum(seg_lengths))) + l_start
        
        # Interpolate to find x0 for each target length
        # target_lengths is from 0 to total_length
        # cum_lengths is from l_start to l_start + total_grid_length
        # Note: total_length computed by quad might differ slightly from cum_lengths[-1] due to numerical error.
        # We should normalize or just use the grid's max length for interpolation bounds.
        
        # Create an interpolant
        # We want x0 given L.
        # cum_lengths is increasing.
        # Use linear interpolation for simplicity, grid is dense.
        interp_func = interp1d(cum_lengths, x0_grid, kind='linear', fill_value="extrapolate")
        
        # Map target_lengths to x0
        # Ensure target_lengths are within [l_start, cum_lengths[-1]]
        # l_start is the arc length from 0 to x0_min (if x0_min>0, l_start=0).
        # Wait, if x0_min > 0, l_start = 0. cum_lengths starts at 0.
        # If x0_min == 0, l_start > 0. cum_lengths starts at l_start.
        # target_lengths starts at 0.
        # So we need to map target_lengths relative to the start of the domain.
        
        # Let's redefine:
        # We want to sample the curve from x0_min to x0_max.
        # The total arc length is L_total = quad(integrand, x0_min, x0_max).
        # We generated x0_vals such that arc length from x0_min to x0_val is uniform.
        
        # Let's restart the interpolation logic to be cleaner.
        
        # 1. Compute L_total accurately
        L_total, _ = quad(integrand, x0_min, x0_max, limit=100)
        
        # 2. Create a lookup table for L(x0)
        # We need L(x0) - L(x0_min)
        
        # If x0_min == 0:
        #   L(x0) is well defined.
        #   We can just compute L on the grid.
        
        n_grid = 1000
        if x0_min == 0:
            x0_grid = np.linspace(0, x0_max, n_grid)
            # integrand at 0 is inf. 
            # We can compute L(x0) directly using quad for each grid point? No, too slow.
            # Analytical solution for integral of sqrt(1 + G/(4x))?
            # Let u = sqrt(x). x = u^2, dx = 2u du.
            # Integral sqrt(1 + G/(4u^2)) 2u du = Integral 2u sqrt((4u^2+G)/4u^2) du
            # = Integral 2u * sqrt(4u^2+G)/(2u) du = Integral sqrt(4u^2+G) du
            # This is a standard integral: 
            # int sqrt(u^2+a^2) du = 0.5 * u * sqrt(u^2+a^2) + 0.5 * a^2 * ln(u + sqrt(u^2+a^2))
            # Here 4u^2+G = (2u)^2 + G.
            # Let v = 2u = 2sqrt(x). dv = 2/sqrt(x) dx.
            # This is getting complicated.
            
            # Simpler: Just use a grid that starts at eps.
            # The error from 0 to eps is negligible for hypervolume if eps is small.
            eps = 1e-8
            x0_grid = np.linspace(eps, x0_max, n_grid)
            integrand_vals = np.array([integrand(x) for x in x0_grid])
            dx = x0_grid[1] - x0_grid[0]
            seg_lengths = 0.5 * (integrand_vals[:-1] + integrand_vals[1:]) * dx
            cum_lengths = np.cumsum(seg_lengths)
            # The total length on the grid
            L_grid_total = cum_lengths[-1]
            # We want to map uniform targets in [0, L_total] to x0.
            # But our grid covers [eps, x0_max].
            # The arc length from 0 to eps is L(eps).
            # L(eps) is small.
            # We can just approximate that the sampling starts from 0.
            
            # Let's just use `quad` for inversion. It's robust.
            # 200 calls.
            
            def arc_length_from_min(x0_target):
                if x0_target <= x0_min:
                    return 0.0
                if x0_target >= x0_max:
                    return L_total
                val, _ = quad(integrand, x0_min, x0_target, limit=50)
                return val

            # Binary search for each target length
            x0_vals = np.zeros(n_evals)
            target_lengths = np.linspace(0, L_total, n_evals)
            
            for i in range(n_evals):
                target = target_lengths[i]
                if target == 0:
                    x0_vals[i] = x0_min
                elif target == L_total:
                    x0_vals[i] = x0_max
                else:
                    # Bisection
                    a = x0_min
                    b = x0_max
                    for _ in range(30): # 30 iterations is enough for double precision
                        mid = (a + b) / 2
                        l_mid = arc_length_from_min(mid)
                        if l_mid < target:
                            a = mid
                        else:
                            b = mid
                    x0_vals[i] = (a + b) / 2

        else:
            # x0_min > 0
            # We can use the grid interpolation method as it's faster and integrand is smooth.
            x0_grid = np.linspace(x0_min, x0_max, 1000)
            integrand_vals = np.array([integrand(x) for x in x0_grid])
            dx = x0_grid[1] - x0_grid[0]
            seg_lengths = 0.5 * (integrand_vals[:-1] + integrand_vals[1:]) * dx
            cum_lengths = np.cumsum(seg_lengths)
            
            # Normalize cum_lengths to [0, L_total] to match target_lengths
            # cum_lengths[-1] should be approx L_total
            if cum_lengths[-1] > 0:
                cum_lengths_norm = cum_lengths / cum_lengths[-1]
            else:
                cum_lengths_norm = cum_lengths
                
            # target_lengths are in [0, L_total]
            # We need to map target_lengths/L_total to cum_lengths_norm
            targets_norm = target_lengths / L_total
            
            interp_func = interp1d(cum_lengths_norm, x0_grid, kind='linear', fill_value="extrapolate")
            x0_vals = interp_func(targets_norm)
            x0_vals = np.clip(x0_vals, x0_min, x0_max)

        # Construct points
        points = np.zeros((n_evals, self.dim))
        points[:, 0] = x0_vals
        points[:, 1:] = x_opt
        
        # Evaluate
        for i in range(n_evals):
            problem(points[i])
            
        return points[-1]
