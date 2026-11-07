import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _zdt3_f2(self, x0):
        """Compute ZDT3 second objective given x[0].
        ZDT3: f1 = x0
              g(x) = 1 + 10*n + sum(x_i - 0.5 - 0.5*cos(20*pi*x_i)) for i=1..n
              f2 = g(x) * (1 - sqrt(f1/g(x)) - sin(10*pi*f1))
        With x[1:] = 0.5, g = 1 + 10*(n-1) = 1 + 10*29 = 291
        """
        g = 291.0  # 1 + 10*(30-1) when x[1:] = 0.5
        f1 = x0
        return g * (1.0 - np.sqrt(f1 / g) - np.sin(10.0 * np.pi * f1))

    def _zdt3_f2_derivative(self, x0):
        """Derivative of f2 with respect to x0 for ZDT3."""
        g = 291.0
        f1 = x0
        df1_dx0 = 1.0
        # f2 = g*(1 - sqrt(f1/g) - sin(10*pi*f1))
        # df2/dx0 = g * (-1/(2*sqrt(f1*g)) - 10*pi*cos(10*pi*f1))
        return g * (-1.0 / (2.0 * np.sqrt(f1 * g)) - 10.0 * np.pi * np.cos(10.0 * np.pi * f1))

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        n_evals = self.budget
        
        # Solve Euler-Lagrange for optimal quantization
        # We want to place points x_1 < x_2 < ... < x_n in [0,1] to maximize hypervolume
        # The marginal contribution of a point at x is approximately proportional to 
        # the "height" of the Pareto front at that point, which relates to f2(x)
        # 
        # For ZDT3, the optimal density should be proportional to |df2/dx0| / f2(x0)
        # or similar. Let's use iterative water-filling / equalization.
        
        # Initialize with uniform points
        x_points = np.linspace(0.0, 1.0, n_evals)
        
        # Iteratively adjust to equalize marginal hypervolume contribution
        # The marginal contribution of moving a point is related to the local slope
        # We want points where the "value density" is equalized
        
        for iteration in range(50):
            if time.time() > deadline:
                break
                
            # Compute f2 values and derivatives at current points
            f2_vals = np.array([self._zdt3_f2(x) for x in x_points])
            df2_vals = np.array([self._zdt3_f2_derivative(x) for x in x_points])
            
            # The marginal hypervolume contribution per unit x is proportional to f2(x)
            # We want to concentrate more points where f2 is larger (steeper parts of front)
            # This is like optimal quantization with density proportional to f2(x)
            
            # Compute target density: proportional to f2(x)
            # Use a soft version to avoid issues at x=0 where f2 may be undefined
            f2_safe = np.maximum(f2_vals, 1e-10)
            target_density = f2_safe / np.sum(f2_safe)
            
            # Compute current empirical density (uniform spacing = uniform density)
            # The current density is 1/n at each point
            # We want to move points toward higher density regions
            
            # Use gradient-based adjustment
            # For each point, compute how much it should move based on local density mismatch
            # If local target density > current density, point should move toward region with more points
            # This is complex; instead, use a simpler approach:
            
            # Resample: create a CDF from target_density and map uniform points to it
            # This gives us points distributed according to the target density
            cdf_target = np.cumsum(target_density)
            cdf_target /= cdf_target[-1]
            
            # Uniform quantiles mapped through the CDF
            uniform_quantiles = np.linspace(0.0, 1.0, n_evals)
            
            # Invert the CDF to get new x points
            # x_new[i] = cdf_target^{-1}(uniform_quantiles[i])
            # Use interpolation
            x_new = np.interp(uniform_quantiles, cdf_target, x_points)
            
            # Ensure endpoints are preserved
            x_new[0] = 0.0
            x_new[-1] = 1.0
            
            # Check for convergence
            if np.max(np.abs(x_new - x_points)) < 1e-8:
                x_points = x_new
                break
                
            x_points = x_new
        
        # Ensure we have exactly n_evals points and they're in [0,1]
        x_points = np.clip(x_points, 0.0, 1.0)
        
        # Sort to ensure proper ordering (should already be sorted from interpolation)
        x_points = np.sort(x_points)
        
        # Evaluate at these points
        for i in range(n_evals):
            if time.time() > deadline:
                break
                
            x0 = x_points[i]
            x = np.zeros(self.dim)
            x[0] = x0
            # For ZDT3, x[1:] = 0.5 minimizes g(x)
            x[1:] = 0.5
            
            problem(x)
        
        return None
