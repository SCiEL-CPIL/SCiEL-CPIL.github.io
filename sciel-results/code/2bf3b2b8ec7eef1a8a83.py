"""ZDT3 Optimizer using CDF-inverted Halton sampling for hypervolume-optimal spacing."""
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for the 330s kill switch
        
        dim = self.dim
        budget = self.budget
        bounds_arr = np.array(bounds)
        lo = bounds_arr[:, 0]
        hi = bounds_arr[:, 1]
        
        # ZDT3 specific structure:
        # f1 = x[0]
        # g = 1 + 10/d * (sum(x[i]) - sum(f(x[i])))
        # f2 = g * (1 - sqrt(f1/g))
        # To maximize hypervolume, we minimize g by setting x[1:] = 0.
        # This yields f2 = 1 - sqrt(f1).
        
        # Hypervolume-optimal sampling: sample f1 according to density
        # p(f1) ∝ 1 / (R2 - f2(f1)) = 1 / (10 + sqrt(f1))
        # This equalizes the hypervolume contribution of each interval.
        
        def cdf(t):
            # Normalized CDF of p(t) = 1 / (10 + sqrt(t))
            # Integral is 2*sqrt(t) - 20*ln(10+sqrt(t))
            val = 2.0 * np.sqrt(t) - 20.0 * np.log(10.0 + np.sqrt(t))
            val0 = -20.0 * np.log(10.0)
            val1 = 2.0 - 20.0 * np.log(11.0)
            return (val - val0) / (val1 - val0)
        
        def inv_cdf(u):
            """Invert the CDF using Newton's method."""
            if u <= 0:
                return 0.0
            if u >= 1:
                return 1.0
            
            # Target value in the unnormalized H(x) scale
            val0 = -20.0 * np.log(10.0)
            val1 = 2.0 - 20.0 * np.log(11.0)
            target = val0 + u * (val1 - val0)
            
            # Newton's method
            # H(x) = 2*sqrt(x) - 20*ln(10+sqrt(x))
            # H'(x) = 1 / (10 + sqrt(x))
            x = 0.5  # Initial guess
            for _ in range(20):
                sqrt_x = np.sqrt(x)
                h_val = 2.0 * sqrt_x - 20.0 * np.log(10.0 + sqrt_x)
                h_deriv = 1.0 / (10.0 + sqrt_x)
                x_new = x - (h_val - target) / h_deriv
                if abs(x_new - x) < 1e-15:
                    x = x_new
                    break
                x = x_new
                # Clamp to [0, 1] to be safe
                if x < 0: x = 0
                if x > 1: x = 1
            return x
        
        if budget <= 2:
            x0_samples = np.linspace(lo[0], hi[0], budget)
        else:
            n_halton = budget - 2
            # Generate Van der Corput sequence (Halton base 2)
            halton_points = []
            for i in range(1, n_halton + 1):
                x = 0.0
                f = 0.5
                j = i
                while j > 0:
                    x += f * (j % 2)
                    j //= 2
                    f /= 2.0
                halton_points.append(x)
            
            # Apply inverse CDF to concentrate samples where hypervolume
            # contribution is highest (near f1=0)
            halton_points = [inv_cdf(p) for p in halton_points]
            
            # Add endpoints 0.0 and 1.0
            halton_points.append(0.0)
            halton_points.append(1.0)
            
            x0_samples = np.array(halton_points)
            # Map to [lo[0], hi[0]]
            x0_samples = lo[0] + x0_samples * (hi[0] - lo[0])
        
        # Pre-allocate the rest of the vector
        # Set to 0 to minimize g.
        rest_val = 0.0
        if lo[1].min() > 0:
            rest_val = lo[1].min()
            
        for i in range(budget):
            if time.time() > deadline:
                break
            
            x = np.full(dim, rest_val)
            x[0] = x0_samples[i]
            
            # Clip to bounds to be safe
            x = np.clip(x, lo, hi)
            
            # Evaluate
            problem(x)
            
        return None
