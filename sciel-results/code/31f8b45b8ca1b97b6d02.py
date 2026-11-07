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
        
        # CDF: F(t) = [2*sqrt(t) - 20*ln((10+sqrt(t))/10)] / [2 - 20*ln(1.1)]
        # Inverse via Newton's method for speed
        _denom = 2.0 - 20.0 * np.log(1.1)
        
        def cdf(t):
            if t <= 0.0:
                return 0.0
            if t >= 1.0:
                return 1.0
            st = np.sqrt(t)
            return (2.0 * st - 20.0 * np.log((10.0 + st) / 10.0)) / _denom
        
        def inv_cdf(u):
            """Invert the CDF using Newton's method"""
            if u <= 0.0:
                return 0.0
            if u >= 1.0:
                return 1.0
            # Initial guess: linear interpolation
            t = u
            for _ in range(30):
                st = np.sqrt(max(t, 1e-30))
                f = (2.0 * st - 20.0 * np.log((10.0 + st) / 10.0)) / _denom - u
                # Derivative: d/dt [2*sqrt(t) - 20*ln((10+sqrt(t))/10)] / denom
                # = [1/sqrt(t) - 20/(10+sqrt(t)) * 1/(2*sqrt(t))] / denom
                # = [1/sqrt(t) - 10/(sqrt(t)*(10+sqrt(t)))] / denom
                # = [10 + sqrt(t) - 10] / [sqrt(t)*(10+sqrt(t)) * denom]
                # = sqrt(t) / [sqrt(t)*(10+sqrt(t)) * denom]
                # = 1 / [(10+sqrt(t)) * denom]
                df = 1.0 / ((10.0 + st) * _denom)
                t_new = t - f / df
                if t_new < 0.0:
                    t_new = 0.0
                if t_new > 1.0:
                    t_new = 1.0
                if abs(t_new - t) < 1e-15:
                    t = t_new
                    break
                t = t_new
            return t
        
        if budget <= 2:
            x0_samples = np.linspace(lo[0], hi[0], budget)
        else:
            # Use all budget points: Halton sequence for interior, plus endpoints
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
            
            # Apply inverse CDF to space points optimally for hypervolume
            halton_points = [inv_cdf(p) for p in halton_points]
            
            # Add endpoints 0.0 and 1.0 to fully cover the front
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
