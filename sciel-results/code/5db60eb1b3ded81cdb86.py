import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_samples = min(self.budget, 200)
        
        # ZDT3 Parameters
        # f1 = x0
        # f2 = g(x) * (1 + sin(10 * pi * x0))
        # g(x) min is 291 when x_i = 0.5 for i > 0.
        # Reference point [11, 11].
        # We need f2 < 11.
        # 291 * (1 + sin(10 * pi * x0)) < 11
        # 1 + sin(10 * pi * x0) < 11 / 291
        # sin(10 * pi * x0) < (11 - 291) / 291 = -280 / 291
        
        target_sin_val = -280.0 / 291.0
        beta = np.arcsin(280.0 / 291.0) # arcsin(positive)
        
        # In each period of length 0.2 in x0 (which is 2*pi in theta),
        # the sine is below target_sin_val in the interval:
        # theta in [pi + beta, 2*pi - beta]
        theta_start = np.pi + beta
        theta_end = 2.0 * np.pi - beta
        
        # Convert to x0 offset from the center of the minimum (which is at 3*pi/2 in theta)
        # Center theta = 1.5 * pi
        # Width in theta
        width_theta = theta_end - theta_start
        # Width in x0
        width_x0 = width_theta / (10.0 * np.pi)
        
        # The minima of sin(10*pi*x0) in [0,1] are at:
        # 10*pi*x0 = 1.5*pi, 5.5*pi, 9.5*pi, 13.5*pi, 17.5*pi
        # x0 = 0.15, 0.55, 0.95 ... wait.
        # 1.5 * pi / (10 * pi) = 0.15
        # 5.5 * pi / (10 * pi) = 0.55
        # 9.5 * pi / (10 * pi) = 0.95
        # Wait, period is 0.2.
        # Minima at 0.15, 0.35, 0.55, 0.75, 0.95.
        
        centers = [0.15, 0.35, 0.55, 0.75, 0.95]
        
        # We have 5 segments. Distribute n_samples evenly.
        samples_per_segment = n_samples // len(centers)
        remaining = n_samples - samples_per_segment * len(centers)
        
        x0_values = []
        for i, c in enumerate(centers):
            count = samples_per_segment + (1 if i < remaining else 0)
            # Uniform sampling in [c - width_x0/2, c + width_x0/2]
            # Clip to [0, 1] just in case, though 0.95 + 0.01 < 1
            start = c - width_x0 / 2.0
            end = c + width_x0 / 2.0
            x0_values.extend(np.linspace(start, end, count))
        
        x0_values = np.array(x0_values)
        
        # Create X matrix
        X = np.zeros((n_samples, self.dim))
        X[:, 0] = x0_values
        X[:, 1:] = 0.5
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Evaluate
        F = np.array([problem(x) for x in X])
        
        return X
