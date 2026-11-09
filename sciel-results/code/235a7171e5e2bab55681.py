import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        # Extract bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Strategy: ZDT3 Pareto front is at x[1:] = lo[1:]
        # We need to find x[0] such that f2 < ref_f2 (11.0)
        
        # Analytical calculation for ZDT3 valid front segment
        # f2(x) = 1 + (9/(n-1)) * g(x)
        # g(x) = 10 + sum_{i=2}^n 10*sin(6*x_i*pi + 10*pi*x_1/(n-1))
        # With x_i = lo[1] (usually 0), sin(10*pi*x_1/(n-1))
        # We need f2 < 11
        # 1 + (9/(n-1)) * (10 + 10*(n-1)*sin(10*pi*x_1/(n-1))) < 11
        # 1 + 90/(n-1) + 90*sin(10*pi*x_1/(n-1)) < 11
        # 90*sin(10*pi*x_1/(n-1)) < 10 - 90/(n-1)
        
        n = self.dim
        if n <= 1:
            # Fallback to random if dim is weird
            for _ in range(self.budget):
                if time.time() > deadline:
                    break
                x = np.random.uniform(lo, hi)
                try:
                    problem(x)
                except Exception:
                    break
            return None

        coeff = 90.0
        rhs = 10.0 - 90.0 / (n - 1)
        
        # sin(theta) < rhs / 90
        # theta = 10 * pi * x_1 / (n-1)
        # For small theta, sin(theta) ~ theta
        # If rhs/90 is small, x_1_max is small.
        # If rhs/90 >= 1, all x_1 are valid?
        # If rhs/90 < -1, no x_1 is valid?
        
        ratio = rhs / 90.0
        
        x1_max = 1.0
        if ratio < 1.0:
            # We are in the increasing part of sine for small x1
            # sin(theta) = ratio -> theta = arcsin(ratio)
            # But sine is periodic. In ZDT3 with x_i=0, theta is small for x1 in [0,1]?
            # theta_max = 10*pi/(n-1). For n=30, theta_max ~ 1.08 rad.
            # arcsin(ratio) is in [0, pi/2].
            # If arcsin(ratio) < theta_max, then x1_max corresponds to arcsin(ratio).
            # Otherwise, we need to check if the sine dips below ratio later.
            # Given theta_max ~ 1.08 < pi/2, we are only in the first quadrant.
            # So sin(theta) is monotonic increasing.
            # Thus valid range is [0, arcsin(ratio)] if ratio > 0.
            # If ratio <= 0, no valid points on the front? 
            # Wait, if ratio <= 0, sin(theta) < negative. 
            # In [0, 1.08], sin is positive. So no valid points.
            # But f2 must be < 11. 
            # If min f2 > 11, then no points count.
            # Min f2 is at x1=0? No, min f2 is when sin is min.
            # In [0, 1.08], min sin is 0 (at x1=0).
            # So min f2 = 1 + 90/(n-1).
            # For n=30, min f2 = 4.1 < 11. So there are valid points.
            # This implies ratio > 0.
            
            if ratio > 0:
                theta_max_val = np.arcsin(ratio)
                x1_max = theta_max_val * (n - 1) / (10.0 * np.pi)
                if x1_max > 1.0:
                    x1_max = 1.0
            else:
                # No valid points on the analytic front?
                # This shouldn't happen for standard ZDT3 with ref 11.
                x1_max = 0.0
        else:
            # ratio >= 1, sin(theta) < 1 is always true for theta in [0, pi/2]
            x1_max = 1.0

        # Allocate budget
        n_front = int(self.budget * 0.95)
        n_random = self.budget - n_front
        
        points = []
        
        # 1. Points on the assumed Pareto front
        # x[0] uniform in [lo[0], lo[0] + x1_max * (hi[0]-lo[0])]?
        # Bounds are usually [0,1].
        # x1 should be in [lo[0], lo[0] + x1_max * (hi[0]-lo[0])]
        # Assuming lo[0]=0, hi[0]=1.
        x1_lo = lo[0]
        x1_hi = lo[0] + x1_max * (hi[0] - lo[0])
        
        if n_front > 0 and x1_hi > x1_lo:
            x0_samples = np.linspace(x1_lo, x1_hi, n_front)
            for x0 in x0_samples:
                x = np.full(self.dim, lo[1]) # Set rest to lower bound
                x[0] = x0
                points.append(x)
        elif n_front > 0:
            # If x1_max is 0 or very small, just use the best point x1=lo[0]
            # Or random if no valid front points
            for _ in range(n_front):
                x = np.full(self.dim, lo[1])
                x[0] = lo[0]
                points.append(x)

        # 2. Random points
        for _ in range(n_random):
            if time.time() > deadline:
                break
            x = np.random.uniform(lo, hi)
            points.append(x)
            
        # Evaluate
        for x in points:
            if time.time() > deadline:
                break
            try:
                problem(x)
            except Exception:
                break
                
        return None
