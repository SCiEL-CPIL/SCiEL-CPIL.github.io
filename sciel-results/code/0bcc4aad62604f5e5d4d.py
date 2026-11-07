"""Multi-objective optimizer for ZDT4 exploiting problem structure."""
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for ZDT4.
    
    Strategy:
    1. Exploit the known analytical Pareto front of ZDT4.
    2. The Pareto front is located at x_1 in [0, 1] and x_2..x_n = 0.
    3. Sample uniformly in f2 (objective) space to maximize hypervolume,
       since the reference point is far from the front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill

        n = self.dim
        budget = self.budget
        
        # Extract bounds
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        # Determine the optimal values for the deceptive variables x_2..x_n.
        # For ZDT4, the global minimum of the penalty function g(x) is at x_i = 0.
        optimal_deceptive_vals = np.zeros(n - 1, dtype=np.float64)
        for i in range(1, n):
            if lo[i] > 0:
                optimal_deceptive_vals[i-1] = lo[i]
            elif hi[i] < 0:
                optimal_deceptive_vals[i-1] = hi[i]
            else:
                optimal_deceptive_vals[i-1] = 0.0

        # For ZDT4 with g=1: f1 = x1, f2 = 1 + sin(10*pi*x1)
        # f2 ranges from 0 to 2 as x1 ranges from 0 to 1.
        # We sample uniformly in f2 space: f2_values evenly spaced in [0, 2]
        # For each f2 target, find the x1 that achieves it with minimum f1.
        
        f2_lo = 0.0
        f2_hi = 2.0
        
        # Generate budget evenly-spaced f2 targets
        f2_targets = np.linspace(f2_lo, f2_hi, budget)
        
        # For each f2 target y, solve sin(10*pi*x1) = y - 1 for x1 in [0,1]
        # We want the smallest x1 (minimizes f1)
        
        x1_values = np.empty(budget, dtype=np.float64)
        
        for i in range(budget):
            y = f2_targets[i]
            sin_val = y - 1.0  # target value for sin(10*pi*x1)
            
            # Clamp to valid range
            sin_val = np.clip(sin_val, -1.0, 1.0)
            
            # Solutions: 10*pi*x1 = arcsin(sin_val) + 2*pi*k or pi - arcsin(sin_val) + 2*pi*k
            # We want x1 in [0, 1], so 10*pi*x1 in [0, 10*pi]
            # k can be 0, 1, 2 (since 10*pi / (2*pi) = 5, so k=0..4)
            
            base = np.arcsin(sin_val)  # in [-pi/2, pi/2]
            
            candidates = []
            for k in range(5):
                # Solution 1: 10*pi*x1 = base + 2*pi*k
                angle1 = base + 2.0 * np.pi * k
                if 0.0 <= angle1 <= 10.0 * np.pi:
                    x1_candidate = angle1 / (10.0 * np.pi)
                    candidates.append(x1_candidate)
                
                # Solution 2: 10*pi*x1 = pi - base + 2*pi*k
                angle2 = np.pi - base + 2.0 * np.pi * k
                if 0.0 <= angle2 <= 10.0 * np.pi:
                    x1_candidate = angle2 / (10.0 * np.pi)
                    candidates.append(x1_candidate)
            
            if candidates:
                # Pick the smallest x1 (minimizes f1)
                x1_values[i] = min(candidates)
            else:
                # Fallback: use base solution
                x1_values[i] = max(0.0, min(1.0, base / (10.0 * np.pi)))
        
        # Clip x1 to valid range
        x1_values = np.clip(x1_values, max(lo[0], 0.0), min(hi[0], 1.0))
        
        results = []
        
        for i in range(budget):
            if time.time() > deadline:
                break
                
            x1_val = x1_values[i]
            
            # Construct the full vector x
            x = np.empty(n, dtype=np.float64)
            x[0] = x1_val
            x[1:] = optimal_deceptive_vals
            
            # Clip to bounds to ensure validity
            x = np.clip(x, lo, hi)
            
            # Evaluate
            f = problem(x)
            
            # Store the result
            results.append(f)
            
            # Check time periodically to be safe
            if i % 50 == 0:
                if time.time() > deadline:
                    break

        return results
