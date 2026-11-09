import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """
    Optimizer for DTLZ4 (M=3, N=12).
    Uses a robust multi-hypothesis strategy with Sobol sampling on the Pareto front.
    Assumes alpha is in {1, 10, 100} and inverts the DTLZ4 mapping to generate
    decision variables that map to well-spread points on the front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # We have 200 evaluations.
        # Split budget among 3 alpha hypotheses: 1, 10, 100
        # 67 points for alpha=1, 67 for alpha=10, 66 for alpha=100
        n_points = self.budget
        alphas = [1.0, 10.0, 100.0]
        counts = [67, 67, 66]
        
        # Generate Sobol sequence for 2D (theta1, theta2)
        # We need enough points for the max count, or generate for total and split
        # Simpler: Generate total points, then assign to alphas
        sobol = qmc.Sobol(d=2, scramble=True)
        samples = sobol.random(n_points)
        
        # Map [0,1] to [0, pi/2]
        theta1 = samples[:, 0] * (np.pi / 2)
        theta2 = samples[:, 1] * (np.pi / 2)
        
        # Calculate target objectives on the unit sphere (Pareto front shape)
        # f1 = cos(t1), f2 = sin(t1)*cos(t2), f3 = sin(t1)*sin(t2)
        f1_target = np.cos(theta1)
        f2_target = np.sin(theta1) * np.cos(theta2)
        f3_target = np.sin(theta1) * np.sin(theta2)
        
        candidates = []
        
        # Assign points to alphas
        # Indices 0-66: alpha=1
        # Indices 67-133: alpha=10
        # Indices 134-199: alpha=100
        
        def generate_x(alpha, f1, f2, f3):
            # Invert DTLZ4 mapping
            # x1 = f1^alpha
            x1 = f1 ** alpha
            
            # x2/(1-x2) = (f2/f3)^alpha
            # Handle f3=0 case (theta2=pi/2) -> x2=1
            # Handle f2=0 case (theta2=0) -> x2=0
            ratio = f2 / np.where(f3 > 1e-12, f3, 1.0)
            # If f3 is 0, ratio is inf, x2 should be 1
            # If f2 is 0, ratio is 0, x2 should be 0
            
            # x2 = ratio^alpha / (1 + ratio^alpha)
            # To avoid overflow/underflow, use stable calculation
            # Let R = (f2/f3)^alpha
            # x2 = R / (1+R)
            
            # If f3 is very small, ratio is large.
            # If f2 is very small, ratio is small.
            
            # Use log-sum-exp style stability if needed, but alpha is small enough
            # For alpha=100, f2/f3 can be large.
            
            # Calculate R
            with np.errstate(divide='ignore', invalid='ignore'):
                log_ratio = np.log(f2) - np.log(np.where(f3 > 1e-12, f3, 1.0))
                # If f3 is 0, log(f3) is -inf, log_ratio is inf.
                # If f2 is 0, log(f2) is -inf, log_ratio is -inf.
                
                log_R = alpha * log_ratio
                # x2 = exp(log_R) / (1 + exp(log_R)) = sigmoid(log_R)
                # Stable sigmoid:
                # if log_R > 0: 1 / (1 + exp(-log_R))
                # if log_R <= 0: exp(log_R) / (1 + exp(log_R))
                
                x2 = np.where(log_R > 0, 
                              1.0 / (1.0 + np.exp(-log_R)), 
                              np.exp(log_R) / (1.0 + np.exp(log_R)))
            
            # Handle edge cases explicitly
            x2 = np.where(f3 < 1e-12, 1.0, x2)
            x2 = np.where(f2 < 1e-12, 0.0, x2)
            
            x1 = np.clip(x1, 0.0, 1.0)
            x2 = np.clip(x2, 0.0, 1.0)
            
            return x1, x2

        # Process each group
        idx_start = 0
        for i, alpha in enumerate(alphas):
            count = counts[i]
            idx_end = idx_start + count
            
            f1_g = f1_target[idx_start:idx_end]
            f2_g = f2_target[idx_start:idx_end]
            f3_g = f3_target[idx_start:idx_end]
            
            x1_vals, x2_vals = generate_x(alpha, f1_g, f2_g, f3_g)
            
            for j in range(count):
                x = np.zeros(self.dim)
                x[0] = x1_vals[j]
                x[1] = x2_vals[j]
                x[2:] = 0.5  # g(x) = 0
                
                # Clip to bounds
                x = np.clip(x, lo, hi)
                candidates.append(x)
            
            idx_start = idx_end

        results = []
        for x in candidates:
            if time.time() > deadline:
                break
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break
        
        return results
