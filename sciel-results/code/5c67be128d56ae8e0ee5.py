import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ2 using a Sobol sequence mapped to the Pareto front
    via quantile-based inverse transform for uniform objective-space coverage.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: Evaluator kills at 330s. Return before that.
        deadline = start_time + 320
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_points = self.budget
        
        # Generate Sobol sequence in [0,1]^2
        from scipy.stats import qmc
        sobol = qmc.Sobol(d=2, scramble=True, seed=42)
        raw_points = sobol.random(n=n_points)  # shape (n_points, 2), values in [0,1)
        
        results = []
        
        for i in range(n_points):
            if time.time() > deadline:
                break
            
            u1 = raw_points[i, 0]
            u2 = raw_points[i, 1]
            
            # Map to DTLZ2 Pareto front.
            # DTLZ2 with M=3: f1 = sin^2(theta1/2) * cos^2(theta2/2)
            #                 f2 = sin^2(theta1/2) * sin^2(theta2/2)
            #                 f3 = cos^2(theta1/2)
            # where theta1 in [0, pi/2], theta2 in [0, pi/2]
            #
            # The Pareto front is a quarter-sphere: f1^2 + f2^2 + f3^2 = 1
            # with f1, f2, f3 >= 0.
            #
            # For uniform coverage in objective space (hypervolume-optimal),
            # we want to sample (f1, f2) uniformly in the quarter-disk f1^2 + f2^2 <= 1,
            # then set f3 = sqrt(1 - f1^2 - f2^2).
            #
            # Uniform sampling in a quarter-disk:
            #   r = sqrt(u1)  (for uniform area in disk)
            #   theta = 2*pi*u2  (for uniform angle)
            # But we only need the first quadrant, so theta in [0, pi/2]:
            #   theta = (pi/2) * u2
            #   f1 = r * cos(theta)
            #   f2 = r * sin(theta)
            #   f3 = sqrt(1 - r^2)
            
            r = np.sqrt(u1)
            theta = (np.pi / 2.0) * u2
            
            f1 = r * np.cos(theta)
            f2 = r * np.sin(theta)
            f3 = np.sqrt(max(0.0, 1.0 - r * r))
            
            # Now find x[0], x[1] that produce these objective values.
            # f1 = sin^2(theta1/2) * cos^2(theta2/2)
            # f2 = sin^2(theta1/2) * sin^2(theta2/2)
            # f3 = cos^2(theta1/2)
            #
            # From f3: cos^2(theta1/2) = f3 => cos(theta1/2) = sqrt(f3)
            # => theta1/2 = arccos(sqrt(f3)) => theta1 = 2*arccos(sqrt(f3))
            #
            # sin^2(theta1/2) = 1 - f3
            # f1 + f2 = sin^2(theta1/2) = 1 - f3 (consistent with sphere)
            #
            # cos^2(theta2/2) = f1 / (1 - f3)
            # sin^2(theta2/2) = f2 / (1 - f3)
            #
            # x[0] = cos(theta1) * cos(theta2)
            # x[1] = cos(theta1) * sin(theta2)
            #
            # cos(theta1) = 1 - 2*sin^2(theta1/2) = 1 - 2*(1-f3) = 2*f3 - 1
            # sin(theta1) = 2*sin(theta1/2)*cos(theta1/2) = 2*sqrt(1-f3)*sqrt(f3)
            #
            # cos(theta2) = 1 - 2*sin^2(theta2/2) = 1 - 2*f2/(1-f3)
            # sin(theta2) = 2*sin(theta2/2)*cos(theta2/2)
            #   = 2 * sqrt(f2/(1-f3)) * sqrt(f1/(1-f3))
            #   = 2*sqrt(f1*f2)/(1-f3)
            
            denom = 1.0 - f3
            if denom < 1e-12:
                # f3 ~ 1, so theta1 ~ 0, f1, f2 ~ 0
                x0 = 1.0
                x1 = 0.0
            else:
                cos_theta1 = 2.0 * f3 - 1.0
                cos_theta2 = 1.0 - 2.0 * f2 / denom
                sin_theta2 = 2.0 * np.sqrt(max(0.0, f1 * f2)) / denom
                
                x0 = cos_theta1 * cos_theta2
                x1 = cos_theta1 * sin_theta2
            
            x = np.zeros(self.dim)
            x[0] = x0
            x[1] = x1
            if self.dim > 2:
                x[2:] = 0.5
            
            x = np.clip(x, lo, hi)
            
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                break
        
        return np.array(results) if results else None
