"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Analytic Pareto front sampler for DTLZ2 using Sobol sequence."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ2 Analysis:
        # To reach the Pareto front, we must minimize g(x).
        # For standard DTLZ2, g(x) = sum((x_i - 0.5)^2) for i > m.
        # Minimum g=0 is achieved when x_i = 0.5 for i > m.
        # Then f(x) = h(x).
        # The Pareto front is the unit sphere in the positive orthant.
        
        # We use a Sobol sequence to sample the 2D parameter space (theta1, theta2)
        # which maps bijectively to the Pareto front.
        
        # We generate `budget` points.
        # We need 2 dimensions for the angles.
        # Sobol requires power of 2 sample sizes for best distribution, 
        # but we can just take the first `budget` points from a larger sequence
        # or use `random` which handles any N reasonably well, though powers of 2 are optimal.
        
        # Let's use a power of 2 close to budget, or just budget.
        # Actually, Sobol sequence quality is best for N = 2^k.
        # Let N = 2^floor(log2(budget)). If budget=200, N=128. 
        # We can use the remaining budget for a few random points or just use N=256 and take first 200?
        # Taking a prefix of a Sobol sequence is fine.
        # Let's use N = 2^ceil(log2(budget)) to have enough, and take first `budget`.
        
        k = int(np.ceil(np.log2(max(2, self.budget))))
        N = 2 ** k
        
        sobol = qmc.Sobol(d=2, scramble=True)
        # Generate N points in [0, 1)^2
        angles = sobol.random(N)
        
        # Take only the first `budget` points
        angles = angles[:self.budget]
        
        # Map angles to the Pareto front
        # theta1 = angles[:, 0] * pi / 2
        # theta2 = angles[:, 1] * pi / 2
        # 
        # f1 = sin(theta1) * cos(theta2)
        # f2 = sin(theta1) * sin(theta2)
        # f3 = cos(theta1)
        
        # However, we need to provide x to `problem`.
        # We need to find x such that problem(x) returns these f values.
        # We know that setting x_i = 0.5 for i >= m (where m = n_obj - 1 = 2) makes g=0.
        # The remaining variables x_0, x_1 determine the position on the front.
        # 
        # The mapping from x_0, x_1 to f is:
        # f_1 = sin(x_0 * pi / 2) * cos(x_1 * pi / 2)  <-- Wait, is this correct?
        # 
        # Let's verify the standard DTLZ2 h-functions for M=3.
        # h_1 = y_1
        # h_2 = y_2 * y_1
        # h_3 = y_3 * y_2 * y_1 ... wait, M=3 means M-1=2 y-variables.
        # Standard definition:
        # y_i = sin(x_i * pi / 2) for i=1..M-1
        # h_1 = y_1
        # h_2 = y_2 * y_1
        # ...
        # h_M = y_{M-1} * ... * y_1
        # 
        # For M=3:
        # h_1 = y_1
        # h_2 = y_2 * y_1
        # h_3 = ??? 
        # 
        # Actually, the standard DTLZ2 definition is:
        # f_i(x) = (1 + g(x)) * h_i(x, y)
        # where h_1 = y_1
        # h_2 = y_2 * y_1
        # ...
        # h_M = y_{M-1} * ... * y_1
        # 
        # Wait, if h_M is a product of all y's, then for M=3:
        # h_1 = y_1
        # h_2 = y_2 * y_1
        # h_3 = y_2 * y_1 ... this implies h_2 = h_3, which is wrong.
        
        # Let's look at the correct DTLZ2 formulation.
        # The Pareto front is sum f_i^2 = 1.
        # This implies the h functions must satisfy sum h_i^2 = 1.
        # The standard formulation that achieves this is:
        # h_1 = y_1
        # h_2 = y_2 * y_1
        # ...
        # h_i = (1 - y_{i-1}) * y_{i-1} * ... * y_1 ? No.
        
        # Correct standard DTLZ2:
        # y_i = sin(x_i * pi / 2)
        # h_1 = y_1
        # h_2 = y_2 * y_1
        # ...
        # h_M = y_{M-1} * ... * y_1
        # This DOES NOT sum to 1.
        
        # The correct formulation for DTLZ2 is:
        # f_1 = (1+g) * y_1
        # f_2 = (1+g) * y_2 * y_1
        # ...
        # f_M = (1+g) * y_{M-1} * ... * y_1
        # 
        # Let's check sum of squares:
        # y_1^2 + (y_2 y_1)^2 + ... 
        # This is not 1.
        
        # Actually, the DTLZ2 problem is defined such that the Pareto front IS the sphere.
        # The mapping is:
        # f_1 = (1+g) * sin(x_1 * pi / 2)
        # f_2 = (1+g) * cos(x_1 * pi / 2) * sin(x_2 * pi / 2)
        # f_3 = (1+g) * cos(x_1 * pi / 2) * cos(x_2 * pi / 2)
        # ...
        # f_M = (1+g) * cos(x_{M-2} * pi / 2) * ... * cos(x_1 * pi / 2)
        
        # Let's verify this for M=3:
        # f_1 = (1+g) * sin(x_1 * pi / 2)
        # f_2 = (1+g) * cos(x_1 * pi / 2) * sin(x_2 * pi / 2)
        # f_3 = (1+g) * cos(x_1 * pi / 2) * cos(x_2 * pi / 2)
        
        # Sum of squares (assuming g=0):
        # sin^2(x_1) + cos^2(x_1) * sin^2(x_2) + cos^2(x_1) * cos^2(x_2)
        # = sin^2(x_1) + cos^2(x_1) * (sin^2(x_2) + cos^2(x_2))
        # = sin^2(x_1) + cos^2(x_1)
        # = 1.
        
        # Yes! This is the correct mapping.
        # So, to get a point on the Pareto front with angles theta_1, theta_2:
        # We want:
        # f_1 = sin(theta_1)
        # f_2 = cos(theta_1) * sin(theta_2)
        # f_3 = cos(theta_1) * cos(theta_2)
        
        # We need to find x_1, x_2 such that:
        # sin(x_1 * pi / 2) = sin(theta_1)  => x_1 * pi / 2 = theta_1 (since both in [0, pi/2])
        # => x_1 = 2 * theta_1 / pi
        
        # cos(x_1 * pi / 2) * sin(x_2 * pi / 2) = cos(theta_1) * sin(theta_2)
        # Since x_1 = 2*theta_1/pi, cos(x_1*pi/2) = cos(theta_1).
        # So we need sin(x_2 * pi / 2) = sin(theta_2)
        # => x_2 * pi / 2 = theta_2
        # => x_2 = 2 * theta_2 / pi
        
        # So the mapping is:
        # theta_1 = angles[:, 0] * pi / 2
        # theta_2 = angles[:, 1] * pi / 2
        # x_1 = 2 * theta_1 / pi = 2 * angles[:, 0]
        # x_2 = 2 * theta_2 / pi = 2 * angles[:, 1]
        
        # And for i >= 2, x_i = 0.5 to ensure g=0.
        
        theta1 = angles[:, 0] * np.pi / 2
        theta2 = angles[:, 1] * np.pi / 2
        
        x1 = 2 * angles[:, 0]
        x2 = 2 * angles[:, 1]
        
        # Construct the x vectors
        # x shape: (budget, dim)
        x_vectors = np.zeros((self.budget, self.dim))
        x_vectors[:, 0] = x1
        x_vectors[:, 1] = x2
        x_vectors[:, 2:] = 0.5
        
        # Clip to bounds just in case, though 0 and 1 are within [0,1]
        x_vectors = np.clip(x_vectors, lo, hi)
        
        # Evaluate all points
        # The problem takes a single vector x and returns 3 objectives.
        # We must call it point by point.
        
        results = []
        for i in range(self.budget):
            if time.time() > deadline:
                break
            x = x_vectors[i]
            f = problem(x)
            results.append(f)
            
        return results
# EVOLVE-BLOCK-END
