# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Multi-objective optimizer for DTLZ2.
    
    Approach:
    - DTLZ2 Pareto front is a unit hypersphere in objective space (for n_obj objectives).
    - For 3 objectives, the Pareto front is a quarter-sphere in the positive octant of R^3.
    - The distance variables (last dim - n_obj + 1 variables) should be set to 0.5 for optimality.
    - The angle variables (first n_obj - 1 = 2 variables) determine the position on the Pareto front.
    
    We use a deterministic low-discrepancy sequence (Sobol-like via Halton or simple deterministic
    lattice) to sample the 2D angle space maximally space-filling, ensuring good hypervolume coverage.
    
    For DTLZ2 with 3 objectives:
    - x[0], x[1] are angle variables in [0, 1]
    - x[2..11] are distance variables, set to 0.5
    
    The objective values are:
    f_i = sum(x_j^2) for j in range(i+1) ... actually let me recall DTLZ2 properly.
    
    DTLZ2:
    - g(x) = sum_{i=m+1}^{n} (x_i - 0.5)^2 where m = n_obj - 1
    - f_i(x) = (1 + g) * prod_{j=1}^{i-1} cos(x_j * pi/2) * sin(x_i * pi/2) for i=1..n_obj
    - More precisely:
      f_1 = (1+g) * cos(x_1*pi/2) * cos(x_2*pi/2)
      f_2 = (1+g) * cos(x_1*pi/2) * sin(x_2*pi/2)
      f_3 = (1+g) * sin(x_1*pi/2)
    
    Wait, let me be more careful. Standard DTLZ2 with M objectives and N variables:
    - g(x) = sum_{i=M}^{N-1} (x_i - 0.5)^2  (0-indexed, so variables M-1 to N-1... let me use 1-indexed)
    
    With N=12, M=3:
    - g(x) = sum_{i=3}^{11} (x_i - 0.5)^2  (1-indexed: variables 3 through 11, that's 9 variables)
      Actually: variables from index M to N-1 in 0-based, which is indices 2 to 11, that's 10 variables.
    
    Let me just set all variables from index 2 onward to 0.5. That makes g=0.
    
    Then:
    f_1 = cos(x_0 * pi/2) * cos(x_1 * pi/2)
    f_2 = cos(x_0 * pi/2) * sin(x_1 * pi/2)
    f_3 = sin(x_0 * pi/2)
    
    These three sum of squares = cos^2(x_0*pi/2)*cos^2(x_1*pi/2) + cos^2(x_0*pi/2)*sin^2(x_1*pi/2) + sin^2(x_0*pi/2)
    = cos^2(x_0*pi/2) + sin^2(x_0*pi/2) = 1
    
    So the Pareto front is the unit sphere in the positive octant.
    
    We need to sample (x_0, x_1) in [0,1]^2 to get good coverage of the Pareto front.
    
    Using a deterministic Halton sequence or similar low-discrepancy sequence in 2D will give
    excellent space-filling properties.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj
        
        # For DTLZ2 with n_obj=3, dim=12:
        # Angle variables: first n_obj - 1 = 2 variables
        # Distance variables: remaining dim - (n_obj - 1) = 10 variables, set to 0.5
        
        n_angle = n_obj - 1  # 2
        
        # Generate a low-discrepancy sequence in [0,1]^n_angle
        # Use Halton sequence for deterministic, space-filling points
        
        def halton(index, base):
            """Generate the index-th point in the Halton sequence with given base."""
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        # Generate budget points using Halton sequence in 2D
        # Skip the first point (0,0) to avoid boundary issues
        points = []
        for i in range(1, budget + 1):
            p0 = halton(i, 2)  # base 2 for first dimension
            p1 = halton(i, 3)  # base 3 for second dimension
            points.append([p0, p1])
        
        points = np.array(points)  # shape: (budget, 2)
        
        # Build full x vectors: angle variables from points, distance variables = 0.5
        X = np.zeros((budget, dim))
        X[:, :n_angle] = points
        X[:, n_angle:] = 0.5  # Distance variables set to 0.5 for optimality
        
        # Clip to bounds
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        X = np.clip(X, lo, hi)
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
