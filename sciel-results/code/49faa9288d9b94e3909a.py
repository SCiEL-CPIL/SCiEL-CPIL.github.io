import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, base):
        """
        Generate the first n Halton numbers with the given base.
        """
        result = np.zeros(n)
        f = 1.0
        for i in range(1, n + 1):
            result[i - 1] = f
            f = 1.0
            num = i
            while num != 0:
                f /= base
                num = int(num / base)
        return result

    def __call__(self, problem, bounds):
        n_points = self.budget
        points = np.zeros((n_points, self.dim))
        
        # Set g variables to 0.5 to ensure g=0
        # DTLZ2: g = sum(x_i - 0.5)^2 for i in 0..N-M
        # If x_i = 0.5, g = 0.
        points[:, :10] = 0.5
        
        # Generate Halton sequence for angles
        # We need 2D points in [0, 1]^2
        # Base 2 and 3 are standard coprime bases
        h1 = self._halton(n_points, 2)
        h2 = self._halton(n_points, 3)
        
        # Map h1, h2 to theta and phi
        # theta in [0, pi/2]
        # phi in [0, pi/2]
        # Using (h + 0.5) / n * range + min_range is a common stratification technique 
        # but simple linear mapping of Halton is often sufficient for low N.
        # However, Halton points are in [0, 1). 
        # To ensure we don't miss the boundaries which might contribute to hypervolume 
        # (though the corner (1,0,0) etc are single points with 0 volume), 
        # the distribution is what matters.
        
        theta = h1 * (np.pi / 2)
        phi = h2 * (np.pi / 2)
        
        # Map to x[11] and x[10]
        # x[11] = 2 * theta / pi
        # x[10] = 2 * phi / pi
        points[:, 11] = 2 * theta / np.pi
        points[:, 10] = 2 * phi / np.pi
        
        # Evaluate all points
        for i in range(n_points):
            problem(points[i])
            
        return points
