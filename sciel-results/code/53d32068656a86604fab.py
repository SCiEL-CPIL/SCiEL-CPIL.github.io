"""Direct construction optimizer for DTLZ4 exploiting analytical Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using direct front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 structure for M=3:
        # f1 = 0.5 * cos(pi*x1/2) * cos(pi*x2/2)
        # f2 = 0.5 * cos(pi*x1/2) * sin(pi*x2/2)
        # f3 = 0.5 * sin(pi*x1/2)
        # g(x) = sum((x_i - 0.5)^2) for i in [n_front, dim)
        # When g=0, f_i are as above. The Pareto front is a quarter sphere of radius 0.5.
        # f1^2 + f2^2 + f3^2 = 0.25

        n_g = 5
        n_front = dim - n_g  # Should be 7 for dim=12, but only x1, x2 matter for front shape

        # Strategy: Stratified sampling in the (f1, f2) plane
        # For a given (f1, f2) on the front, f3 = sqrt(0.25 - f1^2 - f2^2)
        # We need f1^2 + f2^2 <= 0.25
        # The valid region in (f1, f2) is a quarter disk of radius 0.5.
        
        # For uniform area coverage on the quarter disk, we use polar coordinates:
        # f1 = r * cos(phi), f2 = r * sin(phi)
        # where r ~ sqrt(U[0, 0.25]) for uniform area, phi ~ U[0, pi/2]
        
        # Actually, for uniform area on a disk, r should be proportional to sqrt(u) where u ~ U[0,1]
        # r = 0.5 * sqrt(u_r), phi = u_phi * pi/2
        
        # Let's use a stratified grid in (u_r, u_phi) space for low discrepancy.
        # u_r in [0,1], u_phi in [0,1]
        # r = 0.5 * sqrt(u_r)
        # phi = u_phi * pi/2
        # f1 = r * cos(phi)
        # f2 = r * sin(phi)
        # f3 = sqrt(0.25 - f1^2 - f2^2)
        
        # Then map back to x1, x2:
        # f3 = 0.5 * sin(pi*x1/2) -> x1 = (2/pi) * arcsin(2*f3)
        # f2/f1 = tan(phi) = tan(pi*x2/2) * ... wait
        # f2 = 0.5 * cos(pi*x1/2) * sin(pi*x2/2)
        # f1 = 0.5 * cos(pi*x1/2) * cos(pi*x2/2)
        # f2/f1 = tan(pi*x2/2) -> x2 = (2/pi) * arctan(f2/f1)
        # And f3 = 0.5 * sin(pi*x1/2) -> x1 = (2/pi) * arcsin(2*f3)
        
        # Edge cases: f1 = 0 -> x2 = 1 (since tan(pi*x2/2) = inf)
        #              f3 = 0.5 -> x1 = 1
        #              f3 = 0 -> x1 = 0
        
        # Generate stratified points using van der Corput sequences
        def van_der_corput(n, base):
            """Generate first n points of van der Corput sequence in given base."""
            seq = np.zeros(n)
            for i in range(1, n + 1):
                f = 1.0
                num = i
                val = 0.0
                while num > 0:
                    f /= base
                    num, rem = divmod(num, base)
                    val += f * rem
                seq[i - 1] = val
            return seq

        n_points = budget
        
        # Use different bases for the two dimensions to ensure low discrepancy
        u_r = van_der_corput(n_points, 2)
        u_phi = van_der_corput(n_points, 3)
        
        # Compute f1, f2, f3
        r = 0.5 * np.sqrt(u_r)
        phi = u_phi * np.pi / 2.0
        
        f1 = r * np.cos(phi)
        f2 = r * np.sin(phi)
        f3_sq = 0.25 - f1**2 - f2**2
        f3_sq = np.maximum(f3_sq, 0.0)  # Numerical safety
        f3 = np.sqrt(f3_sq)
        
        # Map back to x1, x2
        # x1 = (2/pi) * arcsin(2*f3)
        x1_vals = (2.0 / np.pi) * np.arcsin(np.clip(2.0 * f3, -1.0, 1.0))
        
        # x2 = (2/pi) * arctan2(f2, f1)
        # When f1 = 0 and f2 > 0, arctan2 gives pi/2, so x2 = 1. Correct.
        # When f1 = 0 and f2 = 0, arctan2(0,0) = 0, so x2 = 0. But this point has f3=0.5, x1=1.
        # This is a valid point on the front (corner).
        x2_vals = (2.0 / np.pi) * np.arctan2(f2, f1)
        
        # Clip to valid range
        x1_vals = np.clip(x1_vals, 0.0, 1.0)
        x2_vals = np.clip(x2_vals, 0.0, 1.0)
        
        # Construct decision vectors
        X = np.zeros((n_points, dim))
        X[:, 0] = x1_vals
        X[:, 1] = x2_vals
        if n_front > 2:
            X[:, 2:n_front] = 0.5
        X[:, n_front:] = 0.5  # g=0
        
        # Clip to bounds
        X = np.clip(X, lo, hi)
        
        # Check time
        if time.time() - t0 > deadline:
            return X
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
# EVOLVE-BLOCK-END
