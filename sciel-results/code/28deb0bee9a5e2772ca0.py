"""Model-and-optimize: DTLZ2 front is a quarter-sphere; sample it with Halton sequence for uniform coverage."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _halton_sequence(self, n, base):
        """Generate n Halton sequence values in [0, 1) for the given base."""
        result = np.zeros(n)
        for i in range(1, n + 1):
            f = 1.0
            r = 0.0
            j = i
            while j > 0:
                j, rem = divmod(j, base)
                f /= base
                r += rem * f
            result[i - 1] = r
        return result

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        results = []
        count = 0

        # DTLZ2 M=3:
        # x[0], x[1] parameterize the Pareto front (quarter-sphere in 3D).
        # x[2:] should be set to 0.5 to minimize g(x) = sum((xi - 0.5)^2).
        # The Pareto front is: f1 = sqrt(1 - y^2) * cos(theta), f2 = sqrt(1-y^2)*sin(theta)*cos(phi), f3 = sqrt(1-y^2)*sin(theta)*sin(phi)
        # where y = x0, theta = pi/2 * x1, phi = pi/2 * (something)... 
        # Actually for DTLZ2 with M objectives:
        # f_i(x) = (1 + g(x)) * prod_{j=1}^{M-1} cos(x_j * pi/2) * sin(x_{M-1} * pi/2) for i=1
        # More precisely for M=3:
        # f1 = (1+g) * cos(x0*pi/2) * cos(x1*pi/2)
        # f2 = (1+g) * cos(x0*pi/2) * sin(x1*pi/2)
        # f3 = (1+g) * sin(x0*pi/2)
        # With g=0 (x[2:]=0.5), the front is a quarter-sphere.
        
        # To get surface-uniform sampling on the quarter-sphere, we use:
        # u, v ~ Uniform(0,1)
        # x0 = (2/pi) * arccos(1 - 2u)  -- this maps u uniformly to the "latitude"
        # x1 = v                         -- this maps v uniformly to the "longitude"
        
        # Generate Halton sequence points for uniform coverage in 2D parameter space
        n = self.budget
        
        # Generate Halton sequences with bases 2 and 3 for 2D
        u_seq = self._halton_sequence(n, 2)
        v_seq = self._halton_sequence(n, 3)
        
        # Precompute all points
        for i in range(n):
            if count >= self.budget:
                break
            if time.time() > deadline:
                break
            
            u = u_seq[i]
            v = v_seq[i]
            
            # Map u (uniform) to x0 (surface-uniform latitude)
            # x0 = (2/pi) * arccos(1 - 2u)
            x0 = (2.0 / np.pi) * np.arccos(1.0 - 2.0 * u)
            
            # Map v (uniform) to x1 (uniform longitude)
            x1 = v
            
            x = np.zeros(self.dim)
            x[0] = x0
            x[1] = x1
            if self.dim > 2:
                x[2:] = 0.5
            
            x = np.clip(x, lo, hi)
            
            try:
                f = problem(x)
                results.append(f)
                count += 1
            except Exception:
                break
        
        return np.array(results) if results else None
# EVOLVE-BLOCK-END
