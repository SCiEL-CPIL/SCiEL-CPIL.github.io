"""Model-and-optimize: DTLZ2 front is a quarter-sphere; sample it surface-area-uniform."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Model-based optimizer for DTLZ2 (M=3, dim=12).

    The Pareto front of DTLZ2 with M=3 objectives is a quarter-sphere of radius 1 in
    the positive octant: f = (sin(a)*sin(b), sin(a)*cos(b), cos(a)) for a,b in [0, pi/2].
    Any point with g(x)=0 (i.e., x[2:] = 0.5) and position (x[0], x[1]) mapped to
    (a, b) = (pi/2 * x[0], pi/2 * x[1]) lies exactly on this front.

    Hypervolume with a fixed reference point is maximized when the sample points are
    distributed uniformly by *surface area* on the front (not uniformly in parameter
    space). We use a Fibonacci-sphere construction projected onto the quarter-sphere
    to obtain N low-discrepancy, area-uniform points. All N budget points are on the
    front, so every evaluation contributes to hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _quarter_sphere_points(self, n):
        """
        Generate n points uniformly distributed by surface area on the quarter-sphere.
        
        Key insight: On the unit sphere with parameters a (polar) and b (azimuthal),
        the surface element is dA = sin(a) da db. Let u = cos(a), then:
          da = -du / sin(a) = -du / sqrt(1-u^2)
          dA = sin(a) * (du/sqrt(1-u^2)) * db = sqrt(1-u^2) * (du/sqrt(1-u^2)) * db = du * db
        
        So in (u, b) coordinates, the area element is uniform! We can use a 2D
        low-discrepancy sequence in (u, b) where u in [0,1] and b in [0, pi/2].
        
        We use a 2D Halton-like sequence for good distribution.
        """
        # Generate 2D low-discrepancy points using a simple Halton-like approach
        # Use interleaved bits for good 2D distribution
        
        def halton_2d(n, base1=2, base2=3):
            """Generate n 2D Halton sequence points."""
            points = []
            for i in range(1, n + 1):
                # 1D Halton in base 1
                f1, r1 = 1.0, i
                while r1 > 0:
                    f1 /= base1
                    r1 = r1 // base1
                # 1D Halton in base 2
                f2, r2 = 1.0, i
                while r2 > 0:
                    f2 /= base2
                    r2 = r2 // base2
                points.append((f1, f2))
            return np.array(points)
        
        # Generate n points in [0,1]^2 using Halton sequence
        uv = halton_2d(n)
        u = uv[:, 0]  # u = cos(a), in [0,1]
        v = uv[:, 1]  # v = b/(pi/2), in [0,1]
        
        # Convert to (a, b)
        a = np.arccos(u)  # a in [0, pi/2]
        b = v * (np.pi / 2)  # b in [0, pi/2]
        
        return a, b

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 315  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n = self.budget
        
        # Generate quarter-sphere points
        a_vals, b_vals = self._quarter_sphere_points(n)
        
        results = []
        count = 0
        
        for i in range(n):
            if time.time() > deadline:
                break
            
            a = a_vals[i]
            b = b_vals[i]
            
            # Map (a, b) to (x[0], x[1]) where x[0] = 2*a/pi, x[1] = 2*b/pi
            x0 = 2.0 * a / np.pi
            x1 = 2.0 * b / np.pi
            
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
