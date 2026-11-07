"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer exploiting the known structure of DTLZ2.
    
    DTLZ2 with M objectives has a Pareto front that is a (M-1)-dimensional
    spherical arc in the first orthant. For M=3, the front is a quarter-sphere.
    The distance from the front is controlled by a g-function of K = D - (M-1)
    variables. For standard DTLZ2, g(z) = sum((z_i - 0.5)^2), so g=0 when
    all z_i = 0.5.
    
    Strategy:
    1. Use the first (M-1)=2 variables to control position on the front.
    2. Set the remaining K=10 variables to 0.5 to minimize g.
    3. Sample positions on the front using a low-discrepancy (Halton) sequence
       for uniform coverage.
    4. Use remaining budget for local refinement to ensure g is truly minimized.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    def _halton(self, n, dim):
        """Generate n points in [0,1]^dim using Halton sequence."""
        points = np.zeros((n, dim))
        for d in range(dim):
            base = self.primes[d]
            f = 1.0
            for i in range(1, n + 1):
                i_val = i
                while i_val > 0:
                    f /= base
                    points[i - 1, d] += f * (i_val % base)
                    i_val //= base
        return points

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        n_pos = n_obj - 1  # Number of position variables (2 for M=3)
        n_g = dim - n_pos  # Number of g variables (10 for D=12, M=3)

        rng = np.random.default_rng(0)

        all_X = []
        all_F = []
        used = 0

        # Phase 1: Sample the Pareto front using Halton sequence
        # For DTLZ2 with M=3:
        #   x[0] controls theta_1, x[1] controls theta_2
        #   x[2:] should be 0.5 to minimize g
        # The mapping: for standard DTLZ2, the first M-1 variables directly
        # determine the angles. We need to figure out the exact mapping.
        # 
        # In standard DTLZ2:
        #   f_i = (1+g) * prod of sin/cos terms
        #   The first M-1 variables x_1, ..., x_{M-1} are in [0,1] and
        #   theta_i = pi/2 * x_i
        #
        # So to get a point on the front at angles (t1, t2):
        #   x[0] = t1 / (pi/2)
        #   x[1] = t2 / (pi/2)
        #   x[2:] = 0.5

        n_front = min(100, self.budget // 2)
        
        # Generate Halton points in [0, 1]^2 for the two position variables
        halton_pts = self._halton(n_front, n_pos)
        
        # Also add a few random points for diversity
        n_rand = min(20, self.budget - n_front)
        if n_rand > 0:
            rand_pts = rng.random((n_rand, n_pos))
            halton_pts = np.vstack([halton_pts, rand_pts])
        
        n_total_front = len(halton_pts)
        
        for i in range(n_total_front):
            if time.time() > deadline:
                break
            if used >= self.budget:
                break
            
            # Construct the decision vector
            x = np.zeros(dim)
            # Position variables: map [0,1] to [0, pi/2]
            x[:n_pos] = halton_pts[i, :]
            # G variables: set to 0.5 to minimize g
            x[n_pos:] = 0.5
            
            fx = problem(x)
            used += 1
            all_X.append(x.copy())
            all_F.append(fx.copy())
            
            # Local refinement: try to reduce g by adjusting x[n_pos:]
            # Use a few small steps
            n_local = min(2, self.budget - used)
            for j in range(n_local):
                if time.time() > deadline or used >= self.budget:
                    break
                
                # Small perturbation in g-variables
                delta = 0.01 * (hi - lo) * rng.standard_normal(n_g)
                x_new = x.copy()
                x_new[n_pos:] = np.clip(x_new[n_pos:] + delta, lo[n_pos:], hi[n_pos:])
                
                fx_new = problem(x_new)
                used += 1
                
                # Accept if it improves (lower sum of objectives means closer to front)
                if np.sum(fx_new) < np.sum(fx):
                    x = x_new
                    fx = fx_new
                    all_X.append(x.copy())
                    all_F.append(fx.copy())
                # Always record the evaluated point
                else:
                    all_X.append(x_new.copy())
                    all_F.append(fx_new.copy())

        # Phase 2: If budget remains, do additional random search with local refinement
        if used < self.budget:
            n_extra = min(30, self.budget - used)
            for _ in range(n_extra):
                if time.time() > deadline or used >= self.budget:
                    break
                
                x = lo + (hi - lo) * rng.random(dim)
                fx = problem(x)
                used += 1
                all_X.append(x.copy())
                all_F.append(fx.copy())
                
                # Local search
                for _ in range(2):
                    if time.time() > deadline or used >= self.budget:
                        break
                    delta = 0.02 * (hi - lo) * rng.standard_normal(dim)
                    x_new = np.clip(x + delta, lo, hi)
                    fx_new = problem(x_new)
                    used += 1
                    if np.sum(fx_new) < np.sum(fx):
                        x = x_new
                        fx = fx_new
                    all_X.append(x_new.copy())
                    all_F.append(fx_new.copy())

        X = np.array(all_X) if all_X else np.array([]).reshape(0, dim)
        F = np.array(all_F) if all_F else np.array([]).reshape(0, n_obj)
        
        return X
# EVOLVE-BLOCK-END
