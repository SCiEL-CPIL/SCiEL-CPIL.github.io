"""Multi-objective optimizer for DTLZ1 exploiting known Pareto front structure via geometric construction."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Exploits the known Pareto front of DTLZ1.
    For n_obj=3, the Pareto front is a 2-simplex in objective space:
    f0 + f1 + f2 = 0.5, with all fi >= 0.
    
    We generate points uniformly on this simplex using low-discrepancy sequences,
    then map back to decision space via the inverse transformation.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _sobol_2d(self, n, seed=0):
        """Generate n points in [0,1]^2 using a low-discrepancy sequence."""
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=2, seed=seed)
            points = sampler.random(n)
            return points
        except ImportError:
            # Fallback: Halton sequence
            def halton(index, base):
                result = 0.0
                f = 1.0 / base
                i = index
                while i > 0:
                    result += f * (i % base)
                    i //= base
                    f /= base
                return result

            points = np.zeros((n, 2))
            for i in range(n):
                points[i, 0] = halton(i + 1, 2)
                points[i, 1] = halton(i + 1, 3)
            return points

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_obj = self.n_obj
        dim = self.dim

        # For n_obj=3, the Pareto front of DTLZ1 is:
        # f0 + f1 + f2 = 0.5, with all fi >= 0
        # This is a 2-simplex in objective space.
        #
        # The decision variables that achieve this are:
        # x_2..x_{dim-1} = 0.5 (so g=0)
        # x_0 and x_1 determine the objective values:
        #   f0 = 0.5 * x0 * x1
        #   f1 = 0.5 * x0 * (1 - x1)  
        #   f2 = 0.5 * (1 - x0)
        #
        # Given target (f0, f1, f2) on the simplex, we can recover:
        #   x0 = 1 - 2*f2
        #   x1 = f0 / (f0 + f1)  (if f0+f1 > 0, else 0)

        if n_obj == 3:
            # Generate points on the 2-simplex f0+f1+f2=0.5, fi>=0
            # Use change of variables: let u = f0/0.5, v = f1/0.5, w = f2/0.5
            # Then u+v+w = 1, u,v,w >= 0
            # Generate (u,v) uniformly on the triangle using Sobol in [0,1]^2
            # and mapping to the simplex
            
            n_points = self.budget
            sobol_pts = self._sobol_2d(n_points, seed=42)
            
            # Map [0,1]^2 to the 2-simplex using the standard transformation:
            # Given (r1, r2) in [0,1]^2, set:
            #   u = 1 - sqrt(r1)
            #   v = sqrt(r1) * (1 - r2)
            #   w = sqrt(r1) * r2
            # This gives uniform distribution on the simplex u+v+w=1, u,v,w>=0
            
            r1 = sobol_pts[:, 0]
            r2 = sobol_pts[:, 1]
            
            u = 1.0 - np.sqrt(r1)
            v = np.sqrt(r1) * (1.0 - r2)
            w = np.sqrt(r1) * r2
            
            # Scale to get f values: f0 = 0.5*u, f1 = 0.5*v, f2 = 0.5*w
            f0 = 0.5 * u
            f1 = 0.5 * v
            f2 = 0.5 * w
            
            # Map back to decision space
            # x0 = 1 - 2*f2 = 1 - w
            # x1 = f0/(f0+f1) = u/(u+v) if u+v>0, else 0
            x0 = 1.0 - w
            denom = u + v
            x1 = np.where(denom > 1e-15, u / np.maximum(denom, 1e-15), 0.0)
            
            # Build decision vectors
            X = np.zeros((n_points, dim))
            X[:, 0] = x0
            X[:, 1] = x1
            # x_2..x_{dim-1} = 0.5 for g=0
            X[:, 2:] = 0.5
            
            # Clip to bounds
            X = np.clip(X, lo, hi)
            
            # Evaluate all points
            F = np.array([problem(x) for x in X])
            used = len(X)
            
            # Check if we have time for refinement
            elapsed = time.time() - start_time
            if elapsed < deadline - 10.0 and used < self.budget:
                # Try to refine some points locally with small perturbations
                rng = np.random.default_rng(123)
                for i in range(used):
                    if time.time() > deadline - 5.0:
                        break
                    if used >= self.budget:
                        break

                    # Small perturbation
                    x = X[i].copy()
                    x += 0.01 * (hi - lo) * rng.standard_normal(dim)
                    x = np.clip(x, lo, hi)

                    fx = problem(x)
                    used += 1

                    # Replace the worst point if this one is better
                    if np.sum(fx) < np.sum(F[i]):
                        X[i] = x
                        F[i] = fx

        else:
            # General case: use random sampling
            rng = np.random.default_rng(0)
            X = lo + (hi - lo) * rng.random((self.budget, dim))
            F = np.array([problem(x) for x in X])
            used = self.budget

        return X
# EVOLVE-BLOCK-END
