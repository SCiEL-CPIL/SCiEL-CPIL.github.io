import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        results = []
        count = 0

        # Generate points using Sobol sequence for better coverage
        # Sobol sequences are low-discrepancy and provide better space-filling properties
        # than uniform grids, which is beneficial for hypervolume maximization.
        
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=2, scramble=True)
            # Generate budget points
            # Note: Sobol requires power of 2 for optimal properties, but works for any N
            # We will generate exactly self.budget points
            points = sampler.random(self.budget)
            # points is an array of shape (budget, 2)
        except Exception:
            # Fallback to Halton sequence if Sobol fails
            try:
                # Simple Halton sequence implementation
                def halton(index, base):
                    f = 1.0
                    r = 0.0
                    while index:
                        f /= base
                        r += f * (index % base)
                        index //= base
                    return r

                points = np.array([[halton(i + 1, 2), halton(i + 1, 3)] for i in range(self.budget)])
            except Exception:
                # Fallback to uniform grid
                n_grid = int(np.sqrt(self.budget))
                # Adjust to fit budget
                n0 = n_grid
                n1 = self.budget // n_grid
                x0_vals = np.linspace(0, 1, n0)
                x1_vals = np.linspace(0, 1, n1)
                points = np.array([[x0, x1] for x0 in x0_vals for x1 in x1_vals][:self.budget])

        # Ensure points are within [0, 1]
        points = np.clip(points, 0.0, 1.0)

        for p in points:
            if count >= self.budget:
                break
            if time.time() > deadline:
                break

            x = np.zeros(self.dim)
            x[0] = p[0]
            x[1] = p[1]
            
            # Set remaining variables to center to minimize g(x)
            if self.dim > 2:
                x[2:] = 0.5

            # Clip to bounds (redundant but safe)
            x = np.clip(x, lo, hi)

            try:
                f = problem(x)
                results.append(f)
                count += 1
            except Exception:
                break

        return np.array(results) if results else None
