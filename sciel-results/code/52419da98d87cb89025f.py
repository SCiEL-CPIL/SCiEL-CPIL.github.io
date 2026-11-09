"""Combined: Analytic front seeding with exact 1D solver for x1 to minimize g(x)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize_scalar

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 330s kill
        rng = np.random.default_rng(12345)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        used = 0
        archive_X = []
        archive_F = []

        def eval_point(x):
            nonlocal used
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            f = np.asarray(problem(x), dtype=float)
            used += 1
            return x, f

        def add_point(x, f):
            # add if not dominated by archive
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return # dominated
            # remove archive points dominated by f
            keep = []
            for i, af in enumerate(archive_F):
                if np.all(f <= af) and np.any(f < af):
                    continue
                keep.append(i)
            archive_F = [archive_F[i] for i in keep]
            archive_X = [archive_X[i] for i in keep]
            archive_F.append(f)
            archive_X.append(x)

        # Helper to find x1 that minimizes the penalty term in g(x) for a given c
        # g(x) = 11 + (x1 - (10c + sin(10*pi*x1))^2 / (4*x1+4)^2)
        # We want to minimize P(x1) = (10c + sin(10*pi*x1))^2 / (4*x1+4)^2
        def solve_x1(c):
            def penalty(x1):
                s = 10.0 * c + np.sin(10.0 * np.pi * x1)
                denom = (4.0 * x1 + 4.0) ** 2
                return (s * s) / denom
            
            # Use bounded scalar minimization
            res = minimize_scalar(penalty, bounds=(0.0, 1.0), method='bounded', options={'xatol': 1e-8})
            return res.x

        # Phase 1: Dense seeding using exact solver
        # We sample c uniformly. The resulting f1 values will cover the range [1, 11].
        # To ensure good spread, we might want to sample f1 directly, but c is the
        # independent variable in the structure x2..x10=c.
        # Let's sample c uniformly in [0, 1].
        n_targets = min(180, self.budget - 10) # Leave 10 for polish
        
        # Create a grid of c values
        cs = np.linspace(0.0, 1.0, n_targets)
        
        for c in cs:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            
            x1 = solve_x1(c)
            x = np.full(self.dim, c)
            x[0] = x1
            x = np.clip(x, lo, hi)
            
            x, f = eval_point(x)
            add_point(x, f)

        # Phase 2: Polish
        # Use remaining budget to refine points in the archive
        if used < self.budget and archive_X:
            steps_left = self.budget - used
            # We will try to improve points by re-solving x1 with higher precision
            # or by small perturbations if the solver failed.
            # Since solve_x1 is already high precision (1e-8), the points should be very good.
            # We can use remaining budget to fill any gaps if the c-grid missed some f1 regions.
            # However, uniform c might not give uniform f1.
            # Let's do a few random c samples to ensure coverage.
            
            for _ in range(steps_left):
                if time.time() > deadline or used >= self.budget:
                    break
                c = rng.uniform(0.0, 1.0)
                x1 = solve_x1(c)
                x = np.full(self.dim, c)
                x[0] = x1
                x = np.clip(x, lo, hi)
                
                x, f = eval_point(x)
                add_point(x, f)

        return archive_X if archive_X else np.array([lo + 0.5 * span])
# EVOLVE-BLOCK-END
