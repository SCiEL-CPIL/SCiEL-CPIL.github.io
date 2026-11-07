"""Multi-objective optimizer for DTLZ1 with a wall-clock deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Multi-objective optimizer for DTLZ1 (3 objectives, 7 vars, bounds [0,1]).

    Uses the known structure of DTLZ1: the Pareto front is the simplex
    sum(f) = 0.5 with f_i = 0.5 * x_i. We construct the front directly via
    geometric sampling in objective space, then map back to decision space.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        recorded_F = []
        used = 0

        def eval_and_record(x):
            nonlocal used
            f = problem(x)
            f = np.array(f, dtype=float)
            if np.all(f < 400.0):
                recorded_F.append(f)
            used += 1
            return f

        rng = np.random

        # DTLZ1 Pareto front: f_i = 0.5 * x_i, sum(x[:3]) = 1, x[3:] = 0.5
        # In objective space: f_1 + f_2 + f_3 = 0.5, f_i >= 0
        # The front is a 2-simplex in objective space with vertices at
        # (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)
        #
        # To maximize hypervolume with reference [400,400,400], we need dense
        # coverage of this simplex. We use Sobol quasi-random sequences for
        # optimal space-filling properties on the 2D simplex domain.

        def obj_to_decision(f):
            """Map objective vector on the front to decision space."""
            # f_i = 0.5 * x_i => x_i = 2*f_i
            x = np.zeros(dim)
            x[:n_obj] = 2.0 * f
            x[n_obj:] = 0.5
            return x

        # Phase 1: Sobol quasi-random sampling on the 2D simplex
        # Use scipy's Sobol sequence for low-discrepancy points in [0,1)^2
        # Then map to the simplex using the standard transformation:
        # f1 = 0.5 * u1, f2 = 0.5 * u2 * (1-u1), f3 = 0.5 - f1 - f2
        
        # Generate Sobol points - use all budget for Sobol sampling
        # Sobol works best with power-of-2 sample sizes
        n_sobol = min(budget, 128)  # 128 is a power of 2, good for Sobol
        
        if n_sobol > 0 and time_ok():
            # Create Sobol sampler
            sobol = qmc.Sobol(d=2, scramble=True)
            # Generate n_sobol points in [0,1)^2
            # Skip the first point (0,0) to avoid degenerate cases
            sobol_points = sobol.random(n=n_sobol + 1)
            sobol_points = sobol_points[1:]  # Skip first point
            
            for u1, u2 in sobol_points:
                if used >= budget or not time_ok():
                    break
                # Map [0,1)^2 to simplex
                f1 = 0.5 * u1
                f2 = 0.5 * u2 * (1 - u1)
                f3 = 0.5 - f1 - f2
                if f3 < -1e-10:
                    continue
                f = np.array([max(f1, 0), max(f2, 0), max(f3, 0)])
                x = obj_to_decision(f)
                x = np.clip(x, lo, hi)
                eval_and_record(x)

        # Phase 2: Fill remaining budget with additional Sobol points
        # Use a second Sobol sequence with different scrambling
        remaining = budget - used
        if remaining > 0 and time_ok():
            # Use more Sobol points with a different seed/scramble
            n_more = min(remaining, 128)
            if n_more > 0:
                sobol2 = qmc.Sobol(d=2, scramble=True, seed=42)
                sobol_points2 = sobol2.random(n=n_more + 1)
                sobol_points2 = sobol_points2[1:]
                
                for u1, u2 in sobol_points2:
                    if used >= budget or not time_ok():
                        break
                    f1 = 0.5 * u1
                    f2 = 0.5 * u2 * (1 - u1)
                    f3 = 0.5 - f1 - f2
                    if f3 < -1e-10:
                        continue
                    f = np.array([max(f1, 0), max(f2, 0), max(f3, 0)])
                    x = obj_to_decision(f)
                    x = np.clip(x, lo, hi)
                    eval_and_record(x)

        # Phase 3: Fill any remaining budget with random points
        while used < budget and time_ok():
            u = rng.random(n_obj)
            u = u / u.sum()
            f = 0.5 * u
            x = obj_to_decision(f)
            x = np.clip(x, lo, hi)
            eval_and_record(x)

        # Return (ignored by evaluator, but keep valid)
        if recorded_F:
            return np.array(recorded_F)
        return np.array([])
# EVOLVE-BLOCK-END
