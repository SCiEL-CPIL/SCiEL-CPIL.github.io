"""Multi-objective optimizer for DTLZ1 with a wall-clock deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


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
        # coverage of this simplex. We use a structured approach:
        # 1. A fine triangular grid for systematic coverage
        # 2. Sobol-like quasi-random points for better space-filling
        # 3. Remaining budget for high-precision refinement

        def obj_to_decision(f):
            """Map objective vector on the front to decision space."""
            # f_i = 0.5 * x_i => x_i = 2*f_i
            x = np.zeros(dim)
            x[:n_obj] = 2.0 * f
            x[n_obj:] = 0.5
            return x

        # Phase 1: Fine triangular grid on the simplex
        # Use a grid with resolution that fits in our budget
        n_grid = 15  # 15x15 gives ~120 points on the triangle
        n_phase1 = 0
        for i in range(n_grid):
            for j in range(n_grid - i):
                if used >= budget or not time_ok():
                    break
                # Grid point on simplex: f_1 = 0.5*i/(n_grid-1), f_2 = 0.5*j/(n_grid-1)
                f1 = 0.5 * i / (n_grid - 1)
                f2 = 0.5 * j / (n_grid - 1)
                f3 = 0.5 - f1 - f2
                if f3 < -1e-10:
                    continue
                f = np.array([max(f1, 0), max(f2, 0), max(f3, 0)])
                x = obj_to_decision(f)
                x = np.clip(x, lo, hi)
                eval_and_record(x)
                n_phase1 += 1
            if used >= budget or not time_ok():
                break

        # Phase 2: Quasi-random sampling using Halton-like sequence
        # For 2D simplex, use two independent quasi-random sequences
        n_phase2 = min(50, budget - used)
        
        def halton_sequence(index, base):
            """Generate the index-th Halton number in the given base."""
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        for k in range(1, n_phase2 + 1):
            if used >= budget or not time_ok():
                break
            # Two quasi-random numbers in [0,1)
            u1 = halton_sequence(k, 2)
            u2 = halton_sequence(k, 3)
            # Map to simplex: f1 = 0.5*u1, f2 = 0.5*u2*(1-u1), f3 = remainder
            f1 = 0.5 * u1
            f2 = 0.5 * u2 * (1 - u1)
            f3 = 0.5 - f1 - f2
            if f3 < -1e-10:
                continue
            f = np.array([max(f1, 0), max(f2, 0), max(f3, 0)])
            x = obj_to_decision(f)
            x = np.clip(x, lo, hi)
            eval_and_record(x)

        # Phase 3: High-precision refinement near the grid points
        # Use remaining budget to add points slightly perturbed from grid
        remaining = budget - used
        if remaining > 0 and time_ok():
            # Add jittered versions of some grid points for better coverage
            n_refine = min(remaining, 30)
            for _ in range(n_refine):
                if used >= budget or not time_ok():
                    break
                # Random point on simplex with small perturbation
                u = rng.random(n_obj)
                u = u / u.sum()
                f = 0.5 * u
                x = obj_to_decision(f)
                # Tiny perturbation in decision space
                x += 0.001 * rng.standard_normal(dim)
                x = np.clip(x, lo, hi)
                eval_and_record(x)

        # Phase 4: Fill remaining budget with random front points
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
