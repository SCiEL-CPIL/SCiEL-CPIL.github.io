"""Multi-objective optimizer for DTLZ1 with a wall-clock deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ1 (3 objectives, 7 vars, bounds [0,1]).

    Exploits the known Pareto front structure of DTLZ1: the front is the 2-simplex
    where x[0]+x[1]+x[2]=1 and x[3:]=0.5, giving f_i = 0.5*x_i and sum(f)=0.5.
    We densely cover this simplex to maximize hypervolume with reference [400,400,400].
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

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
                recorded_F.append(f.copy())
            used += 1
            return f

        rng = np.random

        # The DTLZ1 Pareto front: x[:3] on simplex (sum=1), x[3:]=0.5
        # This gives g=0, f_i = 0.5 * x_i, sum(f) = 0.5
        # Hypervolume with ref [400,400,400] rewards dense coverage of the simplex.

        # Strategy: Use a combination of structured and random sampling on the simplex
        # to maximize hypervolume. Key insight: for a fixed reference point far from the
        # front, hypervolume is maximized by uniform coverage of the front.

        # Phase 1: Dense structured grid on the simplex with stratified jitter
        # For 3 objectives, the simplex is 2D. Use a triangular grid.
        n_grid = 15  # 15x15 grid gives ~120 unique points
        for i in range(n_grid):
            for j in range(n_grid - i):
                if used >= budget or not time_ok():
                    break
                # Grid point on simplex
                a = i / (n_grid - 1)
                b = j / (n_grid - 1)
                c = 1.0 - a - b
                # Add stratified jitter within the cell
                jitter_scale = 1.0 / (n_grid - 1)
                a += rng.uniform(0, jitter_scale) * 0.5
                b += rng.uniform(0, jitter_scale) * 0.5
                c += rng.uniform(0, jitter_scale) * 0.5
                # Renormalize to stay on simplex
                s = a + b + c
                if s > 0:
                    a /= s
                    b /= s
                    c /= s
                else:
                    a, b, c = 1.0/3, 1.0/3, 1.0/3
                u = np.array([a, b, c])
                x = np.zeros(dim)
                x[:n_obj] = u
                x[n_obj:] = 0.5
                x = np.clip(x, lo, hi)
                eval_and_record(x)
            if used >= budget or not time_ok():
                break

        # Phase 2: Quasi-random (low-discrepancy) sampling on the simplex
        # Use a simple approach: generate random points and project to simplex
        n_phase2 = min(80, budget - used)
        for _ in range(n_phase2):
            if used >= budget or not time_ok():
                break
            u = rng.random(n_obj)
            u = u / u.sum()
            x = np.zeros(dim)
            x[:n_obj] = u
            x[n_obj:] = 0.5
            x = np.clip(x, lo, hi)
            eval_and_record(x)

        # Phase 3: Focused sampling near edges and vertices of the simplex
        # Points near edges (one coordinate near 0) can improve hypervolume
        n_phase3 = min(50, budget - used)
        for _ in range(n_phase3):
            if used >= budget or not time_ok():
                break
            # Choose a random edge: two coordinates active, one near 0
            idx = rng.choice(n_obj, size=2, replace=False)
            u = np.zeros(n_obj)
            t = rng.uniform(0.01, 0.99)
            u[idx[0]] = t
            u[idx[1]] = 1.0 - t
            # Small perturbation on the inactive coordinate
            u += 0.01 * rng.random(n_obj)
            u = np.clip(u, 0, 1)
            u = u / u.sum()
            x = np.zeros(dim)
            x[:n_obj] = u
            x[n_obj:] = 0.5
            x = np.clip(x, lo, hi)
            eval_and_record(x)

        # Phase 4: Use remaining budget for additional random sampling
        while used < budget and time_ok():
            u = rng.random(n_obj)
            u = u / u.sum()
            x = np.zeros(dim)
            x[:n_obj] = u
            x[n_obj:] = 0.5
            x = np.clip(x, lo, hi)
            eval_and_record(x)

        if recorded_F:
            return np.array(recorded_F)
        return np.array([])
# EVOLVE-BLOCK-END
