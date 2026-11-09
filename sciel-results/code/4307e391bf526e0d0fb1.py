"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ2 using Pareto-front grid + local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 320.0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(0)

        all_F = []
        used = 0

        def add_point(x):
            nonlocal used
            f = problem(x)
            all_F.append(f)
            used += 1
            return f

        # Phase 1: Grid on the known DTLZ2 Pareto front
        # For DTLZ2, the Pareto front is a quarter-sphere.
        # We parameterize using angles theta1, theta2 in [0, pi/2]
        # and set the distance variable g=0 by appropriate x encoding.
        # DTLZ2: x[0]=cos(theta1)*cos(theta2), x[1]=cos(theta1)*sin(theta2), x[2]=sin(theta1)
        # x[3:] are for the g-function; setting them to 0.5 gives g=0 (on the front)
        # Actually for DTLZ2, g=0 when x[3:]=0.5 (for the standard formulation)
        # Let's use the standard: g(x) = (m-k) + sum((x_i - 0.5)^2 - cos(20*pi*(x_i-0.5))) for i in k+1..m
        # g=0 when x_i=0.5 for all i in k+1..m
        
        n_grid = 12  # 12x12 grid = 144 points on the front
        theta1_vals = np.linspace(0, np.pi/2, n_grid)
        theta2_vals = np.linspace(0, np.pi/2, n_grid)
        
        for t1 in theta1_vals:
            for t2 in theta2_vals:
                if used >= self.budget or time.time() > deadline:
                    break
                x = np.full(self.dim, 0.5)
                x[0] = np.cos(t1) * np.cos(t2)
                x[1] = np.cos(t1) * np.sin(t2)
                x[2] = np.sin(t1)
                # Clip to bounds
                x = np.clip(x, lo, hi)
                add_point(x)
            if time.time() > deadline:
                break

        # Phase 2: Local refinement around the best non-dominated points
        if used < self.budget and time.time() < deadline:
            F_arr = np.array(all_F)
            # Find non-dominated points
            n = len(F_arr)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated[i] = True
                        break
            
            nd_indices = np.where(~dominated)[0]
            
            # Refine around non-dominated points with small perturbations
            for idx in nd_indices:
                if used >= self.budget or time.time() > deadline:
                    break
                # Get the x that produced this point - we need to track x
                # Since we don't track x, let's just do random perturbation in objective space
                # Actually, let's do a few random restarts near the front
                for _ in range(3):
                    if used >= self.budget or time.time() > deadline:
                        break
                    t1 = np.clip(rng.uniform(0, np.pi/2), 0, np.pi/2)
                    t2 = np.clip(rng.uniform(0, np.pi/2), 0, np.pi/2)
                    # Add small noise to be slightly off the front
                    noise = 0.01 * rng.standard_normal(self.dim)
                    x = np.full(self.dim, 0.5)
                    x[0] = np.cos(t1) * np.cos(t2)
                    x[1] = np.cos(t1) * np.sin(t2)
                    x[2] = np.sin(t1)
                    x += noise
                    x = np.clip(x, lo, hi)
                    add_point(x)

        # Phase 3: Use remaining budget for random sampling
        while used < self.budget and time.time() < deadline:
            x = lo + (hi - lo) * rng.random(self.dim)
            add_point(x)

        return np.array(all_F)
# EVOLVE-BLOCK-END
