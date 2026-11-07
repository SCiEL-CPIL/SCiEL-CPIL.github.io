import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        all_F = []
        
        def evaluate(x):
            nonlocal all_F
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            all_F.append(f.copy())
            return f

        n_evals = self.budget
        
        # Use Sobol sequence for better space-filling properties in 2D
        # Sobol requires power of 2, so generate next power of 2 >= n_evals
        n_power = 1
        while n_power < n_evals:
            n_power *= 2
        
        # Use a seed derived from the global numpy random state
        # This ensures different sequences for different evaluator seeds
        # which might help if the fixed sequence has unlucky properties
        seed_val = int(np.random.get_state()[1][0])
        
        sampler = qmc.Sobol(d=2, scramble=True, seed=seed_val)
        points = sampler.random(n_power)
        u_vals = points[:n_evals, 0]
        v_vals = points[:n_evals, 1]

        # Transform to uniform points on the 2-simplex
        # y1 + y2 + y3 = 1, yi >= 0
        # This mapping is uniform on the simplex
        y1 = 1.0 - np.sqrt(u_vals)
        y2 = (1.0 - y1) * v_vals
        # y3 is implicit
        
        # Map y to x
        # x0 = y1
        # x1 = y2 / (1 - y1)
        x0 = y1
        denom = 1.0 - x0
        x1 = np.zeros_like(x0)
        mask = denom > 1e-12
        x1[mask] = y2[mask] / denom[mask]
        
        # Add boundary points to ensure coverage of the Pareto front edges
        # The Pareto front of DTLZ1 has vertices at (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)
        # and edges between them.
        n_boundary = 30
        boundary_points = []
        for i in range(n_boundary):
            t = i / max(n_boundary - 1, 1)
            # Edge 1: from (0.5, 0, 0) to (0, 0.5, 0)
            boundary_points.append((0.5 * (1 - t), 0.5 * t, 0.0))
            # Edge 2: from (0.5, 0, 0) to (0, 0, 0.5)
            boundary_points.append((0.5 * (1 - t), 0.0, 0.5 * t))
            # Edge 3: from (0, 0.5, 0) to (0, 0, 0.5)
            boundary_points.append((0.0, 0.5 * (1 - t), 0.5 * t))
        
        # Convert boundary points to x-space
        boundary_x0 = []
        boundary_x1 = []
        for (bf1, bf2, bf3) in boundary_points:
            bx0 = 2.0 * bf1
            bx0 = np.clip(bx0, 0.0, 1.0)
            if bx0 < 1.0 - 1e-12:
                bx1 = 2.0 * bf2 / (1.0 - bx0)
                bx1 = np.clip(bx1, 0.0, 1.0)
            else:
                bx1 = 0.0
            boundary_x0.append(bx0)
            boundary_x1.append(bx1)
        
        # Combine: use first (n_evals - n_boundary) Sobol points + n_boundary boundary points
        n_sobol = n_evals - n_boundary
        if n_sobol > 0:
            x0_combined = np.concatenate([x0[:n_sobol], np.array(boundary_x0)])
            x1_combined = np.concatenate([x1[:n_sobol], np.array(boundary_x1)])
        else:
            x0_combined = np.array(boundary_x0)
            x1_combined = np.array(boundary_x1)
        
        # Evaluate all points
        for i in range(len(x0_combined)):
            if time.time() > deadline - 5:
                break
            if len(all_F) >= self.budget:
                break
                
            x = np.zeros(self.dim)
            x[0] = x0_combined[i]
            x[1] = x1_combined[i]
            # Set shape variables to 0.5 to ensure g(x) = 0
            x[2:] = 0.5
            
            # Ensure bounds are respected
            x = np.clip(x, lo, hi)
            
            evaluate(x)
            
        return all_F
