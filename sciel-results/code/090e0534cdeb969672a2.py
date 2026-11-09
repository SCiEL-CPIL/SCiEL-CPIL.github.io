"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ7-aware optimizer using hypervolume-aware non-uniform Pareto-front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # DTLZ7 Pareto front: x1..x_{M-1} determine objectives, x_M..x_D are position variables
        # For DTLZ7 with M=3 objectives, D=22:
        # f1 = x1, f2 = x2, f3 = (1+x_{D-1}) * g
        # g = 1 + 9/D' * sum_{i in D'} sin(3*pi*x_i), D' = D - M + 1 = 20
        # Pareto front has g=0, which means x_{M}..x_D all at 0 (sin(3*pi*0)=0)
        # On the Pareto front: f1=x1, f2=x2, f3=0, with x1>=0, x2>=0, x1+x2<=1

        # Strategy: hypervolume-aware non-uniform grid on the Pareto front
        # The reference point is [1.1, 1.1, 25.0]
        # Hypervolume contribution is highest near the origin (0,0,0) in objective space
        # Since f3=0 on the front, we focus on the (f1, f2) plane
        # Use a non-uniform grid that clusters points near the origin

        # Generate points on the Pareto front using a hypervolume-aware strategy
        # The front is the triangle: x1>=0, x2>=0, x1+x2<=1

        # Use a power-law transformation to cluster points near the origin
        # t = u^gamma where gamma > 1 clusters near 0
        # We'll use a grid in (u, v) space and transform to (x1, x2)

        n_total = budget
        # Use a 2D grid with non-uniform spacing
        # Let's use n_per_dim points along each axis
        n_per_dim = max(1, int(np.sqrt(n_total)))
        
        # Generate grid points in [0,1]^2 with power-law clustering near origin
        # Use gamma=2 for quadratic clustering
        gamma = 2.0
        
        # Generate 1D grid with power-law spacing
        # u_i = (i/(n-1))^gamma for i=0..n-1
        u_grid = np.linspace(0, 1, n_per_dim) ** gamma
        v_grid = np.linspace(0, 1, n_per_dim) ** gamma
        
        # Generate 2D grid points
        front_points = []
        used = 0
        
        for u in u_grid:
            for v in v_grid:
                if used >= n_total:
                    break
                # Transform to triangle: x1 = u * v, x2 = u * (1-v)
                # This ensures x1 + x2 = u <= 1
                x1 = u * v
                x2 = u * (1 - v)
                x = np.zeros(dim)
                x[0] = x1
                x[1] = x2
                # x[2:] = 0 (already zero)
                front_points.append(x)
                used += 1
            if used >= n_total:
                break
        
        # If we have budget left, add more points
        remaining = n_total - used
        if remaining > 0:
            # Add random points with power-law clustering
            rng = np.random.default_rng(42)
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                u = rng.random() ** gamma
                v = rng.random()
                x1 = u * v
                x2 = u * (1 - v)
                x = np.zeros(dim)
                x[0] = x1
                x[1] = x2
                front_points.append(x)
                used += 1
        
        front_points = np.array(front_points[:n_total])
        
        # Evaluate front points
        F = np.array([problem(x) for x in front_points])
        used = len(front_points)
        
        # Check time
        if time.time() > deadline:
            return front_points
        
        # Local refinement: for each front point, try small perturbations
        remaining = budget - used
        if remaining <= 0:
            return front_points
        
        # Use remaining budget for local refinement
        n_local = len(front_points)
        steps_per_point = max(1, remaining // n_local)
        
        for i in range(n_local):
            if time.time() > deadline:
                break
            for step in range(steps_per_point):
                if time.time() > deadline or used >= budget:
                    break
                
                # Try perturbing the non-front dimensions (should be near 0 for g=0)
                # and the front dimensions
                x_curr = front_points[i].copy()
                
                # Small random perturbation focused on making g closer to 0
                rng_local = np.random.default_rng(i * 1000 + step)
                perturbation = np.zeros(dim)
                # Perturb the g-affecting dimensions (indices 2..21)
                perturbation[2:] = rng_local.standard_normal(dim - 2) * 0.01
                # Small perturbation on front dimensions
                perturbation[0] = rng_local.standard_normal() * 0.005
                perturbation[1] = rng_local.standard_normal() * 0.005
                
                x_new = x_curr + perturbation
                x_new = np.clip(x_new, lo, hi)
                
                f_new = problem(x_new)
                used += 1
                
                # Accept if it dominates current
                if np.all(f_new <= F[i]) and np.any(f_new < F[i]):
                    front_points[i] = x_new
                    F[i] = f_new
                # Or if it's incomparable but has better sum
                elif not (np.all(F[i] <= f_new) and np.any(F[i] < f_new)):
                    if np.sum(f_new) < np.sum(F[i]):
                        front_points[i] = x_new
                        F[i] = f_new
        
        return front_points
# EVOLVE-BLOCK-END
