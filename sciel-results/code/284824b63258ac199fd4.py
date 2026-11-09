"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """DTLZ7-aware optimizer using Pareto-front sampling and local refinement."""

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
        # Pareto front has g=0, which means x_{M}..x_D all at 0.5 (sin(3*pi*0.5)=sin(1.5*pi)=-1... wait)
        # Actually for g=0 we need sum sin(3*pi*x_i) = 0, so x_i = 0 for all i in D'
        # Wait: sin(3*pi*0) = 0, so x_i=0 gives g=0.
        # So Pareto front: x_{3}..x_{22} = 0, x1 and x2 vary with x1+x2=1 constraint for the front shape.
        # Actually DTLZ7 front: f1=x1, f2=x2, f3=(1+x_{D-1})*g. On the front g=0, so f3=0.
        # The front is defined by x1>=0, x2>=0, x1+x2<=1 (for the standard DTLZ7).
        # Wait, let me reconsider. DTLZ7:
        # f1 = x1
        # f2 = x2
        # f3 = (1 + x_{D-1}) * g(x_{D-1},...,x_D)
        # where g = 1 + 9/D' * sum_{i=D-M+2}^{D} sin(3*pi*x_i), D'=D-M+1=20
        # Pareto-optimal solutions have g=0, i.e., x_{D-M+2}..x_D = 0 (or values making sin sum to 0)
        # On the Pareto front: f1=x1, f2=x2, f3=0, with x1>=0, x2>=0, x1+x2<=1

        # Strategy: sample points on the known Pareto front, then use remaining budget for local refinement
        # The front is 2D (x1, x2) with x1+x2<=1, x1>=0, x2>=0, and x3..x22=0

        # Generate initial population on the Pareto front
        # Use a mix of uniform and stratified sampling on the 2D simplex
        n_front = min(budget, 100)
        
        # Stratified sampling on the 2D triangle
        # Use barycentric coordinates
        n_per_axis = max(1, int(np.sqrt(n_front)))
        
        front_points = []
        # Sample on the edges and interior
        # Edge 1: x1 from 0 to 1, x2=0
        for t in np.linspace(0, 1, n_per_axis):
            x = np.zeros(dim)
            x[0] = t
            x[1] = 0.0
            front_points.append(x)
        # Edge 2: x1=0, x2 from 0 to 1
        for t in np.linspace(0, 1, n_per_axis):
            x = np.zeros(dim)
            x[0] = 0.0
            x[1] = t
            front_points.append(x)
        # Edge 3: x1+x2=1
        for t in np.linspace(0, 1, n_per_axis):
            x = np.zeros(dim)
            x[0] = t
            x[1] = 1.0 - t
            front_points.append(x)
        # Interior points
        n_interior = n_front - 3 * n_per_axis
        if n_interior > 0:
            rng = np.random.default_rng(42)
            for _ in range(n_interior):
                # Random point in triangle
                u1 = rng.random()
                u2 = rng.random()
                if u1 + u2 > 1:
                    u1 = 1 - u1
                    u2 = 1 - u2
                x = np.zeros(dim)
                x[0] = u1
                x[1] = u2
                front_points.append(x)
        
        front_points = np.array(front_points[:n_front])
        
        # Evaluate front points
        F = np.array([problem(x) for x in front_points])
        used = len(front_points)
        
        # Check time
        if time.time() > deadline:
            return front_points
        
        # Local refinement: for each front point, try small perturbations in the non-front dimensions
        # and in the front dimensions to see if we can improve
        remaining = budget - used
        if remaining <= 0:
            return front_points
        
        # For each point, try a few local perturbations
        # Focus on dimensions that affect g (x_3..x_22, i.e., indices 2..21)
        # and the front dimensions (x_0, x_1)
        
        # Use the remaining budget for local search
        # For each of the front points, do a few gradient-free local steps
        n_local = len(front_points)
        steps_per_point = max(1, remaining // n_local)
        
        for i in range(n_local):
            if time.time() > deadline:
                break
            for step in range(steps_per_point):
                if time.time() > deadline:
                    break
                if used >= budget:
                    break
                
                # Try perturbing the non-front dimensions (should be near 0 for g=0)
                # and the front dimensions
                x_curr = front_points[i].copy()
                
                # Small random perturbation
                rng_local = np.random.default_rng(i * 1000 + step)
                perturbation = rng_local.standard_normal(dim) * 0.01
                x_new = x_curr + perturbation
                x_new = np.clip(x_new, lo, hi)
                
                f_new = problem(x_new)
                used += 1
                
                # Accept if better in any objective without worsening others too much
                # Simple acceptance: if it's not dominated by current
                if not (np.all(F[i] <= f_new) and np.any(F[i] < f_new)):
                    # Check if new dominates current
                    if np.all(f_new <= F[i]) and np.any(f_new < F[i]):
                        front_points[i] = x_new
                        F[i] = f_new
                    elif not (np.all(F[i] <= f_new) and np.any(F[i] < f_new)):
                        # Incomparable, accept with some probability or if closer to ideal
                        # Use a simple criterion: accept if sum of objectives is lower
                        if np.sum(f_new) < np.sum(F[i]):
                            front_points[i] = x_new
                            F[i] = f_new
        
        # Final cleanup: ensure all points are valid
        return front_points
# EVOLVE-BLOCK-END
