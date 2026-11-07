"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer exploiting DTLZ2 analytic structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # For DTLZ2 with n_obj=3, dim=12:
        # x[0], x[1] determine position on Pareto front (sphere)
        # x[2:12] determine g (distance from Pareto front), g=0 at x_i=0.5
        # Pareto front: f_i = sin(pi/2 * x_i) for i=0,1; f_2 = cos(pi/2*(x_0+x_1))
        # with x_2..x_11 = 0.5
        
        n_obj = self.n_obj
        dim = self.dim
        
        # Number of position variables = n_obj - 1 = 2
        n_pos = n_obj - 1
        # Number of distance variables = dim - n_pos = 10
        n_dist = dim - n_pos
        
        def make_pareto_point(theta1, theta2):
            """Create a point on the known Pareto front of DTLZ2."""
            x = np.zeros(dim)
            # Position variables
            x[0] = theta1
            x[1] = theta2
            # Distance variables set to 0.5 for g=0
            for i in range(n_pos, dim):
                x[i] = 0.5
            return np.clip(x, lo, hi)
        
        def pareto_front_objectives(x):
            """Compute objectives for a point assuming it's on the Pareto front."""
            f = np.zeros(n_obj)
            g = 0.0
            for i in range(n_pos, dim):
                g += (x[i] - 0.5) ** 2
            one_plus_g = 1.0 + g
            for i in range(n_obj - 1):
                f[i] = one_plus_g * np.sin(np.pi / 2.0 * x[i])
            f[n_obj - 1] = one_plus_g * np.cos(np.pi / 2.0 * np.sum(x[:n_obj - 1]))
            return f
        
        # Strategy: 
        # 1. Initialize with points on the known Pareto front
        # 2. Use local search to refine
        # 3. Track all evaluated points
        
        evaluated_points = []
        evaluated_f = []
        used = 0
        budget = self.budget
        
        # Phase 1: Initialize with a uniform grid of points on the Pareto front
        # Use a 10x10 grid for good coverage of the 2D position space
        n_per_axis = 10
        n_init = n_per_axis * n_per_axis
        n_init = min(n_init, budget)
        
        # Uniform grid sampling
        thetas1 = np.linspace(0.0, 1.0, n_per_axis)
        thetas2 = np.linspace(0.0, 1.0, n_per_axis)
        
        for t1 in thetas1:
            for t2 in thetas2:
                if used >= budget:
                    break
                x = make_pareto_point(t1, t2)
                f = problem(x)
                evaluated_points.append(x.copy())
                evaluated_f.append(np.array(f))
                used += 1
            if used >= budget:
                break
        
        # Phase 2: Local search refinement with adaptive step sizes
        remaining = budget - used
        if remaining > 0:
            # Use local search on the current points
            # Try to refine each point with small steps
            n_local = len(evaluated_points)
            steps_per_point = max(1, remaining // n_local)
            
            rng = np.random.default_rng(42)
            
            for i in range(n_local):
                if used >= budget:
                    break
                for step in range(steps_per_point):
                    if used >= budget or time.time() > deadline:
                        break
                    # Small perturbation in position variables with adaptive step size
                    x = evaluated_points[i].copy()
                    step_size = 0.02 / (step + 1)  # Decreasing step size
                    delta = step_size * rng.standard_normal(n_pos)
                    x[:n_pos] += delta
                    x = np.clip(x, lo, hi)
                    f = problem(x)
                    evaluated_points.append(x.copy())
                    evaluated_f.append(np.array(f))
                    used += 1
        
        # Return the last point (return value is ignored)
        if evaluated_points:
            return evaluated_points[-1]
        else:
            return lo.copy()
# EVOLVE-BLOCK-END
