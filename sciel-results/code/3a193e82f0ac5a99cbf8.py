"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for DTLZ2."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
        n = self.dim
        m = self.n_obj
        budget = self.budget

        # DTLZ2 specific structure:
        # x[0:m] determines the position on the Pareto front (via sin/cos)
        # x[m:n] determines the distance from the front (g function)
        # We use a deterministic grid for the position variables to ensure coverage.
        
        # 1. Generate a grid of positions for the first m variables
        # For m=3, we need to cover the surface of a sphere (or the relevant hyper-surface).
        # A simple approach: sample x[0:m-1] from a grid in [0, pi/2] and derive x[m-1]?
        # Actually, DTLZ2 definition:
        # f_i = (1+g) * prod_{j=1}^{i-1} sin(x_j) * cos(x_i) for i < m
        # f_m = (1+g) * prod_{j=1}^{m-1} sin(x_j)
        # where x_j are in [0, pi/2].
        # Since bounds are [0,1], we map [0,1] to [0, pi/2].
        
        # Let's create a grid of size k x k for m=3 (2 independent variables effectively? 
        # No, DTLZ2 with m objectives uses m-1 variables for position? 
        # Standard DTLZ2: x_1..x_{m-1} are angles. x_m..x_n are for g.
        # Wait, standard DTLZ2 has m objectives. The position is determined by x_1..x_{m-1}.
        # If m=3, x_1, x_2 determine position. x_3..x_12 determine g.
        
        # Grid size
        grid_size = 10
        n_grid_points = grid_size ** (m - 1)
        
        # Ensure we don't exceed budget
        if n_grid_points > budget:
             grid_size = int(np.sqrt(budget))
             n_grid_points = grid_size ** (m - 1)
        
        # Generate grid points for x[0 : m-1]
        # Map [0, 1] to [0, pi/2]
        angles = np.linspace(0, np.pi/2, grid_size)
        
        grid_positions = []
        for a in angles:
            for b in angles:
                # For m=3, we have 2 angles. If m > 3, we'd need more dimensions.
                # Here m=3 is fixed by task.
                pos = np.array([a, b])
                grid_positions.append(pos)
        
        # If m was different, we'd need to handle higher dimensions, but task says n_obj=3.
        
        X = []
        F = []
        used = 0
        
        # 2. Initialize with grid points
        # For each grid position, set x[0:m-1] to the angle, and x[m-1:n] to 0.5 (midpoint)
        # This minimizes g? No, g = sum(x_i^2 - 10*cos(20*pi*x_i))^2.
        # The minimum of g is 0 when x_i = 0 or 1 (for cos term to be 1? No, cos(0)=1, so x=0 gives 0^2-10*1 = -10? No.
        # g = sum (x_i - x_i^*)^2 where x_i^* is the minimizer of (x^2 - 10cos(20pi x))^2.
        # The global minimum is 0 at x=0? 
        # Let's check: f(x) = x^2 - 10 cos(20 pi x). f(0) = -10. f'(x) = 2x + 200 pi sin(20 pi x).
        # There are many local minima. The global minimum is not necessarily at 0 or 1.
        # However, setting x to 0 is a good guess for some, but often 0.5 is safer or we can just use 0.
        # Actually, for DTLZ2, the Pareto front is achieved when g=0.
        # g=0 is achieved when x_i = 0 for all i? 
        # If x=0, term is (0 - 10*1)^2 = 100. Not 0.
        # If x=0.5, term is (0.25 - 10*cos(10pi))^2 = (0.25 - 10)^2 = 95.06.
        # The minimizer of (x^2 - 10 cos(20 pi x))^2 is not trivial.
        # But wait, the task says "bounds [0,1]".
        # Usually DTLZ2 is defined on [0,1].
        # Let's just use 0 for the g-variables initially, as it's a common heuristic, or random.
        # Actually, a better strategy: Set g-variables to 0. This might not be optimal, but we have budget to refine.
        
        for pos in grid_positions:
            x = np.zeros(n)
            x[0:m-1] = pos
            # Set remaining variables to 0
            # x[m-1:n] = 0.0 
            # Or maybe 0.5? Let's use 0.
            x[m-1:n] = 0.0
            
            X.append(x)
            F.append(problem(x))
            used += 1
            
            if used >= budget:
                break
            if time.time() > deadline:
                break

        # 3. Use remaining budget to refine the g-variables (x[m-1:n])
        # We will perform local searches on the g-variables to minimize g.
        # Since we have a grid of positions, we can try to improve the distance for each position.
        # Strategy: Random perturbation of g-variables.
        
        # Convert to arrays for easier manipulation
        X = np.array(X)
        F = np.array(F)
        
        # Track best point for each grid position? 
        # Or just maintain a population and do evolution on the g-variables?
        # Given the small budget left (200 - 100 = 100), we can do a simple hill climbing or random search.
        
        # Let's try to improve the points we have.
        # We will randomly pick a point, perturb its g-variables, and if it improves (reduces sum of objectives?), keep it.
        # But we want to maintain diversity.
        # Since the grid already provides diversity in position, we just need to minimize g for each position.
        
        # We can group points by their position (first m-1 vars).
        # For each group, we have one point initially.
        # We can use the remaining budget to do local search on these points.
        
        # Simple approach:
        # While budget remains:
        #   Pick a random point from X.
        #   Perturb its g-variables (x[m-1:n]).
        #   Evaluate.
        #   If the new point dominates the old one (or has lower sum of objectives), replace it.
        
        # To be safe and fast, we'll just do random restarts/local steps.
        
        while used < budget:
            if time.time() > deadline:
                break
            
            # Pick a random index to improve
            idx = rng.integers(len(X))
            x_parent = X[idx].copy()
            
            # Perturb g-variables (indices m-1 to n-1)
            # We can try different strategies:
            # 1. Small random step
            # 2. Reset to 0
            # 3. Random in [0,1]
            
            # Let's try a mix.
            if rng.random() < 0.5:
                # Small step
                step = rng.normal(0, 0.1, size=n - (m-1))
                x_child = x_parent.copy()
                x_child[m-1:n] += step
            else:
                # Random reset for g-variables
                x_child = x_parent.copy()
                x_child[m-1:n] = rng.random(n - (m-1))
            
            x_child = np.clip(x_child, lo, hi)
            
            f_child = problem(x_child)
            used += 1
            
            # Check if child is better
            # For DTLZ2, if position is fixed, lower g is better.
            # f = (1+g) * pos_vec. So minimizing sum(f) or any f is equivalent to minimizing g.
            # We compare using dominance.
            if np.all(f_child <= F[idx]) and np.any(f_child < F[idx]):
                X[idx] = x_child
                F[idx] = f_child
            elif rng.random() < 0.1: # Occasional random replacement to escape local minima
                X[idx] = x_child
                F[idx] = f_child

        return X
# EVOLVE-BLOCK-END
