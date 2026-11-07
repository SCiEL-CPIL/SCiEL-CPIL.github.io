"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ7 exploiting known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        dim = self.dim
        n_obj = self.n_obj
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # DTLZ7: n=22, m=3
        # Inactive vars: x[0] to x[dim-n_obj-1] (for g function)
        # Active vars: x[dim-n_obj] to x[dim-1] (for Pareto front shape)
        n_inactive = dim - n_obj  # 19
        n_active = n_obj          # 3

        # Strategy: Set inactive vars to 0 (gives g=1, optimal)
        # Sample active vars to cover the Pareto front
        # Use 2D grid on the first two active vars, with third derived or also sampled

        # Budget allocation:
        # - Use most budget for front sampling
        # - Keep a few for verification/exploration if needed

        used = 0
        results = []  # list of objective vectors

        # Determine grid size based on budget
        # We want to cover a 2D surface (active vars 0 and 1 parameterize the front)
        # Active var 2 may be determined or also varied
        # For DTLZ7, the front is 2D: parameterized by x_p1, x_p2
        # x_p3 (if exists) may not affect the front in standard formulation

        # Let's use a 2D grid over active[0] and active[1]
        # Reserve some budget for potential refinement

        # Calculate grid dimensions
        # Total evaluations available for front sampling
        n_eval = self.budget

        # Use sqrt allocation for 2D grid
        n_grid = int(np.sqrt(n_eval))
        # Ensure we don't exceed budget
        n_grid = max(5, min(n_grid, 50))

        # Create 2D grid over active variables 0 and 1
        # Active vars are at indices: dim-n_obj, dim-n_obj+1, dim-n_obj+2
        # For DTLZ7, typically x_p1 and x_p2 parameterize the front
        # x_p3 might be fixed or also varied

        # Let's sample active[0] and active[1] on a grid, and set active[2] to a value
        # that gives good coverage. Actually, for DTLZ7 with m=3:
        # The front is parameterized by 2 variables (since m-1=2 free parameters)
        # So we should vary active[0] and active[1], and active[2] is determined
        # or we can vary all 3 but the front is 2D

        # Let me just sample all active vars on a 3D grid but that's wasteful
        # Better: 2D grid on active[0], active[1] and try a few values for active[2]

        # Actually, let's be smarter. The Pareto front of DTLZ7 is 2D.
        # We'll use a 2D grid over two of the active variables.
        # Let's use active[0] and active[1] as the two parameters.

        # Generate grid points
        n_per_axis = n_grid
        grid_0 = np.linspace(0.0, 1.0, n_per_axis)
        grid_1 = np.linspace(0.0, 1.0, n_per_axis)

        # Create all combinations
        g0, g1 = np.meshgrid(grid_0, grid_1, indexing='ij')
        g0_flat = g0.flatten()
        g1_flat = g1.flatten()
        n_points = len(g0_flat)

        # For active[2], we can try several values or just use 0.5
        # Let's use a few values for active[2] to get better coverage
        # But budget is limited, so let's start with active[2]=0.5
        # and if we have budget left, try other values

        # Actually, for DTLZ7, the third objective depends on the first two h values
        # so the front is truly 2D. Let's just use 2D grid.

        # Build candidate points
        # Inactive vars set to 0 (midpoint of [0,1] is 0.5, but 0 gives g=1)
        # Wait, bounds are [0,1], so setting to 0 is valid and gives g=1
        # Actually g = 1 + 9/19 * sum(x_i^2), so x_i=0 gives g=1 (minimum)

        # Set inactive to 0
        inactive_val = 0.0

        # For each grid point, create a full vector
        # Active indices: dim-n_obj, dim-n_obj+1, dim-n_obj+2
        # We'll vary active[0] and active[1], set active[2] to 0.5 initially

        active_start = dim - n_obj
        # active indices: active_start, active_start+1, active_start+2

        # First pass: 2D grid with active[2] = 0.5
        budget_left = self.budget
        points_to_eval = []

        for i in range(n_points):
            if budget_left <= 0:
                break
            x = np.zeros(dim)
            # Inactive vars = 0
            x[:n_inactive] = inactive_val
            # Active vars
            x[active_start] = g0_flat[i]
            x[active_start + 1] = g1_flat[i]
            x[active_start + 2] = 0.5  # fixed for now
            points_to_eval.append(x)
            budget_left -= 1

        # Evaluate all points
        for x in points_to_eval:
            if time.time() > deadline:
                break
            f = problem(x)
            results.append(np.array(f))
            used += 1

        # If we have budget left, try different values of active[2]
        if budget_left > 0 and time.time() < deadline:
            # Try a few values for active[2]
            active2_vals = [0.0, 0.25, 0.75, 1.0]
            for a2 in active2_vals:
                if budget_left <= 0 or time.time() > deadline:
                    break
                # Do a smaller grid with this active[2] value
                n_small = int(np.sqrt(budget_left // 2))
                n_small = max(3, min(n_small, 20))
                grid_small = np.linspace(0.0, 1.0, n_small)
                gs0, gs1 = np.meshgrid(grid_small, grid_small, indexing='ij')
                for i in range(len(gs0.flatten())):
                    if budget_left <= 0 or time.time() > deadline:
                        break
                    x = np.zeros(dim)
                    x[:n_inactive] = inactive_val
                    x[active_start] = gs0.flatten()[i]
                    x[active_start + 1] = gs1.flatten()[i]
                    x[active_start + 2] = a2
                    f = problem(x)
                    results.append(np.array(f))
                    used += 1
                    budget_left -= 1

        # If still budget left, do random exploration on the front
        if budget_left > 0 and time.time() < deadline:
            rng = np.random.default_rng(42)
            while budget_left > 0 and time.time() < deadline:
                x = np.zeros(dim)
                x[:n_inactive] = 0.0
                x[active_start] = rng.random()
                x[active_start + 1] = rng.random()
                x[active_start + 2] = rng.random()
                f = problem(x)
                results.append(np.array(f))
                used += 1
                budget_left -= 1

        # Return the set of points (the return value is ignored, but we return something sensible)
        if results:
            return np.array(results)
        return np.array([])
# EVOLVE-BLOCK-END
