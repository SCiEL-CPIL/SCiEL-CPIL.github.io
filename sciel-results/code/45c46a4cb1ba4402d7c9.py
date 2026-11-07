"""Lattice-based deterministic grid search exploiting Rastrigin periodicity."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Lattice-based deterministic grid search for Rastrigin (f3).
    
    Strategy:
    1. Exploit the periodic structure of Rastrigin: f(x) = 10*n + sum((x_i - shift_i)^2 - 10*cos(2*pi*(x_i - shift_i)))
       The global minimum is near the shifted origin. Since the shift is hidden but the minimizer is in [-4,4]^5,
       and the function has local minima at integer-like offsets from the shift, we can systematically probe
       integer and half-integer grid points to find the global basin.
       
    2. Phase 1: Coarse lattice search over integer and half-integer coordinates in [-5, 5]^5.
       This gives a structured set of candidate points that align with the periodicity of cos(2*pi*x).
       We evaluate these points to find the best basin.
       
    3. Phase 2: Local refinement using a (1+1)-ES starting from the best point found in Phase 1.
       This refines the solution within the identified basin.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Generate lattice points: integer and half-integer coordinates in [-5, 5]
        # Coordinates: -5, -4.5, -4, ..., 4.5, 5 -> 21 points per dimension
        # Total points: 21^5 = 4,084,101 which is too many for budget=500
        
        # Instead, use a smarter approach:
        # Phase 1: Evaluate a subset of lattice points. Since budget is 500, we can evaluate at most 500 points.
        # We'll use a space-filling or structured subset.
        
        # Better approach: Use a two-phase strategy.
        # Phase 1 (first ~100 calls): Evaluate integer grid points (11^5 = 161,051 is still too many).
        # So we need to be more selective.
        
        # Alternative: Since the minimizer is in [-4, 4]^5, and Rastrigin has local minima at each integer point
        # relative to the shift, the global minimum is at the shift point. The shift is hidden but fixed per instance.
        
        # Key insight: For Rastrigin, the function value at a point x is:
        # f(x) = 10*dim + sum_i [(x_i - s_i)^2 - 10*cos(2*pi*(x_i - s_i))]
        # where s is the hidden shift.
        # The minimum is at x = s, with f_opt = -10*dim + 10*dim = 0? No:
        # At x_i = s_i: (0)^2 - 10*cos(0) = -10, so sum = -10*dim, plus 10*dim = 0.
        # But the problem says f_opt is in [-100, 100], so there's an offset.
        
        # Let's use a practical approach:
        # 1. Start with a coarse grid of points that are likely to be near local minima.
        #    For Rastrigin, local minima occur at integer values (relative to shift).
        #    Since we don't know the shift, we'll probe a structured set of points.
        
        # Phase 1: Evaluate 200 points on a structured grid.
        # Use coordinates from {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5} but only a subset.
        # Actually, let's use a Latin hypercube or a structured subset.
        
        # Simpler: Use a (1+1)-ES with a good initialization strategy.
        # Initialize at 0 (center), then use the ES to refine.
        # But the directive says to use lattice-based deterministic grid search.
        
        # Let's do this:
        # Phase 1: Evaluate points on a coarse grid with spacing 1.0 in the range [-4, 4]^5.
        # That's 9^5 = 59,049 points, still too many.
        
        # With budget 500, we can only evaluate 500 points. Let's be strategic.
        
        # Phase 1 (250 calls): Evaluate a set of 250 points sampled from the lattice.
        # Use a space-filling curve (e.g., Sobol or just a structured subset).
        
        # Actually, let's use a simpler approach that exploits the periodicity:
        # The Rastrigin function has the property that f(x) depends on the distance to the nearest integer
        # in each dimension (relative to the shift). So points that are close to integer values in each
        # dimension (relative to the shift) will have low values.
        
        # Since we don't know the shift, we can probe points that are at integer coordinates and
        # half-integer coordinates. The best among these will be close to the global minimum.
        
        # Let's generate a set of candidate points:
        # For each dimension, use coordinates: -4, -3, -2, -1, 0, 1, 2, 3, 4 (9 values)
        # But 9^5 = 59,049 is too many.
        
        # With 500 budget, let's use:
        # Phase 1: 100 random points from the lattice (integer coords in [-4,4]^5)
        # Phase 2: 400 calls for (1+1)-ES refinement from the best point.
        
        # But the directive says "deterministic grid search". Let's use a deterministic subset.
        
        # Use a Z-order (Morton) code to select a space-filling subset of the lattice.
        
        # Generate all integer lattice points in [-4, 4]^5
        coords = np.arange(-4, 5, 1)  # -4, -3, ..., 4
        # Create meshgrid
        grids = np.meshgrid(*([coords] * dim), indexing='ij')
        lattice = np.stack([g.ravel() for g in grids], axis=-1)  # shape (9^5, 5)
        
        # 9^5 = 59,049 points. We need to select 200 of them deterministically.
        # Use a stride to select a space-filling subset.
        n_points = len(lattice)
        n_select = min(200, n_points)
        
        # Select every k-th point to get a space-filling subset
        k = n_points // n_select
        if k < 1:
            k = 1
        selected_indices = np.arange(0, n_points, k)[:n_select]
        if len(selected_indices) < n_select:
            # Pad with additional points
            remaining = np.setdiff1d(np.arange(n_points), selected_indices)
            extra = remaining[:n_select - len(selected_indices)]
            selected_indices = np.concatenate([selected_indices, extra])
        
        selected_points = lattice[selected_indices]
        
        # Phase 1: Evaluate selected lattice points
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        for i in range(len(selected_points)):
            if calls_made >= budget - 100:  # Reserve 100 calls for refinement
                break
            x = selected_points[i]
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: Local refinement using (1+1)-ES from best_x
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1
        
        rng = np.random.default_rng(0)
        x = best_x
        fx = best_f
        sigma = 0.5  # Start with smaller sigma since we're already near a local minimum
        
        remaining_budget = budget - calls_made
        for _ in range(remaining_budget):
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
            else:
                sigma *= 0.9
            if sigma < 1e-10:
                break
        
        return x, fx
# EVOLVE-BLOCK-END
