# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.

    Uses a differentiable density-field approach: each square's side is derived
    from local repulsion to other squares and boundaries. We optimize positions
    to maximize the sum of sides computed from smooth distance-based potentials.
    """
    n = 10
    tol = 1e-7

    def compute_sides_from_positions(pos, sigma=0.1):
        """
        Given positions (n, 2), compute side lengths based on local minimum
        distance to other squares and boundaries using a smooth kernel.
        
        For each square i, the available space is limited by:
        - Distance to each other square j: min distance between edges
        - Distance to boundaries
        
        We use a soft version where side_i is proportional to the minimum
        of (distance to constraint) terms, but smoothed for differentiability.
    """
        cx = pos[:, 0]
        cy = pos[:, 1]
        sides = np.zeros(n)
        
        for i in range(n):
            # Distance to boundaries
            d_left = cx[i]
            d_right = 1.0 - cx[i]
            d_bottom = cy[i]
            d_top = 1.0 - cy[i]
            
            # For each other square j, compute the minimum distance that would
            # cause overlap if both had side s
            # Two squares i and j overlap if |cx_i - cx_j| < (s_i + s_j)/2
            # and |cy_i - cy_j| < (s_i + s_j)/2
            # So the constraint is: s_i + s_j <= 2 * min(|cx_i-cx_j|, |cy_i-cy_j|)
            # In the worst case (both have same side s), this becomes 2s <= 2*d => s <= d
            
            min_dist = min(d_left, d_right, d_bottom, d_top)
            
            for j in range(n):
                if i == j:
                    continue
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                # The limiting distance is min(dx, dy) for the sum of half-sides
                d_ij = min(dx, dy)
                min_dist = min(min_dist, d_ij)
            
            # Side length is bounded by 2 * min_dist (since s_i + s_j <= 2*d means
            # if s_i = s_j = s, then 2s <= 2*d => s <= d)
            # But we also need s_i <= 2 * d_boundary for each boundary
            # Actually, for boundary: cx - s/2 >= 0 => s <= 2*cx, and cx + s/2 <= 1 => s <= 2*(1-cx)
            # So s <= 2 * min(cx, 1-cx, cy, 1-cy)
            
            # For other squares: if s_i = s_j = s, then |cx_i - cx_j| >= s and |cy_i - cy_j| >= s
            # So s <= min(dx, dy) for each pair
            # Thus s_i <= min(2*min_boundary_dist, min_over_j(min(dx_ij, dy_ij)))
            
            sides[i] = min_dist
        
        return sides

    def objective(pos):
        """
        Negative of the sum of sides (we minimize negative to maximize sum).
        Add a small penalty for invalid configurations.
        """
        cx = pos[:, 0]
        cy = pos[:, 1]
        
        # Compute sides based on positions
        sides = np.zeros(n)
        for i in range(n):
            d_left = cx[i]
            d_right = 1.0 - cx[i]
            d_bottom = cy[i]
            d_top = 1.0 - cy[i]
            
            min_dist = min(d_left, d_right, d_bottom, d_top)
            
            for j in range(n):
                if i == j:
                    continue
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                d_ij = min(dx, dy)
                min_dist = min(min_dist, d_ij)
            
            sides[i] = min_dist
        
        total = np.sum(sides)
        
        # Penalty for positions outside [0, 1]
        penalty = 0.0
        for i in range(n):
            if cx[i] < 0 or cx[i] > 1 or cy[i] < 0 or cy[i] > 1:
                penalty += 100.0 * (abs(cx[i] - 0.5) + abs(cy[i] - 0.5))
        
        return -(total) + penalty

    # Initial guess: grid layout
    k = 4  # 4x4 grid, use first 10 cells
    side_init = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx_init = (cols + 0.5) * side_init
    cy_init = (rows + 0.5) * side_init
    x0 = np.column_stack([cx_init, cy_init]).flatten()
    
    # Try multiple random initializations and keep the best
    best_pos = x0.copy()
    best_val = objective(x0)
    
    # Grid-based initializations
    for k_try in [3, 4, 5]:
        side_init = 1.0 / k_try
        idx = np.arange(n)
        cols = idx % k_try
        rows = idx // k_try
        cx_init = (cols + 0.5) * side_init
        cy_init = (rows + 0.5) * side_init
        x_try = np.column_stack([cx_init, cy_init]).flatten()
        val = objective(x_try)
        if val < best_val:
            best_val = val
            best_pos = x_try.copy()
    
    # Random initializations
    np.random.seed(42)
    for _ in range(50):
        cx_rand = np.random.uniform(0.1, 0.9, n)
        cy_rand = np.random.uniform(0.1, 0.9, n)
        x_try = np.column_stack([cx_rand, cy_rand]).flatten()
        val = objective(x_try)
        if val < best_val:
            best_val = val
            best_pos = x_try.copy()
    
    # Optimize from the best starting point
    result = minimize(
        objective,
        best_pos,
        method='L-BFGS-B',
        bounds=[(0.0, 1.0)] * (2 * n),
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    if result.fun < best_val:
        best_pos = result.x
    
    # Final optimization with tighter settings
    result = minimize(
        objective,
        best_pos,
        method='L-BFGS-B',
        bounds=[(0.0, 1.0)] * (2 * n),
        options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
    )
    
    if result.fun < best_val:
        best_pos = result.x
    
    pos = best_pos.reshape(n, 2)
    cx = pos[:, 0]
    cy = pos[:, 1]
    
    # Compute final sides
    sides = np.zeros(n)
    for i in range(n):
        d_left = cx[i]
        d_right = 1.0 - cx[i]
        d_bottom = cy[i]
        d_top = 1.0 - cy[i]
        
        min_dist = min(d_left, d_right, d_bottom, d_top)
        
        for j in range(n):
            if i == j:
                continue
            dx = abs(cx[i] - cx[j])
            dy = abs(cy[i] - cy[j])
            d_ij = min(dx, dy)
            min_dist = min(min_dist, d_ij)
        
        sides[i] = min_dist
    
    # Ensure sides are positive and finite
    sides = np.maximum(sides, 1e-9)
    sides = np.where(np.isfinite(sides), sides, 1e-9)
    
    centers = pos.astype(np.float64)
    sides = sides.astype(np.float64)
    
    return centers, sides


# EVOLVE-BLOCK-END
