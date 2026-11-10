# EVOLVE-BLOCK-START
import numpy as np
import time

def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares inside the unit square [0,1]^2 to maximize
    the sum of their side lengths, with no overlapping interiors.
    
    Returns:
        centers: np.ndarray of shape (10, 2)
        sides:   np.ndarray of shape (10,)
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 600  # safety margin
    
    def validate(centers, sides, tol=1e-7):
        """Check if the packing is valid."""
        if centers.shape != (n, 2) or sides.shape != (n,):
            return False
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if not np.all(sides > 0):
            return False
        # Check boundaries
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                    return False
        return True
    
    def objective(x):
        """Minimize negative sum of sides (we maximize sum). x encodes centers and sides."""
        # x: [cx0, cy0, s0, cx1, cy1, s1, ...] for n squares
        centers = x[0::3].reshape(n, 2)
        sides = x[1::3]
        # Penalize violations heavily
        penalty = 0.0
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if s <= 0:
                penalty += 100 * (0.1 - s)
            if cx - s/2 < 0:
                penalty += 100 * (0 - (cx - s/2))
            if cx + s/2 > 1:
                penalty += 100 * ((cx + s/2) - 1)
            if cy - s/2 < 0:
                penalty += 100 * (0 - (cy - s/2))
            if cy + s/2 > 1:
                penalty += 100 * ((cy + s/2) - 1)
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                if dx < t and dy < t:
                    # Overlap penalty
                    overlap_area = max(0, t - dx) * max(0, t - dy)
                    penalty += 100 * overlap_area
        return -np.sum(sides) + penalty
    
    # Try Differential Evolution
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    # Initial guess: 3x3 + 1 corner packing
    s_base = 1.0 / 3.0 - 1e-4
    centers_init = np.zeros((n, 2))
    sides_init = np.zeros(n)
    for i in range(9):
        col = i % 3
        row = i // 3
        centers_init[i] = [(col + 0.5) * s_base, (row + 0.5) * s_base]
        sides_init[i] = s_base
    # 10th square in corner
    centers_init[9] = [s_base + 0.005, s_base + 0.005]
    sides_init[9] = 0.01
    
    if validate(centers_init, sides_init):
        best_centers = centers_init.copy()
        best_sides = sides_init.copy()
        best_score = np.sum(sides_init)
    
    # Also try 4x4 grid as fallback
    k = 4
    side_grid = 1.0 / k
    centers_grid = np.zeros((n, 2))
    sides_grid = np.full(n, side_grid)
    for i in range(n):
        col = i % k
        row = i // k
        centers_grid[i] = [(col + 0.5) * side_grid, (row + 0.5) * side_grid]
    if validate(centers_grid, sides_grid):
        score_grid = np.sum(sides_grid)
        if score_grid > best_score:
            best_centers = centers_grid.copy()
            best_sides = sides_grid.copy()
            best_score = score_grid
    
    # Run Differential Evolution if time permits
    if time.time() < deadline - 120:  # leave 2 min margin
        try:
            from scipy.optimize import differential_evolution
            from numpy.random import default_rng
            
            rng = default_rng(42)
            
            # Bounds: centers in [0, 1], sides in [0.01, 0.5]
            bounds = []
            for i in range(n):
                bounds.append((0.0, 1.0))  # cx
                bounds.append((0.0, 1.0))  # cy
                bounds.append((0.01, 0.5))  # side
            
            x_init = np.concatenate([
                centers_init.ravel(),
                sides_init
            ])
            # Reorder: x = [cx0, cy0, s0, cx1, cy1, s1, ...]
            x_init_reordered = np.zeros(3 * n)
            for i in range(n):
                x_init_reordered[3*i] = centers_init[i, 0]
                x_init_reordered[3*i + 1] = centers_init[i, 1]
                x_init_reordered[3*i + 2] = sides_init[i]
            
            result = differential_evolution(
                objective,
                bounds,
                x0=x_init_reordered,
                maxiter=200,
                popsize=20,
                tol=1e-6,
                mutation=(0.5, 1.5),
                recombination=0.7,
                seed=42,
                polish=True
            )
            
            if result.x is not None:
                x = result.x
                centers_de = x[0::3].reshape(n, 2)
                sides_de = x[1::3]
                
                # Clamp to valid range
                sides_de = np.clip(sides_de, 0.01, 0.5)
                centers_de[:, 0] = np.clip(centers_de[:, 0], 0, 1)
                centers_de[:, 1] = np.clip(centers_de[:, 1], 0, 1)
                
                if validate(centers_de, sides_de):
                    score_de = np.sum(sides_de)
                    if score_de > best_score:
                        best_centers = centers_de
                        best_sides = sides_de
                        best_score = score_de
        except Exception:
            pass
    
    # If we still have the initial or grid solution, return it
    if best_centers is None:
        # Absolute fallback: 4x4 grid
        k = 4
        side_grid = 1.0 / k
        best_centers = np.zeros((n, 2))
        best_sides = np.full(n, side_grid)
        for i in range(n):
            col = i % k
            row = i // k
            best_centers[i] = [(col + 0.5) * side_grid, (row + 0.5) * side_grid]
    
    return best_centers.astype(np.float64), best_sides.astype(np.float64)

# EVOLVE-BLOCK-END
