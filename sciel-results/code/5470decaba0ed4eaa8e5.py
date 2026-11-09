# EVOLVE-BLOCK-START
import numpy as np


def squares_in_square10() -> tuple:
    """
    Packs 10 squares into the unit square to maximize the sum of their side lengths.
    
    Strategy:
    Uses a row-based packing with 4 rows.
    Row 1: 1 square (small)
    Row 2: 3 squares (large)
    Row 3: 3 squares (large)
    Row 4: 3 squares (large)
    
    This configuration allows the 9 large squares to be nearly 1/3 in size,
    and the 1 small square to fit in the remaining vertical space.
    The sum of side lengths approaches 3.0, which is significantly better
    than the 2.5 from a simple 4x4 grid.
    """
    n = 10
    
    # Define the packing parameters
    # We want 9 squares of side s_large and 1 square of side s_small.
    # Constraints:
    # 1. 3 * s_large <= 1  (width of rows with 3 squares)
    # 2. s_small <= 1      (width of row with 1 square)
    # 3. s_small + 3 * s_large <= 1 (total height)
    
    # To maximize 9 * s_large + s_small:
    # From (3), s_small = 1 - 3 * s_large.
    # Objective: 9 * s_large + (1 - 3 * s_large) = 6 * s_large + 1.
    # Max s_large is 1/3.
    # We choose s_large slightly less than 1/3 to ensure s_small > 0.
    
    s_large = 1.0 / 3.0 - 1e-6
    s_small = 1.0 - 3.0 * s_large
    
    # Verify s_small is positive
    if s_small <= 0:
        # Fallback to grid if something went wrong (should not happen)
        k = 4
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
        return centers, sides

    centers = np.zeros((n, 2), dtype=np.float64)
    sides = np.zeros(n, dtype=np.float64)
    
    # Row 4 (bottom): 3 squares of size s_large
    # y range: [0, s_large]
    y_base = s_large / 2.0
    x_centers = [s_large / 2.0, 3.0 * s_large / 2.0, 5.0 * s_large / 2.0]
    
    for i in range(3):
        centers[i, 0] = x_centers[i]
        centers[i, 1] = y_base
        sides[i] = s_large
        
    # Row 3: 3 squares of size s_large
    # y range: [s_large, 2 * s_large]
    y_base = s_large + s_large / 2.0
    for i in range(3):
        idx = 3 + i
        centers[idx, 0] = x_centers[i]
        centers[idx, 1] = y_base
        sides[idx] = s_large
        
    # Row 2: 3 squares of size s_large
    # y range: [2 * s_large, 3 * s_large]
    y_base = 2.0 * s_large + s_large / 2.0
    for i in range(3):
        idx = 6 + i
        centers[idx, 0] = x_centers[i]
        centers[idx, 1] = y_base
        sides[idx] = s_large
        
    # Row 1 (top): 1 square of size s_small
    # y range: [3 * s_large, 1]
    # Note: 3 * s_large + s_small = 1
    y_base = 3.0 * s_large + s_small / 2.0
    centers[9, 0] = 0.5
    centers[9, 1] = y_base
    sides[9] = s_small
    
    return centers, sides

# EVOLVE-BLOCK-END
