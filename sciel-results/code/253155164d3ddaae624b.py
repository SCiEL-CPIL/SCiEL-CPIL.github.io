# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def _is_valid(centers, sides, tol=1e-7):
    """Check if the configuration is valid."""
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # Check bounds
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol or cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i][0] - centers[j][0]) < t and abs(centers[i][1] - centers[j][1]) < t:
                return False
    return True

def _score(centers, sides):
    """Compute the score (sum of side lengths)."""
    if not _is_valid(centers, sides):
        return -np.inf
    return np.sum(sides)

def squares_in_square10() -> tuple:
    """
    Maximize the sum of side lengths of 10 non-overlapping axis-aligned squares
    inside the unit square [0,1]^2.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds deadline
    n = 10
    
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    # Initial heuristic: heterogeneous packing
    # Try different grid configurations
    # Configuration 1: 3x3 grid with one extra
    # Configuration 2: 2x5 grid
    # Configuration 3: Custom asymmetric packing
    
    def try_configuration(config_name):
        nonlocal best_centers, best_sides, best_score
        nonlocal deadline, start_time
        
        if time.time() > deadline - 5:
            return
        
        if config_name == "3x3+1":
            # 3x3 grid of side 1/3, plus one smaller square
            # Actually, let's try a packing where we have 9 squares of side ~1/3
            # and 1 square of side ~1/9 or so
            # But 3x3 grid of side 1/3 fills the square exactly.
            # We need to make room for a 10th square.
            # Let's try: 3 columns, 3 rows.
            # Maybe make the top row squares smaller to fit a 4th? No, 3x3=9.
            # Let's try a 3x3 grid where one cell is split into two smaller squares.
            # Or simply: 3x3 grid of side 0.32, leaving gaps.
            
            # Better idea: Use a 3x3 grid of side s, and place the 10th square
            # in the corner gap if we shrink the corner squares.
            
            # Let's try a simple asymmetric packing:
            # 4 squares of side 0.25 in a 2x2 grid, plus 6 smaller squares.
            # Or: 5 squares of side 0.2 in a row, and 5 in another row?
            
            # Let's try: 2 rows.
            # Top row: 5 squares of side s1
            # Bottom row: 5 squares of side s2
            # s1 = s2 = 0.2 gives sum = 2.0.
            # If we make top row squares larger, we need to shrink bottom.
            
            # Let's try a known good packing for n=10.
            # According to literature, the optimal packing for 10 squares
            # has sum of sides around 3.0+.
            
            # Try: 3 large squares of side 1/3 = 0.333, and 7 small squares.
            # 3 * 0.333 = 1.0, so they fill one row.
            # Remaining area is 2/3 * 1.
            # Fit 7 squares in 1 x 2/3 rectangle.
            # Max side in 1 x 2/3 is 2/3 if one square, but we need 7.
            # Grid of 3x3 in 1x2/3: side = min(1/3, (2/3)/3) = 1/3.
            # So 3x3 grid of side 1/3 in the remaining 2/3 height? No, 3*1/3=1 > 2/3.
            # So side = (2/3)/3 = 2/9 = 0.222.
            # Sum = 3*(1/3) + 7*(2/9) = 1 + 14/9 = 23/9 = 2.555.
            
            # Let's try to optimize this.
            # Let the 3 large squares have side L, placed in top row.
            # L = 1/3 is max.
            # The 7 small squares are in the bottom 2/3 height.
            # Let them be in a 3x3 grid minus 2 corners? Or 4+3?
            # Let's try 4 in top sub-row, 3 in bottom sub-row of the bottom 2/3.
            # Sub-row 1 (height h1): 4 squares of side s1. 4*s1 <= 1 => s1 <= 0.25.
            # Sub-row 2 (height h2): 3 squares of side s2. 3*s2 <= 1 => s2 <= 1/3.
            # h1 + h2 = 2/3.
            # To maximize 4*s1 + 3*s2 subject to s1 <= 0.25, s2 <= 1/3, s1 <= h1, s2 <= h2.
            # Set s1 = min(0.25, h1), s2 = min(1/3, h2).
            # If h1 >= 0.25 and h2 >= 1/3, then s1=0.25, s2=1/3.
            # h1 + h2 = 0.25 + 1/3 = 7/12 = 0.583 < 2/3.
            # So we can increase h1 or h2? No, s1 and s2 are capped.
            # Sum = 4*0.25 + 3*(1/3) = 1 + 1 = 2.
            # Total sum = 3*(1/3) + 2 = 3.0.
            
            # Let's verify this packing:
            # Top row: 3 squares of side 1/3.
            # Bottom area: 1 x 2/3.
            # In bottom area:
            #   Top sub-row (height 0.25): 4 squares of side 0.25.
            #   Bottom sub-row (height 1/3): 3 squares of side 1/3.
            #   Total height used: 0.25 + 1/3 = 7/12 = 0.5833.
            #   Remaining height: 2/3 - 7/12 = 8/12 - 7/12 = 1/12.
            #   This works.
            
            centers = np.zeros((10, 2))
            sides = np.zeros(10)
            
            # Top row: 3 squares of side 1/3
            s_large = 1.0 / 3.0
            for i in range(3):
                centers[i, 0] = (i + 0.5) * s_large
                centers[i, 1] = 1.0 - s_large / 2
                sides[i] = s_large
            
            # Bottom area: y from 0 to 2/3
            # Sub-row 1: 4 squares of side 0.25, centered at y = 2/3 - 0.125
            s_med = 0.25
            y_med = 2.0 / 3.0 - s_med / 2
            for i in range(4):
                centers[3 + i, 0] = (i + 0.5) * s_med
                centers[3 + i, 1] = y_med
                sides[3 + i] = s_med
            
            # Sub-row 2: 3 squares of side 1/3, centered at y = 1/6
            s_small = 1.0 / 3.0
            y_small = s_small / 2
            for i in range(3):
                centers[7 + i, 0] = (i + 0.5) * s_small
                centers[7 + i, 1] = y_small
                sides[7 + i] = s_small
            
            score = _score(centers, sides)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_sides = sides.copy()
        
        elif config_name == "2x5":
            # 2 rows of 5 squares each
            # Top row: 5 squares of side s1
            # Bottom row: 5 squares of side s2
            # s1 = s2 = 0.2 is baseline.
            # Can we do better?
            # If s1 = s2 = s, then 5s <= 1 => s <= 0.2. Sum = 10s <= 2.0.
            # If we make s1 != s2, we still have 5s1 <= 1 and 5s2 <= 1.
            # So max sum is 2.0.
            
            centers = np.zeros((10, 2))
            sides = np.zeros(10)
            s = 0.2
            for i in range(5):
                centers[i, 0] = (i + 0.5) * s
                centers[i, 1] = 0.75
                sides[i] = s
                centers[5 + i, 0] = (i + 0.5) * s
                centers[5 + i, 1] = 0.25
                sides[5 + i] = s
            
            score = _score(centers, sides)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_sides = sides.copy()
        
        elif config_name == "grid4x4":
            # 4x4 grid of side 0.25
            centers = np.zeros((10, 2))
            sides = np.zeros(10)
            s = 0.25
            idx = np.arange(10)
            cols = idx % 4
            rows = idx // 4
            centers[:, 0] = (cols + 0.5) * s
            centers[:, 1] = (rows + 0.5) * s
            sides[:] = s
            
            score = _score(centers, sides)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_sides = sides.copy()
        
        elif config_name == "asymmetric":
            # Try a more complex asymmetric packing
            # 1 large square of side 0.5 in corner
            # 4 medium squares of side 0.25
            # 5 small squares of side 0.1
            # Sum = 0.5 + 4*0.25 + 5*0.1 = 0.5 + 1.0 + 0.5 = 2.0
            
            centers = np.zeros((10, 2))
            sides = np.zeros(10)
            
            # Large square in bottom-left
            s_large = 0.5
            centers[0, 0] = s_large / 2
            centers[0, 1] = s_large / 2
            sides[0] = s_large
            
            # 4 medium squares in top-right 0.5x0.5 area
            s_med = 0.25
            for i in range(4):
                col = i % 2
                row = i // 2
                centers[1 + i, 0] = 0.5 + (col + 0.5) * s_med
                centers[1 + i, 1] = 0.5 + (row + 0.5) * s_med
                sides[1 + i] = s_med
            
            # 5 small squares in remaining space
            # Remaining space: bottom-right 0.5x0.5 minus large square? No, large is bottom-left.
            # Remaining: top-left 0.5x0.5 and bottom-right 0.5x0.5.
            # Top-left: 0.5x0.5. Place 2 small squares of side 0.25?
            # Bottom-right: 0.5x0.5. Place 3 small squares of side 0.25?
            # But 0.25*2 = 0.5, so 2x1 grid fits.
            # Let's place 2 in top-left, 3 in bottom-right.
            s_small = 0.25
            # Top-left
            centers[5, 0] = 0.125
            centers[5, 1] = 0.75
            sides[5] = s_small
            centers[6, 0] = 0.375
            centers[6, 1] = 0.75
            sides[6] = s_small
            
            # Bottom-right
            centers[7, 0] = 0.625
            centers[7, 1] = 0.125
            sides[7] = s_small
            centers[8, 0] = 0.875
            centers[8, 1] = 0.125
            sides[8] = s_small
            centers[9, 0] = 0.75
            centers[9, 1] = 0.375
            sides[9] = s_small
            
            score = _score(centers, sides)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_sides = sides.copy()
    
    # Try all configurations
    try_configuration("3x3+1")
    try_configuration("2x5")
    try_configuration("grid4x4")
    try_configuration("asymmetric")
    
    # Local optimization using gradient-free methods
    if best_centers is not None and time.time() < deadline - 10:
        from scipy.optimize import minimize
        
        def objective(x):
            centers = x[:20].reshape(10, 2)
            sides = x[20:]
            if not _is_valid(centers, sides):
                return -1000 + np.sum(sides)  # Penalty
            return -np.sum(sides)
        
        x0 = np.concatenate([best_centers.flatten(), best_sides])
        
        # Use Nelder-Mead for local optimization
        result = minimize(objective, x0, method='Nelder-Mead', 
                         options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if result.fun < -best_score:
            new_centers = result.x[:20].reshape(10, 2)
            new_sides = result.x[20:]
            if _is_valid(new_centers, new_sides):
                new_score = np.sum(new_sides)
                if new_score > best_score:
                    best_score = new_score
                    best_centers = new_centers
                    best_sides = new_sides
    
    # Return the best result found
    if best_centers is None:
        # Fallback to simple grid
        centers = np.zeros((10, 2))
        sides = np.zeros(10)
        s = 0.25
        idx = np.arange(10)
        cols = idx % 4
        rows = idx // 4
        centers[:, 0] = (cols + 0.5) * s
        centers[:, 1] = (rows + 0.5) * s
        sides[:] = s
        return centers, sides
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
