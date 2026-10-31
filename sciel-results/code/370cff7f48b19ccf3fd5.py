"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
# EVOLVE-BLOCK-START

def _pack_grid(n, w, h):
    """
    Pack n circles in a rectangle of width w and height h.
    Returns list of (x, y, r).
    Uses a simple grid-like approach with optimization.
    """
    # Estimate number of rows
    # We want roughly square cells
    # Let's try to fit n circles in a grid
    # Find integer k, m such that k*m >= n and aspect ratio is reasonable
    best_pack = None
    best_score = -1
    
    # Try different grid configurations
    for k in range(1, n + 1):
        m = int(np.ceil(n / k))
        # Check if this grid fits in w x h
        # Cell size
        cw = w / k
        ch = h / m
        # Max radius for a circle in a cell (if isolated) is min(cw, ch)/2
        # But we need to check if they fit.
        # Actually, let's just place them in a grid and see.
        
        # If k*m > n, we have empty slots.
        # Let's just generate a grid of k*m circles and take the first n?
        # No, that might leave large empty spaces.
        # Better: distribute the n circles as evenly as possible in k columns.
        
        # Let's try a specific strategy:
        # Place circles in rows.
        # Row i has either floor(n/k) or ceil(n/k) circles.
        
        r_per_row = [0] * k
        for i in range(n):
            r_per_row[i % k] += 1
        
        # Now we have counts per row.
        # For each row j with count c_j, the width available is w.
        # The circles in that row should be packed.
        # Max radius for a row of c circles in width w is w/(2*c) if they touch walls and each other.
        # But they also need to fit in height h.
        # The vertical position of row j depends on the radii.
        
        # This is getting complex. Let's use a simpler heuristic:
        # Assume all circles have the same radius r.
        # We want to maximize r.
        # Constraints:
        # 1. Circles don't overlap.
        # 2. Circles are inside the rectangle.
        
        # Let's try to solve for r and positions using optimization.
        # Variables: x_1..x_n, y_1..y_n, r
        # Objective: maximize r
        # Constraints:
        #   x_i >= r, x_i <= w - r
        #   y_i >= r, y_i <= h - r
        #   (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2 = (2r)^2 for all i != j
        
        # Since n=21, this is a small problem. We can use scipy.optimize.
        # However, the non-convex constraints make it tricky.
        # We can use a penalty method or SLSQP with initial guess.
        
        # Let's try a specific initial guess: grid layout.
        # Grid of k columns and m rows.
        # x_i = (col + 0.5) * w/k
        # y_i = (row + 0.5) * h/m
        # r = min(w/k, h/m) / 2
        
        # Let's just return a simple grid packing for now to ensure validity.
        # We can refine this with optimization later if time permits.
        
        # Calculate r for this grid
        r = min(w / k, h / m) / 2
        
        # Check if circles fit
        # They fit if r > 0.
        
        # Construct positions
        positions = []
        for i in range(n):
            row = i // k
            col = i % k
            x = (col + 0.5) * w / k
            y = (row + 0.5) * h / m
            positions.append([x, y, r])
        
        # This is a valid packing.
        # Score is sum of radii = n * r
        score = n * r
        if score > best_score:
            best_score = score
            best_pack = positions
            
    return best_pack

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s margin
    
    # Perimeter P = 2(w + h) = 4 => w + h = 2
    # To maximize area (and thus likely radius sum), we want w = h = 1 (square).
    # Let's assume square w=1, h=1.
    
    w = 1.0
    h = 1.0
    
    # Initial guess: grid
    pack = _pack_grid(N, w, h)
    if pack is None:
        # Fallback
        r = 0.1
        pack = [[(i % 5) * 0.2 + 0.1, (i // 5) * 0.2 + 0.1, r] for i in range(N)]
    
    # Convert to numpy array
    result = np.array(pack)
    
    # Try to improve with optimization
    # We can try to adjust positions and radii.
    # Let's use a simple local search or gradient ascent if time allows.
    
    # For now, return the grid packing.
    # Check validity
    # 1. Perimeter <= 4
    # 2. Disjoint interiors
    
    # The grid packing is valid.
    
    return result

# EVOLVE-BLOCK-END
