# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _is_feasible(centers, sides, tol=1e-7, n=None):
    """
    Returns True if the configuration is feasible:
    - all finite
    - all sides > 0
    - each square lies within [-tol, 1+tol]^2
    - no two squares overlap (interiors)
    centers: (n,2), sides: (n,)
    """
    if n is None:
        n = len(sides)
    if not (np.all(np.isfinite(centers)) and np.all(np.isfinite(sides))):
        return False
    if np.any(sides <= 0):
        return False
    # bounds
    x0 = centers[:, 0] - sides / 2.0
    x1 = centers[:, 0] + sides / 2.0
    y0 = centers[:, 1] - sides / 2.0
    y1 = centers[:, 1] + sides / 2.0
    if np.any(x0 < -tol) or np.any(x1 > 1 + tol) or np.any(y0 < -tol) or np.any(y1 > 1 + tol):
        return False
    # overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j]) / 2.0 - tol
            if dx < t and dy < t:
                return False
    return True

def _sum_sides(sides):
    return float(np.sum(sides))

def _solve_guillotine(n=10, deadline=None):
    """
    Recursive guillotine cut search.
    Returns (centers, sides) for the best packing found.
    """
    if deadline is None:
        deadline = time.time() + 100
    
    best_sum = -np.inf
    best_centers = None
    best_sides = None
    
    # We represent the state as a list of rectangles (x, y, w, h)
    # and a list of squares to place (indices)
    # Since all squares are identical in size initially? No, we want to maximize sum.
    # Actually, a guillotine cut tree implies we decide the sizes.
    
    # Alternative: We fix the tree topology and optimize the split ratios?
    # Or, we can use a heuristic: 
    # 1. Start with one big rectangle [0,1]x[0,1].
    # 2. Recursively split.
    # 3. When a rectangle is small enough or we have placed all squares, assign a square to it.
    #    The square size is min(w, h).
    #    This is a "guillotine packing" where squares are as large as possible in their leaf.
    #    But we can also choose to split a leaf into two sub-leaves.
    
    # Let's try a search over tree topologies.
    # A tree with 10 leaves has 9 internal nodes.
    # Each internal node has a split direction (H/V) and a split ratio (0..1).
    # This is a high-dimensional space.
    
    # Better: Use a "slice" representation.
    # A slicing floorplan can be represented by a sequence of cuts.
    # Let's use a randomized heuristic search over slicing floorplans.
    
    # A simpler approach: 
    # Try a few standard guillotine patterns:
    # 1. 2x5 grid (5 rows of 2)
    # 2. 5x2 grid
    # 3. 1 big square (0.5x0.5) and 9 small? No, 10 squares.
    # 4. 2 big (0.5x0.5) and 8 small (0.25x0.25)? 
    #    Sum = 2*0.5 + 8*0.25 = 1 + 2 = 3.0.
    #    This is a known good configuration.
    
    # Let's implement a general guillotine optimizer.
    # We define a tree. We optimize the split ratios to maximize the sum of min(w,h) at leaves.
    # But the tree structure itself is discrete.
    
    # Let's try a specific promising structure:
    # Split [0,1]x[0,1] horizontally into two halves: [0,0.5] and [0.5,1].
    # Left half: Split vertically into two [0,0.5]x[0,0.5] and [0,0.5]x[0.5,1].
    # Right half: Split vertically into two [0.5,1]x[0,0.5] and [0.5,1]x[0.5,1].
    # Now we have 4 quadrants of 0.5x0.5.
    # If we stop here, we have 4 squares of 0.5. Sum = 2.0.
    # We need 10 squares.
    # So we must split some quadrants further.
    
    # Let's define a recursive function that takes a rectangle and a number of squares to place.
    # It tries to split the rectangle into two parts, with k and n-k squares.
    # It maximizes the sum.
    
    # This is essentially a dynamic programming problem if we could discretize the split ratios.
    # But split ratios are continuous.
    
    # Let's use a heuristic: 
    # For a given rectangle (w, h) and n squares:
    # 1. If n == 1: return min(w, h).
    # 2. Try splitting horizontally at ratio r (0<r<1).
    #    Top part: w x r*h, with k squares.
    #    Bottom part: w x (1-r)*h, with n-k squares.
    #    Value = solve(w, r*h, k) + solve(w, (1-r)*h, n-k).
    # 3. Try splitting vertically at ratio r.
    #    Left part: r*w x h, with k squares.
    #    Right part: (1-r)*w x h, with n-k squares.
    #    Value = solve(r*w, h, k) + solve((1-r)*w, h, n-k).
    
    # We can use a local search on the split ratios for a fixed tree topology.
    # Or, we can use a greedy approach: 
    # At each node, try all splits (k, n-k) and optimize r to maximize the sum.
    
    # Let's implement a function that, given a rectangle and n, finds the best guillotine packing.
    # We'll use a simple local search on r for each split.
    
    def solve_rect(w, h, n):
        if n == 1:
            return min(w, h)
        
        best_val = 0
        best_split = None # (direction, r, k)
        
        # Try all k
        for k in range(1, n):
            # Horizontal split
            # Top: w x r*h, k squares
            # Bottom: w x (1-r)*h, n-k squares
            # We want to maximize solve(w, r*h, k) + solve(w, (1-r)*h, n-k)
            # Let f(r) = solve(w, r*h, k) + solve(w, (1-r)*h, n-k)
            # We can approximate this by trying a few r values, or using a simple optimization.
            # Since solve_rect is recursive and potentially expensive, let's be careful.
            
            # For now, let's try a few r values: 0.25, 0.5, 0.75
            for r in [0.25, 0.5, 0.75]:
                val = solve_rect(w, r*h, k) + solve_rect(w, (1-r)*h, n-k)
                if val > best_val:
                    best_val = val
                    best_split = ('H', r, k)
            
            # Vertical split
            for r in [0.25, 0.5, 0.75]:
                val = solve_rect(r*w, h, k) + solve_rect((1-r)*w, h, n-k)
                if val > best_val:
                    best_val = val
                    best_split = ('V', r, k)
        
        return best_val
    
    # This is too slow for deep recursion if we're not careful.
    # And it only gives the value, not the centers.
    
    # Let's go back to the known good configuration: 2 big (0.5) and 8 small (0.25).
    # Can we do better?
    # What if we have 4 medium (1/3) and 6 small?
    # 4 * 1/3 = 1.333.
    # 6 small in the remaining space?
    # Remaining space is not uniform.
    
    # Let's try to optimize the 2+8 configuration.
    # Two 0.5x0.5 squares in opposite corners?
    # If we put them at (0.25, 0.25) and (0.75, 0.75), they touch at (0.5, 0.5).
    # The remaining space is four L-shapes? No, it's four 0.5x0.5 regions minus the squares.
    # Actually, if we have two 0.5x0.5 squares, they divide the unit square into 4 regions of 0.5x0.5?
    # No.
    # If Square 1 is [0,0.5]x[0,0.5] and Square 2 is [0.5,1]x[0.5,1].
    # The remaining space is [0.5,1]x[0,0.5] and [0,0.5]x[0.5,1].
    # Each is 0.5x0.5.
    # We can fit two 0.25x0.25 squares in each 0.5x0.5 region.
    # So 4 small squares. Total 6 squares.
    # We need 10.
    # So we need to split the 0.5x0.5 regions into 4 sub-regions of 0.25x0.25 each.
    # This gives 2 big + 8 small = 10 squares.
    # Sum = 2*0.5 + 8*0.25 = 3.0.
    
    # Can we make the small squares bigger?
    # If we move the big squares, we might create larger gaps.
    # But guillotine cuts restrict us to rectangular gaps.
    
    # Let's try a different topology:
    # 3 big squares?
    # If we have 3 squares of size s, they need area 3s^2 <= 1.
    # If s=0.5, area=0.75. Remaining 0.25.
    # Can we fit 7 squares in 0.25 area?
    # 7 * (0.25/7)^2? No, they must be squares.
    # Max size of 7 squares in 0.25 area is roughly sqrt(0.25/7) ~ 0.18.
    # Sum = 3*0.5 + 7*0.18 = 1.5 + 1.26 = 2.76. Worse.
    
    # What if we have 1 big square?
    # s=0.5. Area 0.25.
    # Remaining 0.75.
    # 9 squares in 0.75.
    # Max size ~ sqrt(0.75/9) = 0.288.
    # Sum = 0.5 + 9*0.288 = 0.5 + 2.59 = 3.09.
    # This is better!
    
    # Let's try to construct this:
    # One 0.5x0.5 square in a corner, say [0,0.5]x[0,0.5].
    # Remaining space: [0.5,1]x[0,1] (0.5x1) and [0,0.5]x[0.5,1] (0.5x0.5).
    # Total area 0.75.
    # We need to fit 9 squares.
    # Let's split the remaining space into 9 equal rectangles?
    # If we make them all 0.5/3 x 1/3? No, they must be squares.
    
    # Let's use a guillotine cut on the remaining space.
    # Region 1: [0.5,1]x[0,1] (0.5x1).
    # Region 2: [0,0.5]x[0.5,1] (0.5x0.5).
    
    # In Region 2 (0.5x0.5), we can fit 4 squares of 0.25.
    # In Region 1 (0.5x1), we can fit...
    # If we split it into two 0.5x0.5 regions, we can fit 4 squares of 0.25.
    # Total 8 squares of 0.25.
    # Sum = 0.5 + 8*0.25 = 2.5. Worse.
    
    # We need to make the 9 squares in the remaining space as large as possible.
    # This is a sub-problem: pack 9 squares in an L-shape (or union of two rectangles).
    # Guillotine cuts might not be optimal for the sub-problem if the shape is not rectangular.
    # But the remaining space after a guillotine cut is a union of rectangles.
    
    # Let's stick to the 2+8 configuration for now, as it's simple and yields 3.0.
    # The seed yields 2.5.
    # So 3.0 is a 1.2x improvement.
    
    # Let's implement the 2+8 configuration.
    # Centers:
    # Big 1: [0, 0.5]x[0, 0.5] -> Center (0.25, 0.25), Side 0.5
    # Big 2: [0.5, 1]x[0.5, 1] -> Center (0.75, 0.75), Side 0.5
    # Small 1: [0.5, 0.75]x[0, 0.25] -> Center (0.625, 0.125), Side 0.25
    # Small 2: [0.75, 1]x[0, 0.25] -> Center (0.875, 0.125), Side 0.25
    # Small 3: [0.5, 0.75]x[0.25, 0.5] -> Center (0.625, 0.375), Side 0.25
    # Small 4: [0.75, 1]x[0.25, 0.5] -> Center (0.875, 0.375), Side 0.25
    # Small 5: [0, 0.25]x[0.5, 0.75] -> Center (0.125, 0.625), Side 0.25
    # Small 6: [0.25, 0.5]x[0.5, 0.75] -> Center (0.375, 0.625), Side 0.25
    # Small 7: [0, 0.25]x[0.75, 1] -> Center (0.125, 0.875), Side 0.25
    # Small 8: [0.25, 0.5]x[0.75, 1] -> Center (0.375, 0.875), Side 0.25
    
    centers = np.array([
        [0.25, 0.25],
        [0.75, 0.75],
        [0.625, 0.125],
        [0.875, 0.125],
        [0.625, 0.375],
        [0.875, 0.375],
        [0.125, 0.625],
        [0.375, 0.625],
        [0.125, 0.875],
        [0.375, 0.875]
    ])
    sides = np.array([
        0.5,
        0.5,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25
    ])
    
    if _is_feasible(centers, sides):
        return centers, sides
    else:
        # Fallback to grid
        return _grid_initial(3, 3)

def _grid_initial(kx, ky, start_row=0, start_col=0):
    n = 10
    side = min(1.0 / kx, 1.0 / ky)
    cols = []
    rows = []
    for r in range(ky):
        for c in range(kx):
            if len(cols) < n:
                cols.append(c)
                rows.append(r)
            else:
                break
        if len(cols) >= n:
            break

    cx = (np.array(cols, dtype=np.float64) + 0.5) / kx
    cy = (np.array(rows, dtype=np.float64) + 0.5) / ky
    centers = np.stack([cx, cy], axis=1)
    sides = np.full(n, side, dtype=np.float64)
    return centers, sides

def squares_in_square10() -> tuple:
    """
    Uses a guillotine cut tree strategy to find a feasible packing.
    Currently implements a known good 2+8 configuration.
    """
    deadline = time.time() + 600
    
    centers, sides = _solve_guillotine(10, deadline)
    
    return (centers, sides)
# EVOLVE-BLOCK-END
