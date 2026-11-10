# EVOLVE-BLOCK-START
import numpy as np
from math import gcd


def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization. Construct 54 integer points and 27 integer
    lines to maximize the number of point-line incidences.

    Construction:
    1. Start with a 4x13 integer grid (52 points).
    2. Select 22 lines that each pass through 4 grid points (total 88 incidences).
    3. Select 5 lines that each pass through 3 grid points (total 15 incidences).
       Total for grid: 103 incidences.
    4. Add 2 extra points (total 54) placed to lie on 2 of the 4-point lines.
       This adds 2 more incidences (one per new point, on its respective line).
       Total incidences: 105.

    The 22 four-point lines are chosen from slopes 1-4.
    The 5 three-point lines are chosen from slopes 1-4 (ensuring no duplicates).
    """
    # 1. Generate 4x13 grid
    xs = np.arange(1, 5)
    ys = np.arange(1, 14)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Grid points are unique by construction.
    
    # 2. Identify lines
    # We define lines by (A, B, C) such that A*x + B*y + C = 0.
    # We want to maximize incidence.
    # Let's manually construct the specific lines based on the analysis.
    
    lines = []
    used_line_keys = set()
    
    def add_line(A, B, C):
        # Canonicalize for uniqueness check
        # Divide by gcd
        g = gcd(A, B)
        g = gcd(g, C)
        if g != 0:
            A, B, C = A // g, B // g, C // g
        # Ensure first non-zero is positive
        if A < 0 or (A == 0 and B < 0):
            A, B, C = -A, -B, -C
        key = (A, B, C)
        if key not in used_line_keys:
            used_line_keys.add(key)
            lines.append((A, B, C))
            return True
        return False

    # --- 4-Point Lines (22 total) ---
    # Slope 1: y = x + b. Points (x, x+b). x in 1..4, y in 1..13.
    # 1+b >= 1 => b >= 0. 4+b <= 13 => b <= 9.
    # b in {0, ..., 9} -> 10 lines.
    for b in range(0, 10):
        # Line: x - y + b = 0 => (1, -1, b)
        add_line(1, -1, b)
        
    # Slope 2: y = 2x + b. Points (x, 2x+b).
    # 1: 2+b >= 1 => b >= -1.
    # 4: 8+b <= 13 => b <= 5.
    # b in {-1, ..., 5} -> 7 lines.
    for b in range(-1, 6):
        # Line: 2x - y + b = 0 => (2, -1, b)
        add_line(2, -1, b)
        
    # Slope 3: y = 3x + b.
    # 1: 3+b >= 1 => b >= -2.
    # 4: 12+b <= 13 => b <= 1.
    # b in {-2, -1, 0, 1} -> 4 lines.
    for b in range(-2, 2):
        # Line: 3x - y + b = 0 => (3, -1, b)
        add_line(3, -1, b)
        
    # Slope 4: y = 4x + b.
    # 1: 4+b >= 1 => b >= -3.
    # 4: 16+b <= 13 => b <= -3.
    # b = -3.
    # b in {-3} -> 1 line.
    for b in [-3]:
        # Line: 4x - y + b = 0 => (4, -1, b)
        add_line(4, -1, b)
        
    # Check count of 4-point lines: 10 + 7 + 4 + 1 = 22. Correct.

    # --- 3-Point Lines (5 total) ---
    # We need 5 more lines. We must ensure they are distinct from the 22 above.
    # Slope 1, 3-point lines:
    # b = -1: y = x - 1. Points: (2,1), (3,2), (4,3). (1,0) not in grid.
    # b = 10: y = x + 10. Points: (1,11), (2,12), (3,13). (4,14) not in grid.
    # These are distinct from b in 0..9.
    add_line(1, -1, -1) # (1, -1, -1)
    add_line(1, -1, 10) # (1, -1, 10)
    
    # Slope 2, 3-point lines:
    # b = -2: y = 2x - 2. Points: (2,2), (3,4), (4,6). (1,0) not in grid.
    # b = 6: y = 2x + 6. Points: (1,8), (2,10), (3,12). (4,14) not in grid.
    # These are distinct from b in -1..5.
    add_line(2, -1, -2) # (2, -1, -2)
    add_line(2, -1, 6)  # (2, -1, 6)
    
    # Slope 3, 3-point lines:
    # b = -3: y = 3x - 3. Points: (2,3), (3,6), (4,9). (1,0) not in grid.
    # Distinct from b in -2..1.
    add_line(3, -1, -3) # (3, -1, -3)
    
    # Total lines: 22 + 5 = 27.

    # 3. Add 2 extra points
    # We want to add points that increase the incidence count.
    # Ideally, place a point on a line that already has many points? No, incidence is sum of points on lines.
    # Adding a point to a line increases that line's count by 1.
    # So we want to place the new points on lines that are already in our set.
    # We have 22 lines with 4 points. If we place a new point on one of these, it becomes 5.
    # Incidence increases by 1 per point added to an existing line.
    # If we place them on distinct 4-point lines, we get +2 incidences.
    # Total: 103 + 2 = 105.
    
    # Let's pick two lines from the 4-point set.
    # Line 1: Slope 1, b=0 -> y = x. Points: (1,1), (2,2), (3,3), (4,4).
    # Let's place a point on this line. E.g., (5, 5).
    # (5,5) is not in the grid (x in 1..4). So it's a new point.
    # Does (5,5) lie on any other selected lines?
    # Check against all 27 lines.
    # It shouldn't, as they are defined by specific intercepts.
    # Let's verify quickly.
    # Slope 1 lines: y=x+b. (5,5) -> 5=5+b -> b=0. Only b=0 line.
    # Slope 2 lines: y=2x+b. 5=10+b -> b=-5. Not in {-1..5, -2, 6}.
    # Slope 3 lines: y=3x+b. 5=15+b -> b=-10. Not in {-2..1, -3}.
    # Slope 4 lines: y=4x+b. 5=20+b -> b=-15. Not in {-3}.
    
    # Line 2: Slope 2, b=0 -> y = 2x. Points: (1,2), (2,4), (3,6), (4,8).
    # Let's place a point on this line. E.g., (5, 10).
    # (5,10) is not in grid.
    # Check other lines.
    # Slope 1: 10=5+b -> b=5. Is b=5 in our set?
    # Our Slope 1 4-pt lines: b=0..9. Yes, b=5 is in the set!
    # Wait. If (5,10) is on line y=x+5 (which is in our set), it adds an incidence to that line too.
    # So adding (5,10) adds 1 incidence to y=2x and 1 incidence to y=x+5.
    # Total +2 incidences from this single point?
    # Let's check if y=x+5 is one of the 22 four-point lines.
    # Slope 1, b=5. Yes, it is.
    # So point (5,10) lies on two lines in our set: y=2x and y=x+5.
    # This adds 2 incidences.
    
    # Let's try to find a point that lies on multiple lines.
    # Actually, if we add (5,5) it is only on y=x. (+1)
    # If we add (5,10) it is on y=2x and y=x+5. (+2)
    # We need 2 points.
    # Point 1: (5,10). Adds 2 incidences.
    # Point 2: Let's find another one.
    # Maybe (0, ?) or (6, ?)
    # Let's try (0,0).
    # Slope 1: 0=0+b -> b=0. Line y=x is in set. (+1)
    # Slope 2: 0=0+b -> b=0. Line y=2x is in set. (+1)
    # Slope 3: 0=0+b -> b=0. Line y=3x is in set (b=0 in -2..1). (+1)
    # Slope 4: 0=0+b -> b=0. Line y=4x? b=-3 is the only one. 0 != -3. No.
    # So (0,0) lies on y=x, y=2x, y=3x.
    # All three are in our set?
    # y=x: b=0. Yes.
    # y=2x: b=0. Yes.
    # y=3x: b=0. Yes.
    # So (0,0) adds 3 incidences.
    
    # Total added incidences: 2 (from 5,10) + 3 (from 0,0) = 5.
    # Base 103 + 5 = 108.
    
    # Let's double check (0,0) and (5,10) are not in the original grid.
    # Grid x in 1..4. So 0 and 5 are outside.
    # Are they distinct? Yes.
    # Are they distinct from grid points? Yes.
    
    # Let's verify no other lines contain (0,0) or (5,10).
    # (0,0):
    # Slope 1: b=0. (Line 1)
    # Slope 2: b=0. (Line 11)
    # Slope 3: b=0. (Line 18)
    # Slope 4: b=0. Not in set (only b=-3).
    # Any 3-point lines?
    # Slope 1: b=-1, 10. (0,0) -> b=0. No.
    # Slope 2: b=-2, 6. (0,0) -> b=0. No.
    # Slope 3: b=-3. (0,0) -> b=0. No.
    
    # (5,10):
    # Slope 1: b=5. (Line 5). Yes.
    # Slope 2: b=0. (Line 11). Yes.
    # Slope 3: b=-5. Not in set (-2..1, -3).
    # Slope 4: b=-10. Not in set.
    # 3-point lines:
    # Slope 1: b=-1, 10. No.
    # Slope 2: b=-2, 6. No.
    # Slope 3: b=-3. No.
    
    # So incidences added:
    # (0,0) adds 3.
    # (5,10) adds 2.
    # Total 5.
    
    # New points: (0,0) and (5,10).
    # Total points: 52 + 2 = 54.
    
    # Append to points
    points = np.vstack([points, np.array([[0, 0], [5, 10]], dtype=np.int64)])
    
    # Final sanity check:
    # 54 points, 27 lines.
    # No repeated points.
    # No repeated lines.
    # All integers.
    
    lines_arr = np.array(lines, dtype=np.int64)
    
    return points, lines_arr

# EVOLVE-BLOCK-END
