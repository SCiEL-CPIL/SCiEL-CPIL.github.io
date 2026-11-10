import itertools
import time
import math

# EVOLVE-BLOCK-START

def _line_key(p1, p2):
    """
    Returns a normalized key for the line passing through p1 and p2.
    p1, p2 are (x, y) tuples.
    Line equation: Ax + By + C = 0
    A = y1 - y2
    B = x2 - x1
    C = x1*y2 - x2*y1
    Normalize by GCD and sign.
    """
    x1, y1 = p1
    x2, y2 = p2
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    
    if A == 0 and B == 0:
        return None # Should not happen for distinct points
        
    # Find GCD of A, B, C
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return abs(a)
        
    g = gcd(gcd(A, B), C)
    if g != 0:
        A //= g
        B //= g
        C //= g
        
    # Normalize sign: first non-zero of A, B, C should be positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A = -A
        B = -B
        C = -C
        
    return (A, B, C)

def _count_ordinary_lines(points):
    """
    Counts the number of lines containing exactly two of the given points.
    """
    n = len(points)
    line_pairs = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            line_pairs[key] = line_pairs.get(key, 0) + 1
            
    ordinary_count = 0
    for count in line_pairs.values():
        if count == 1:
            ordinary_count += 1
    return ordinary_count

def _get_candidate_configs():
    """
    Generates a list of candidate configurations of 12 points.
    These are known or suspected to have low ordinary line counts.
    """
    candidates = []
    
    # 1. 3x4 Grid (Standard fallback, usually 12 or more ordinary lines)
    # Points: (0,0) to (3,3)
    pts_3x4 = [(x, y) for x in range(4) for y in range(3)]
    candidates.append(pts_3x4)
    
    # 2. 4x3 Grid (Transposed)
    pts_4x3 = [(x, y) for x in range(3) for y in range(4)]
    candidates.append(pts_4x3)
    
    # 3. Hesse Configuration Subset
    # The Hesse configuration has 9 points on 12 lines. 
    # We need 12 points. 
    # A known dense configuration for 12 points is related to the affine plane of order 3 
    # or specific subsets of the Hesse configuration.
    # Let's try a configuration based on the 3x3 grid plus 3 points at infinity (scaled).
    # However, we must use finite integer coordinates.
    
    # Let's use a known good configuration from literature for n=12.
    # One such configuration is the "12-point configuration" found in 
    # "Ordinary lines and the Sylvester-Gallai theorem" related searches.
    
    # Configuration A: 3x4 grid is often cited as having 12 ordinary lines.
    # Can we do better? 
    # The theoretical minimum for 12 points is 6.
    # Configurations achieving 6 or 7 are rare and specific.
    
    # Let's generate a few specific patterns:
    
    # Pattern: 2x6 grid
    pts_2x6 = [(x, y) for x in range(6) for y in range(2)]
    candidates.append(pts_2x6)
    
    # Pattern: 4x3 grid with one point moved?
    # Let's stick to simple grids first, then add specific "Hesse-like" points.
    
    # Hesse-like configuration on integer grid:
    # The 9 points of the Hesse configuration can be embedded in a 3x3 grid.
    # To get 12 points, we can add 3 more points.
    # A common extension is to take the 3x3 grid and add points on the diagonals or midpoints.
    
    # Let's try the 3x3 grid plus 3 points: (0,3), (1,3), (2,3) -> 3x4 grid (already have)
    # Let's try 3x3 grid plus (3,0), (3,1), (3,2) -> 4x3 grid (already have)
    
    # What about a "cross" shape or other symmetries?
    
    # Let's include a configuration derived from the 12 points of the 
    # "Klein's icosahedral configuration" or similar, but scaled to integers.
    # Actually, for small n, grid subsets are often optimal or near-optimal.
    
    # Let's add a specific configuration:
    # 4 points on x-axis, 4 on y-axis, 4 on diagonal?
    
    # Configuration: 
    # (0,0), (1,0), (2,0), (3,0)
    # (0,1), (1,1), (2,1), (3,1)
    # (0,2), (1,2), (2,2), (3,2)
    # This is the 3x4 grid.
    
    # Let's try a non-rectangular grid.
    # 12 points: 
    # Row 0: 0, 1, 2, 3
    # Row 1: 0, 1, 2
    # Row 2: 0, 1
    # Row 3: 0
    # Total 10. Need 2 more.
    
    # Let's just evaluate the standard grids and a few specific manual constructions.
    
    # Construction 1: 3x4 grid
    # Construction 2: 2x6 grid
    # Construction 3: 4x3 grid
    
    # Construction 4: "L" shape with extra points
    # (0,0) to (3,0)
    # (0,1) to (3,1)
    # (0,2) to (3,2)
    # (0,3)
    # (1,3)
    # (2,3)
    # This is 3x4 + 3 points in a row? No, that's 12 points.
    # Let's check if this is distinct from 3x4.
    # 3x4 is 4x3 or 3x4. 
    # My pts_3x4 is x in 0..3, y in 0..2. That's 4 columns, 3 rows.
    # My pts_4x3 is x in 0..2, y in 0..3. That's 3 columns, 4 rows.
    
    # Let's add a configuration with points at (0,0), (1,0), (2,0), (3,0),
    # (0,1), (1,1), (2,1), (3,1),
    # (0,2), (1,2), (2,2), (3,2)
    # This is exactly pts_3x4.
    
    # Let's try a configuration based on the 12 vertices of a regular dodecagon?
    # Coordinates would be irrational. Need integers.
    
    # Let's try a configuration from a known table of minimal ordinary lines.
    # For n=12, min ordinary lines is 6.
    # Example achieving 6:
    # Points: 
    # (0,0), (1,0), (2,0), (3,0)
    # (0,1), (1,1), (2,1), (3,1)
    # (0,2), (1,2), (2,2), (3,2)
    # Wait, I should verify if 3x4 grid has 6 ordinary lines.
    # Let's calculate for 3x4 (4 cols, 3 rows):
    # Horizontal lines: 3 lines, each with 4 points. Pairs: 6 per line. Not ordinary.
    # Vertical lines: 4 lines, each with 3 points. Pairs: 3 per line. Not ordinary.
    # Diagonals:
    # Slope 1:
    # (0,0)-(1,1)-(2,2)-(3,3) -> Only 3 points in grid (0,0) to (2,2). (3,3) not in grid.
    # So (0,0)-(1,1)-(2,2) is a line with 3 points.
    # (1,0)-(2,1)-(3,2) is a line with 3 points.
    # (0,1)-(1,2) is a line with 2 points. ORDINARY.
    # (1,1)-(2,2) is part of the 3-point line.
    # (2,1)-(3,2) is part of the 3-point line.
    # (0,2) is isolated in slope 1 direction?
    # Let's rely on the function to count.
    
    # There is a known configuration for n=12 with 6 ordinary lines.
    # It is often a subset of the 4x4 grid or similar.
    
    # Let's add a specific candidate:
    # The "12-point configuration" from the paper "On the number of ordinary lines"
    # One such configuration is:
    # (0,0), (1,0), (2,0), (3,0)
    # (0,1), (1,1), (2,1), (3,1)
    # (0,2), (1,2), (2,2), (3,2)
    # This is the 3x4 grid.
    
    # Another candidate:
    # (0,0), (1,0), (2,0), (3,0)
    # (0,1), (1,1), (2,1), (3,1)
    # (0,2), (1,2), (2,2), (3,2)
    # (0,3), (1,3), (2,3), (3,3) -> 4x4 grid has 16 points.
    
    # Let's just include the 3x4, 4x3, and 2x6 grids.
    # And a "diamond" shape.
    
    # Diamond:
    # (0,1), (1,0), (1,1), (1,2), (2,1)
    # (0,2), (2,0)
    # (3,1), (1,3)
    # (0,3), (3,0)
    # (2,2), (2,3)?
    # This is getting complicated.
    
    # Let's stick to the grids and add a few random-looking but structured ones.
    
    # Configuration 5: 3x3 grid + 3 points on the perimeter
    # 3x3: (0,0) to (2,2)
    # Add: (3,0), (3,1), (3,2) -> This is 4x3 grid.
    
    # Configuration 6: 3x3 grid + 3 points on another side
    # Add: (0,3), (1,3), (2,3) -> This is 3x4 grid.
    
    # It seems 3x4 and 4x3 are the main grid candidates.
    
    # Let's add a configuration with a "hole".
    # 4x4 grid minus 4 corners?
    # (1,0), (2,0)
    # (0,1), (1,1), (2,1), (3,1)
    # (0,2), (1,2), (2,2), (3,2)
    # (1,3), (2,3)
    # Total 10 points. Need 2 more.
    # Add (0,0) and (3,3)?
    # Let's try:
    pts_hole = [
        (1,0), (2,0),
        (0,1), (1,1), (2,1), (3,1),
        (0,2), (1,2), (2,2), (3,2),
        (1,3), (2,3),
        (0,0), (3,3)
    ]
    candidates.append(pts_hole)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 10  # 10s margin
    
    candidates = _get_candidate_configs()
    
    best_count = float('inf')
    best_points = None
    
    for pts in candidates:
        if time.time() > deadline:
            break
            
        # Ensure distinctness
        if len(set(pts)) != 12:
            continue
            
        # Ensure not all collinear (grids are not)
        
        count = _count_ordinary_lines(pts)
        if count < best_count:
            best_count = count
            best_points = pts
            # If we hit the theoretical minimum, we can stop
            if best_count <= 6:
                break
                
    # If no candidate found (should not happen), fallback to 3x4
    if best_points is None:
        best_points = [(x, y) for x in range(4) for y in range(3)]
        
    # Convert to list of lists of integers
    result = [[int(x), int(y)] for x, y in best_points]
    
    return result

# EVOLVE-BLOCK-END
