"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def count_ordinary_lines(points):
    """Count the number of lines containing exactly 2 points."""
    n = len(points)
    if n < 2:
        return 0
    
    line_counts = {}
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            
            # Line through two points: a*x + b*y + c = 0
            a = y2 - y1
            b = x1 - x2
            c = x2 * y1 - x1 * y2
            
            # Normalize the line representation
            g = math.gcd(math.gcd(abs(a), abs(b)), abs(c)) if (a != 0 or b != 0 or c != 0) else 1
            if g == 0:
                g = 1
            a, b, c = a // g, b // g, c // g
            
            # Ensure consistent sign: make the first non-zero of (a,b,c) positive
            if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
                a, b, c = -a, -b, -c
            
            key = (a, b, c)
            line_counts[key] = line_counts.get(key, 0) + 1
    
    # Count lines with exactly 2 points (count == 1 pair on that line means 2 points)
    ordinary = sum(1 for count in line_counts.values() if count == 1)
    return ordinary

def construct():
    """Return 12 integer points minimizing ordinary lines."""
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # The 3x4 grid is a strong candidate for achieving the theoretical minimum of 6
    grid_3x4 = [[x, y] for x in range(4) for y in range(3)]
    best_points = grid_3x4
    best_ordinary = count_ordinary_lines(grid_3x4)
    
    # Try a few alternative configurations to see if we can do better
    candidates = []
    
    # 2x6 grid
    if time.time() < deadline - 10:
        grid_2x6 = [[x, y] for x in range(6) for y in range(2)]
        candidates.append(grid_2x6)
    
    # 4x3 grid (same structure as 3x4, just transposed)
    if time.time() < deadline - 10:
        grid_4x3 = [[x, y] for x in range(3) for y in range(4)]
        candidates.append(grid_4x3)
    
    # Try a configuration with points arranged to maximize 3+ point lines
    # e.g., 3 lines of 4 points each, sharing intersection points
    if time.time() < deadline - 10:
        # Three lines through origin: y=0, y=x, y=2x
        # 4 points on each, but they share (0,0)
        # Line y=0: (0,0),(1,0),(2,0),(3,0)
        # Line y=x: (0,0),(1,1),(2,2),(3,3)
        # Line y=2x: (0,0),(1,2),(2,4),(3,6)
        # That's 1 + 3 + 3 + 3 = 10 points, need 2 more
        three_lines = [[0,0], [1,0], [2,0], [3,0],
                       [1,1], [2,2], [3,3],
                       [1,2], [2,4], [3,6]]
        # Add 2 more points that create more 3+ point lines
        three_lines.append([0,1])  # might form lines with others
        three_lines.append([0,2])
        candidates.append(three_lines)
    
    # Try a more symmetric configuration
    if time.time() < deadline - 10:
        # 6 points on a line, 6 on another, sharing 2 points
        # Line 1 (y=0): (0,0),(1,0),(2,0),(3,0),(4,0),(5,0)
        # Line 2 (y=1): (0,1),(1,1),(2,1),(3,1),(4,1),(5,1)
        # This is just a 6x2 grid, already covered
        pass
    
    # Evaluate all candidates
    for cand in candidates:
        if len(cand) != 12:
            continue
        # Check not all collinear
        if len(set((p[0], p[1]) for p in cand)) < 12:
            continue
        # Check not all collinear
        if len(cand) >= 3:
            (x1,y1), (x2,y2), (x3,y3) = cand[0], cand[1], cand[2]
            if (x2-x1)*(y3-y1) == (y2-y1)*(x3-x1):
                # Check if ALL are collinear
                all_collinear = True
                for i in range(3, len(cand)):
                    (x4,y4) = cand[i]
                    if (x2-x1)*(y4-y1) != (y2-y1)*(x4-x1):
                        all_collinear = False
                        break
                if all_collinear:
                    continue
        
        ord_count = count_ordinary_lines(cand)
        if ord_count < best_ordinary:
            best_ordinary = ord_count
            best_points = cand
    
    # The theoretical minimum is 6, so if we achieve 6, we can't do better
    # Return the best found
    return [[int(p[0]), int(p[1])] for p in best_points]

# EVOLVE-BLOCK-END
