import numpy as np
from math import gcd

def szemeredi_trotter_incidences():
    # Core: 3x3 grid at (0..2, 0..2)
    core_points = []
    for x in range(3):
        for y in range(3):
            core_points.append((x, y))
            
    core_lines = []
    # 3 horizontal
    for y in range(3):
        core_lines.append((0, 1, -y)) # 0*x + 1*y - y = 0
    # 3 vertical
    for x in range(3):
        core_lines.append((1, 0, -x)) # 1*x + 0*y - x = 0
    # 2 diagonals
    core_lines.append((1, -1, 0)) # x - y = 0
    core_lines.append((1, 1, -2)) # x + y - 2 = 0
    
    # 12 pair lines for remaining pairs
    # Pairs in 3x3 grid:
    # (0,0)-(1,1), (0,1)-(1,0), (0,2)-(1,1) -- wait, (0,2)-(1,1) is on y=-x+2? No.
    # Let's just generate all pairs and filter out those on existing lines.
    existing_line_set = set()
    for A, B, C in core_lines:
        # Normalize
        g = gcd(abs(A), gcd(abs(B), abs(C)))
        if g > 0:
            A, B, C = A//g, B//g, C//g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A, B, C = -A, -B, -C
        existing_line_set.add((A, B, C))
        
    pair_lines = []
    points_list = core_points
    for i in range(len(points_list)):
        for j in range(i+1, len(points_list)):
            x1, y1 = points_list[i]
            x2, y2 = points_list[j]
            # Line through (x1,y1) and (x2,y2)
            # (y2-y1)x - (x2-x1)y + (x2-x1)y1 - (y2-y1)x2 = 0
            A = y2 - y1
            B = -(x2 - x1)
            C = (x2 - x1)*y1 - (y2 - y1)*x2
            # Normalize
            g = gcd(abs(A), gcd(abs(B), abs(C)))
            if g > 0:
                A, B, C = A//g, B//g, C//g
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A, B, C = -A, -B, -C
            key = (A, B, C)
            if key not in existing_line_set:
                existing_line_set.add(key)
                pair_lines.append((A, B, C))
                
    # We need exactly 12 pair lines.
    # The 3x3 grid has 36 pairs. 8 lines cover 24 pairs.
    # 12 pairs remain. Each defines a unique line.
    # So we should get exactly 12 lines.
    assert len(pair_lines) == 12, f"Expected 12 pair lines, got {len(pair_lines)}"
    
    all_lines = core_lines + pair_lines
    
    # Outer: 3x15 grid at x in {10, 11, 12}, y in {10..24}
    outer_points = []
    for x in [10, 11, 12]:
        for y in range(10, 25):
            outer_points.append((x, y))
            
    outer_lines = []
    # 3 vertical lines (size 15)
    for x in [10, 11, 12]:
        outer_lines.append((1, 0, -x))
    # 4 horizontal lines (size 3)
    for y in range(10, 14): # 10, 11, 12, 13
        outer_lines.append((0, 1, -y))
        
    # Combine
    all_points = core_points + outer_points
    all_lines = all_lines + outer_lines
    
    # Convert to numpy arrays
    points = np.array(all_points, dtype=np.int64)
    lines = np.array(all_lines, dtype=np.int64)
    
    return points, lines
