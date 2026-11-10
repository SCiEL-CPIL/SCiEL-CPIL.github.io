import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    start_time = time.time()
    deadline = start_time + 600
    
    # Element (a, b) maps to a + 7*b
    def to_int(a, b):
        return a + 7 * b
    
    def to_pair(x):
        return (x % 7, x // 7)
    
    # Field multiplication table
    mul_table = np.zeros((49, 49), dtype=np.int64)
    for i in range(49):
        a, b = i % 7, i // 7
        for j in range(49):
            c, d = j % 7, j // 7
            ra = (a * c + 3 * b * d) % p
            rb = (a * d + b * c) % p
            mul_table[i, j] = to_int(ra, rb)
    
    # Point (x, y) maps to x + 49*y
    total_points = 2401
    num_slopes = 49
    num_lines = num_slopes * 49 + 49  # 2450
    
    # line_counts: for each line, how many points are currently in N
    line_counts = np.full(num_lines, 49, dtype=np.int64)
    
    # in_N: whether each point is in N
    in_N = np.ones(total_points, dtype=bool)
    
    # Precompute for each point, the list of line IDs passing through it
    # point_lines[point_idx] = list of 50 line IDs
    point_lines = np.zeros((total_points, 50), dtype=np.int64)
    
    for x in range(49):
        for y in range(49):
            pt = x + 49 * y
            for m in range(49):
                # c = y - m*x (field arithmetic)
                mx = mul_table[m, x]
                c = (y - mx) % 49
                line_id = m * 49 + c
                point_lines[pt, m] = line_id
            # Vertical
            point_lines[pt, 49] = 49 * 49 + x
    
    # Greedy: try to remove points
    # Order: random shuffle for diversity
    random.seed(42)
    order = list(range(total_points))
    random.shuffle(order)
    
    removed = 0
    for pt in order:
        if not in_N[pt]:
            continue
        
        # Check if any line through pt has count == 49
        can_remove = False
        for i in range(50):
            lid = point_lines[pt, i]
            if line_counts[lid] == 49:
                can_remove = True
                break
        
        if can_remove:
            in_N[pt] = False
            removed += 1
            # Decrement line counts for all 50 lines through pt
            for i in range(50):
                lid = point_lines[pt, i]
                line_counts[lid] -= 1
    
    # Build result
    points = []
    for idx in range(total_points):
        if in_N[idx]:
            x = idx % 49
            y = idx // 49
            xa, xb = to_pair(x)
            ya, yb = to_pair(y)
            points.append([[xa, xb], [ya, yb]])
    
    result = np.array(points, dtype=np.int64)
    return result
