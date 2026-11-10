# EVOLVE-BLOCK-START
import numpy as np


def szemeredi_trotter_incidences():
    """
    Constructs a set of 54 points and 27 lines to maximize incidences.
    Uses 9 vertical, 9 horizontal, 5 diagonal (slope 1), and 4 diagonal (slope -1) lines.
    This mix creates a denser web of intersections, allowing 10 points to lie on 4 lines
    (1 vertical, 1 horizontal, 1 of each diagonal type) and 44 points on 3 lines.
    Total incidences: 10 * 4 + 44 * 3 = 40 + 132 = 172.
    """
    lines = []
    
    # 9 vertical lines: x = 0..8  ->  (1, 0, -x)
    for i in range(9):
        lines.append((1, 0, -i))
        
    # 9 horizontal lines: y = 0..8  ->  (0, 1, -y)
    for j in range(9):
        lines.append((0, 1, -j))
        
    # 5 diagonal lines with slope 1: y = x + k -> x - y + k = 0 -> (1, -1, k)
    # We choose k to maximize the number of (i,j) in [0,8]x[0,8] such that j-i = k
    # k=0: 9 points
    # k=1: 8 points
    # k=-1: 8 points
    # k=2: 7 points
    # k=-2: 7 points
    k_values = [0, 1, -1, 2, -2]
    for k in k_values:
        lines.append((1, -1, k))
        
    # 4 diagonal lines with slope -1: y = -x + s -> x + y - s = 0 -> (1, 1, -s)
    # We choose s to maximize the number of (i,j) in [0,8]x[0,8] such that i+j = s
    # s=7: 6 points
    # s=8: 5 points
    # s=9: 4 points
    # s=10: 3 points
    s_values = [7, 8, 9, 10]
    for s in s_values:
        lines.append((1, 1, -s))
        
    lines = np.array(lines, dtype=np.int64)
    
    points = []
    # Generate all 81 grid points, calculate incidence count for each
    point_counts = []
    for i in range(9):
        for j in range(9):
            count = 2  # Always on one vertical and one horizontal
            k = j - i
            if k in k_values:
                count += 1
            s = i + j
            if s in s_values:
                count += 1
            point_counts.append((count, i, j))
    
    # Sort by count descending, then by i, j for determinism
    point_counts.sort(key=lambda x: (-x[0], x[1], x[2]))
    
    # Select top 54 points
    selected_points = [(i, j) for (_, i, j) in point_counts[:54]]
    points = np.array(selected_points, dtype=np.int64)
    
    return points, lines


# EVOLVE-BLOCK-END
