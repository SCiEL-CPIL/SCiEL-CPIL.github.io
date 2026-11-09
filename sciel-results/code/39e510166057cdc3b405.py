# EVOLVE-BLOCK-START
import numpy as np
from math import gcd
from itertools import combinations

def szemeredi_trotter_incidences():
    # Build a 6x9 grid of 54 points
    pts = [(x, y) for x in range(6) for y in range(9)]
    n_pts = len(pts)
    
    # Collect all unique lines determined by pairs of points
    line_set = {}
    
    def norm(A, B, C):
        g = gcd(abs(A), gcd(abs(B), abs(C)))
        if g == 0:
            return (0, 0, 0)
        A //= g
        B //= g
        C //= g
        if A < 0 or (A == 0 and B < 0):
            A, B, C = -A, -B, -C
        return (A, B, C)
    
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            key = norm(A, B, C)
            if key not in line_set:
                line_set[key] = 0
    
    # For each line, count incidences with all 54 points
    line_data = []
    for (A, B, C) in line_set:
        cnt = 0
        for (x, y) in pts:
            if A * x + B * y + C == 0:
                cnt += 1
        line_data.append(((A, B, C), cnt))
    
    # Sort by incidence count descending, then lexicographically for determinism
    line_data.sort(key=lambda t: (-t[1], t[0]))
    
    # Greedy select top 27 lines
    # For additive objective with cardinality constraint, selecting the 27 lines
    # with highest individual counts is optimal (no interaction term)
    selected = line_data[:27]
    
    points_arr = np.array(pts, dtype=np.int64)
    lines_arr = np.array([item[0] for item in selected], dtype=np.int64)
    
    return points_arr, lines_arr
# EVOLVE-BLOCK-END
