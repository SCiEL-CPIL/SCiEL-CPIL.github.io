# EVOLVE-BLOCK-START
import numpy as np
from math import gcd
from itertools import product

def _normalize_line(A, B, C):
    """Normalize line (A,B,C) so first nonzero of (A,B) is positive, and gcd(|A|,|B|,|C|)=1."""
    if A == 0 and B == 0:
        return None
    g = gcd(gcd(abs(A), abs(B)), abs(C)) if C != 0 else gcd(abs(A), abs(B))
    if g == 0:
        g = 1
    A, B, C = A // g, B // g, C // g
    # Make first nonzero entry positive
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidence(point, A, B, C):
    """Check if point (x,y) lies on line Ax + By + C = 0."""
    return A * point[0] + B * point[1] + C == 0

def szemeredi_trotter_incidences():
    """
    Constructs a 6x9 grid of points and selects 27 lines to maximize incidences.
    Uses greedy selection over all candidate lines.
    """
    # Generate points: 6 columns, 9 rows
    xs = np.arange(1, 7)  # 1 to 6
    ys = np.arange(1, 10) # 1 to 9
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    n_points = len(points)
    
    # Generate all candidate lines that pass through at least 2 points
    # A line is determined by any two distinct points
    candidate_lines = set()
    
    for i in range(n_points):
        for j in range(i + 1, n_points):
            x1, y1 = points[i]
            x2, y2 = points[j]
            # Line through (x1,y1) and (x2,y2):
            # (y2-y1)*x - (x2-x1)*y + (x2-x1)*y1 - (y2-y1)*x2 = 0
            A = y2 - y1
            B = -(x2 - x1)
            C = (x2 - x1) * y1 - (y2 - y1) * x2
            
            normalized = _normalize_line(A, B, C)
            if normalized is not None:
                candidate_lines.add(normalized)
    
    # Count incidences for each candidate line
    line_incidents = {}
    for line in candidate_lines:
        A, B, C = line
        count = 0
        for p in points:
            if A * p[0] + B * p[1] + C == 0:
                count += 1
        line_incidents[line] = count
    
    # Sort lines by incidence count (descending)
    sorted_lines = sorted(line_incidents.items(), key=lambda x: -x[1])
    
    # Greedily select 27 lines with highest incidences
    # We need to be careful: selecting a line doesn't affect incidence counts of other lines
    # (each point-line pair is independent), so greedy is optimal here.
    selected_lines = []
    for line, count in sorted_lines:
        if len(selected_lines) == 27:
            break
        selected_lines.append(line)
    
    lines = np.array(selected_lines, dtype=np.int64)
    
    return points, lines

# EVOLVE-BLOCK-END
