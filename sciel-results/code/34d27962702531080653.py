import numpy as np
from math import gcd

def _normalize_line(A, B, C):
    """
    Normalize a line (A, B, C) such that:
    1. gcd(|A|, |B|, |C|) = 1
    2. The first non-zero entry among A, B is positive.
    3. If A=0 and B=0, C is positive (though A=B=0 is invalid input, we handle it).
    """
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    
    # Make first non-zero of A, B positive
    if A < 0:
        A = -A
        B = -B
        C = -C
    elif A == 0 and B < 0:
        B = -B
        C = -C
        
    return (A, B, C)

def _count_incidence(line, points):
    """
    Count points on the line.
    line: (A, B, C)
    points: Nx2 array
    """
    A, B, C = line
    # A*x + B*y + C == 0
    # Use integer arithmetic
    # points[:, 0] * A + points[:, 1] * B + C == 0
    vals = points[:, 0] * A + points[:, 1] * B + C
    return np.sum(vals == 0)

def _generate_candidate_lines(points):
    """
    Generate all unique lines passing through at least 2 points.
    """
    lines = set()
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            # Line through (x1, y1) and (x2, y2)
            # A = y1 - y2
            # B = x2 - x1
            # C = x1*y2 - x2*y1
            A = y1 - y2
            B = x2 - x1
            C = x1 * y2 - x2 * y1
            norm = _normalize_line(A, B, C)
            if norm != (0, 0, 0):
                lines.add(norm)
    return list(lines)

def szemeredi_trotter_incidences():
    # Grid 6 x 9
    k = 6
    M = 9
    xs = np.arange(1, k + 1)
    ys = np.arange(1, M + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Generate candidate lines
    candidates = _generate_candidate_lines(points)
    
    # Calculate incidence for each
    scored_lines = []
    for line in candidates:
        inc = _count_incidence(line, points)
        scored_lines.append((inc, line))
        
    # Sort by incidence descending, then by line tuple for determinism
    scored_lines.sort(key=lambda x: (-x[0], x[1]))
    
    # Pick top 27
    top_lines = [line for _, line in scored_lines[:27]]
    
    lines_array = np.array(top_lines, dtype=np.int64)
    
    return points, lines_array
