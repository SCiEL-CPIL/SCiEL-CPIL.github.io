# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
import time


def _normalize_line(A, B, C):
    """Normalize line (A, B, C) so that gcd(|A|,|B|,|C|)=1 and first nonzero is positive."""
    from math import gcd
    A, B, C = int(A), int(B), int(C)
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _line_from_points(p1, p2):
    """Get line through two points as (A, B, C) with A*x + B*y + C = 0."""
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return _normalize_line(A, B, C)


def _count_incidence(points, lines):
    """Count point-line incidences."""
    n = len(points)
    m = len(lines)
    count = 0
    for i in range(m):
        A, B, C = lines[i]
        for j in range(n):
            x, y = points[j]
            if A * x + B * y + C == 0:
                count += 1
    return count


def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines.
    
    Strategy: Use a 6x9 grid of points, then greedily select 27 lines
    from all lines determined by pairs of grid points to maximize total incidences.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    # Create a 6x9 grid of points
    xs = np.arange(1, 7)  # 6 columns
    ys = np.arange(1, 10)  # 9 rows
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    n_points = len(points)
    
    # Generate all candidate lines from pairs of points
    candidate_lines = {}
    for i in range(n_points):
        for j in range(i + 1, n_points):
            line = _line_from_points(points[i], points[j])
            if line != (0, 0, 0):
                candidate_lines[line] = True
    
    candidate_list = list(candidate_lines.keys())
    
    # Compute incidence count for each candidate line
    line_incidence = {}
    for line in candidate_list:
        A, B, C = line
        cnt = 0
        for p in points:
            x, y = p
            if A * x + B * y + C == 0:
                cnt += 1
        line_incidence[line] = cnt
    
    # Greedy selection: pick lines that maximize total incidences
    # We track which points are "covered" but since points can be on multiple lines,
    # we just want to maximize sum of incidences
    # However, to avoid redundancy, we can use a greedy approach where we pick
    # the line with the highest incidence first, then the next best, etc.
    
    # Sort lines by incidence count descending
    sorted_lines = sorted(candidate_list, key=lambda l: line_incidence[l], reverse=True)
    
    selected_lines = []
    total_incidence = 0
    # Track incidence per point to help with greedy (optional refinement)
    point_counts = np.zeros(n_points, dtype=np.int64)
    
    for line in sorted_lines:
        if len(selected_lines) >= 27:
            break
        A, B, C = line
        # Compute how many new incidences this line adds
        # Actually, since we're just maximizing total, we add all incidences
        cnt = line_incidence[line]
        selected_lines.append(line)
        total_incidence += cnt
        # Update point counts (for potential future use)
        for idx, p in enumerate(points):
            x, y = p
            if A * x + B * y + C == 0:
                point_counts[idx] += 1
        if time.time() > deadline:
            break
    
    # If we didn't get 27 lines (shouldn't happen, but safety)
    while len(selected_lines) < 27:
        # Add arbitrary lines
        for line in candidate_list:
            if line not in selected_lines:
                selected_lines.append(line)
                if len(selected_lines) >= 27:
                    break
    
    lines = np.array(selected_lines, dtype=np.int64)
    
    # Verify validity
    assert points.shape == (54, 2), f"Points shape: {points.shape}"
    assert lines.shape == (27, 3), f"Lines shape: {lines.shape}"
    
    return points, lines


# EVOLVE-BLOCK-END
