# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
from math import gcd
from collections import defaultdict


def _normalize_line(A, B, C):
    """Normalize a line (A, B, C) so that:
    - gcd(|A|, |B|, |C|) divides all coefficients (divide by gcd)
    - first nonzero entry is positive
    Returns tuple (a, b, c) with small ints.
    """
    A, B, C = int(A), int(B), int(C)
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A //= g
    B //= g
    C //= g
    # Make first nonzero entry positive
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)


def _count_incidences(points, lines):
    """Count exact incidences using int64 arithmetic."""
    count = 0
    for x, y in points:
        for A, B, C in lines:
            if A * x + B * y + C == 0:
                count += 1
    return count


def _evaluate_point_set(points):
    """Given 54 points, find the top 27 lines by incidence count.
    Returns (incidence_count, lines_array)."""
    line_counts = defaultdict(int)
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            A = dy
            B = -dx
            C = dx * y1 - dy * x1
            norm = _normalize_line(A, B, C)
            if norm == (0, 0, 0):
                continue
            line_counts[norm] += 1
    
    if not line_counts:
        return 0, np.zeros((27, 3), dtype=np.int64)
    
    sorted_lines = sorted(line_counts.items(), key=lambda x: (-x[1], x[0]))
    top27 = sorted_lines[:27]
    
    # Pad if fewer than 27 distinct lines
    while len(top27) < 27:
        top27.append(((1, 0, -999999 + len(top27)), 0))
    
    inc = sum(cnt for _, cnt in top27)
    lines_arr = np.array([norm for norm, _ in top27], dtype=np.int64)
    return inc, lines_arr


def _generate_grid_points(X, Y, x_start=1, y_start=1):
    """Generate an X by Y grid of points."""
    xs = np.arange(x_start, x_start + X)
    ys = np.arange(y_start, y_start + Y)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)


def _generate_parallel_line_points(num_lines, points_per_line, spacing=1, offset=0):
    """Generate points on parallel horizontal lines."""
    points = np.zeros((num_lines * points_per_line, 2), dtype=np.int64)
    for li in range(num_lines):
        for i in range(points_per_line):
            points[li * points_per_line + i, 0] = i * spacing + 1 + offset
            points[li * points_per_line + i, 1] = li
    return points


def _generate_arithmetic_progression_points(n, a, b, c, d):
    """Generate points (a*i + c, b*i + d) for i in 0..n-1."""
    points = np.zeros((n, 2), dtype=np.int64)
    for i in range(n):
        points[i, 0] = a * i + c
        points[i, 1] = b * i + d
    return points


def _generate_mixed_points(config):
    """Generate points based on a configuration dict.
    config: list of (count, x_func, y_func) where x_func and y_func are 
    callables taking index i (0-based within group) and returning x, y.
    """
    points = []
    for count, x_func, y_func in config:
        for i in range(count):
            points.append([x_func(i), y_func(i)])
    return np.array(points, dtype=np.int64)


def szemeredi_trotter_incidences():
    """
    Dual-space hypergraph approach: select 54 points to maximize the sum of
    the top 27 line incidences. We enumerate diverse point configurations and
    for each, compute all lines through point pairs, then select the top 27.
    """
    
    best_inc = -1
    best_points = None
    best_lines = None
    
    def try_points(pts):
        nonlocal best_inc, best_points, best_lines
        if len(pts) != 54:
            return
        # Check for duplicate points
        unique_pts = set(map(tuple, pts.tolist()))
        if len(unique_pts) != 54:
            return
        inc, lines = _evaluate_point_set(pts)
        if inc > best_inc:
            best_inc = inc
            best_points = pts.copy()
            best_lines = lines.copy()
    
    # Strategy A: Grid configurations
    for X in [1, 2, 3, 6, 9, 18, 27, 54]:
        Y = 54 // X
        if X * Y != 54:
            continue
        pts = _generate_grid_points(X, Y)
        try_points(pts)
    
    # Strategy B: Parallel lines with various counts
    for num_lines in [1, 2, 3, 4, 5, 6, 9]:
        if 54 % num_lines != 0:
            continue
        ppl = 54 // num_lines
        pts = _generate_parallel_line_points(num_lines, ppl)
        try_points(pts)
    
    # Strategy C: Two parallel lines with offset
    for offset in range(0, 10):
        pts = _generate_parallel_line_points(2, 27, spacing=1, offset=offset)
        try_points(pts)
    
    # Strategy D: Three parallel lines with offsets
    for off1 in range(0, 5):
        for off2 in range(0, 5):
            pts = np.zeros((54, 2), dtype=np.int64)
            for i in range(18):
                pts[i, 0] = i + 1
                pts[i, 1] = 0
                pts[i + 18, 0] = i + 1 + off1
                pts[i + 18, 1] = 1
                pts[i + 36, 0] = i + 1 + off2
                pts[i + 36, 1] = 2
            try_points(pts)
    
    # Strategy E: Points on a parabola y = x^2
    pts = np.zeros((54, 2), dtype=np.int64)
    for i in range(54):
        pts[i, 0] = i + 1
        pts[i, 1] = (i + 1) ** 2
    try_points(pts)
    
    # Strategy F: Points on a line (all collinear)
    pts = np.zeros((54, 2), dtype=np.int64)
    for i in range(54):
        pts[i, 0] = i + 1
        pts[i, 1] = 0
    try_points(pts)
    
    # Strategy G: 27 points on y=0, 27 on y=1 (interleaved x)
    pts = np.zeros((54, 2), dtype=np.int64)
    for i in range(27):
        pts[i, 0] = 2 * i + 1
        pts[i, 1] = 0
        pts[i + 27, 0] = 2 * i + 2
        pts[i + 27, 1] = 1
    try_points(pts)
    
    # Strategy H: 27 points on y=0, 27 on y=1 (same x)
    pts = np.zeros((54, 2), dtype=np.int64)
    for i in range(27):
        pts[i, 0] = i + 1
        pts[i, 1] = 0
        pts[i + 27, 0] = i + 1
        pts[i + 27, 1] = 1
    try_points(pts)
    
    # Strategy I: 6 groups of 9 on 6 parallel lines
    for spacing in [1, 2, 3]:
        pts = np.zeros((54, 2), dtype=np.int64)
        for li in range(6):
            for i in range(9):
                pts[li * 9 + i, 0] = i * spacing + 1
                pts[li * 9 + i, 1] = li * spacing
        try_points(pts)
    
    # Strategy J: 9 groups of 6 on 9 parallel lines
    for spacing in [1, 2]:
        pts = np.zeros((54, 2), dtype=np.int64)
        for li in range(9):
            for i in range(6):
                pts[li * 6 + i, 0] = i * spacing + 1
                pts[li * 6 + i, 1] = li * spacing
        try_points(pts)
    
    # Strategy K: 3x18 grid with shifted columns
    for shift in range(0, 4):
        pts = np.zeros((54, 2), dtype=np.int64)
        for i in range(3):
            for j in range(18):
                pts[i * 18 + j, 0] = i + 1
                pts[i * 18 + j, 1] = j + 1 + (i * shift) % 3
        try_points(pts)
    
    # Strategy L: 18x3 grid with shifted rows
    for shift in range(0, 4):
        pts = np.zeros((54, 2), dtype=np.int64)
        for i in range(18):
            for j in range(3):
                pts[i * 3 + j, 0] = i + 1 + (j * shift) % 3
                pts[i * 3 + j, 1] = j + 1
        try_points(pts)
    
    # Strategy M: Points at (i, j) for i in 1..6, j in 1..9 with modular shifts
    for s1 in range(3):
        for s2 in range(3):
            pts = np.zeros((54, 2), dtype=np.int64)
            idx = 0
            for i in range(6):
                for j in range(9):
                    pts[idx, 0] = i + 1
                    pts[idx, 1] = j + 1 + ((i * s1 + j * s2) % 3)
                    idx += 1
            try_points(pts)
    
    # Strategy N: Two clusters - 27 on y=0 at x=1..27, 27 on y=2 at x=1..27
    pts = np.zeros((54, 2), dtype=np.int64)
    for i in range(27):
        pts[i, 0] = i + 1
        pts[i, 1] = 0
        pts[i + 27, 0] = i + 1
        pts[i + 27, 1] = 2
    try_points(pts)
    
    # Strategy O: Three clusters - 18 each on y=0, y=2, y=4
    pts = np.zeros((54, 2), dtype=np.int64)
    for i in range(18):
        pts[i, 0] = i + 1
        pts[i, 1] = 0
        pts[i + 18, 0] = i + 1
        pts[i + 18, 1] = 2
        pts[i + 36, 0] = i + 1
        pts[i + 36, 1] = 4
    try_points(pts)
    
    # Strategy P: 27 points on y=0 at even x, 27 on y=1 at odd x
    pts = np.zeros((54, 2), dtype=np.int64)
    for i in range(27):
        pts[i, 0] = 2 * (i + 1)
        pts[i, 1] = 0
        pts[i + 27, 0] = 2 * i + 1
        pts[i + 27, 1] = 1
    try_points(pts)
    
    # Strategy Q: 54 points in a 1x54 line but with spacing
    for gap in [1, 2, 3]:
        pts = np.zeros((54, 2), dtype=np.int64)
        for i in range(54):
            pts[i, 0] = i * gap + 1
            pts[i, 1] = 0
        try_points(pts)
    
    # Strategy R: 2x27 grid with large y-spacing
    for ygap in [1, 2, 3, 5, 10]:
        pts = np.zeros((54, 2), dtype=np.int64)
        for i in range(27):
            pts[i, 0] = i + 1
            pts[i, 1] = 0
            pts[i + 27, 0] = i + 1
            pts[i + 27, 1] = ygap
        try_points(pts)
    
    # Strategy S: 3x18 grid with large y-spacing
    for ygap in [1, 2, 3]:
        pts = np.zeros((54, 2), dtype=np.int64)
        idx = 0
        for i in range(3):
            for j in range(18):
                pts[idx, 0] = i + 1
                pts[idx, 1] = j * ygap + 1
                idx += 1
        try_points(pts)
    
    # Strategy T: 6x9 grid
    for ygap in [1, 2]:
        pts = np.zeros((54, 2), dtype=np.int64)
        idx = 0
        for i in range(6):
            for j in range(9):
                pts[idx, 0] = i + 1
                pts[idx, 1] = j * ygap + 1
                idx += 1
        try_points(pts)
    
    # Strategy U: 9x6 grid
    for ygap in [1, 2]:
        pts = np.zeros((54, 2), dtype=np.int64)
        idx = 0
        for i in range(9):
            for j in range(6):
                pts[idx, 0] = i + 1
                pts[idx, 1] = j * ygap + 1
                idx += 1
        try_points(pts)
    
    # Strategy V: 18x3 grid
    for ygap in [1, 2]:
        pts = np.zeros((54, 2), dtype=np.int64)
        idx = 0
        for i in range(18):
            for j in range(3):
                pts[idx, 0] = i + 1
                pts[idx, 1] = j * ygap + 1
                idx += 1
        try_points(pts)
    
    # Strategy W: 27x2 grid
    for ygap in [1, 2, 3]:
        pts = np.zeros((54, 2), dtype=np.int64)
        idx = 0
        for i in range(27):
            for j in range(2):
                pts[idx, 0] = i + 1
                pts[idx, 1] = j * ygap
                idx += 1
        try_points(pts)
    
    # Strategy X: 54x1 grid (all on one line)
    pts = np.zeros((54, 2), dtype=np.int64)
    for i in range(54):
        pts[i, 0] = i + 1
        pts[i, 1] = 0
    try_points(pts)
    
    # Strategy Y: Points forming a triangle-like structure
    # 18 on each side of a triangle
    pts = np.zeros((54, 2), dtype=np.int64)
    idx = 0
    # Side 1: (1,0) to (18,0)
    for i in range(18):
        pts[idx, 0] = i + 1
        pts[idx, 1] = 0
        idx += 1
    # Side 2: (18,0) to (1, 18) - but we need distinct points
    # Let's use (18,1) to (2,18) roughly
    for i in range(18):
        pts[idx, 0] = 18 - i
        pts[idx, 1] = i + 1
        idx += 1
    # Side 3: (1,18) to (1,0) - but (1,0) is taken
    # Use (2,17) to (1,1)
    for i in range(18):
        pts[idx, 0] = 1 + (i % 2)
        pts[idx, 1] = 18 - i
        idx += 1
    # Check for duplicates
    if len(set(map(tuple, pts.tolist()))) == 54:
        try_points(pts)
    
    # Strategy Z: 27 points on y=0, 27 points forming a line with slope 1
    pts = np.zeros((54, 2), dtype=np.int64)
    for i in range(27):
        pts[i, 0] = i + 1
        pts[i, 1] = 0
    for i in range(27):
        pts[i + 27, 0] = i + 30
        pts[i + 27, 1] = i + 1
    try_points(pts)
    
    # Validate final result
    if best_points is None or best_lines is None:
        # Fallback to 3x18 grid
        k = 3
        xs = np.arange(1, k + 1)
        ys = np.arange(1, 2 * k * k + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        best_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        rows = []
        for a in range(1, k + 1):
            for b in range(1, k * k + 1):
                rows.append((a, -1, b))
        best_lines = np.array(rows, dtype=np.int64)
    
    # Ensure correct shapes
    assert best_points.shape == (54, 2), f"Points shape {best_points.shape}"
    assert best_lines.shape == (27, 3), f"Lines shape {best_lines.shape}"
    
    return best_points, best_lines


# EVOLVE-BLOCK-END
