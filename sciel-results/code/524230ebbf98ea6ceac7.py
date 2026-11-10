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


def _find_best_lines_for_points(points):
    """Find the 27 lines that maximize incidences for a given point set."""
    line_counts = defaultdict(int)
    n = len(points)
    
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:
                continue
            A = dy
            B = -dx
            C = dx * y1 - dy * x1
            norm = _normalize_line(A, B, C)
            line_counts[norm] += 1
    
    sorted_lines = sorted(line_counts.items(), key=lambda x: -x[1])
    top27 = sorted_lines[:27]
    inc = sum(cnt for _, cnt in top27)
    lines_arr = np.array([norm for norm, _ in top27], dtype=np.int64)
    return inc, lines_arr


def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization via dual-space hypergraph approach.
    
    We search over structured point sets and line sets to maximize incidences.
    The strategy:
    1. Generate candidate point sets from various geometric structures.
    2. For each candidate point set, find the 27 lines that maximize incidences.
    3. Return the best (points, lines) pair.
    
    Key insight: Instead of fixing the grid structure, we explore:
    - Rectangular grids with varying aspect ratios
    - Points concentrated along few lines (high multiplicity)
    - Hybrid structures combining dense clusters with sparse spread
    
    The baseline (k=3 grid) gives 81 incidences. We try to exceed this.
    """
    
    best_inc = -1
    best_points = None
    best_lines = None
    
    # Strategy 1: Original grid construction with k=3
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    rows = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            rows.append((a, -1, b))
    lines = np.array(rows, dtype=np.int64)
    inc = _count_incidences(points, lines)
    if inc > best_inc:
        best_inc = inc
        best_points = points.copy()
        best_lines = lines.copy()
    
    # Strategy 2: Try different grid dimensions
    # Points: X x Y grid where X*Y = 54
    # Lines: try to find 27 lines maximizing incidences
    for X in [1, 2, 3, 6, 9, 18, 27, 54]:
        Y = 54 // X
        if X * Y != 54:
            continue
        
        # Generate grid points
        xs = np.arange(1, X + 1)
        ys = np.arange(1, Y + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        cand_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        
        # Find best 27 lines for this point set
        inc, lines_arr = _find_best_lines_for_points(cand_points)
        
        if inc > best_inc:
            best_inc = inc
            best_points = cand_points.copy()
            best_lines = lines_arr.copy()
    
    # Strategy 3: Concentrate points on few lines
    # Put many points on 1-3 lines, rest scattered
    # E.g., 27 points on line y=0, 27 points on line y=1
    # Then lines y=0 and y=1 each have 27 incidences, plus lines through pairs
    
    # Try: 27 points on y=0 at x=1..27, 27 points on y=1 at x=1..27
    points3 = np.zeros((54, 2), dtype=np.int64)
    for i in range(27):
        points3[i] = [i + 1, 0]
        points3[i + 27] = [i + 1, 1]
    
    inc3, lines3 = _find_best_lines_for_points(points3)
    if inc3 > best_inc:
        best_inc = inc3
        best_points = points3.copy()
        best_lines = lines3.copy()
    
    # Strategy 4: 18 points on y=0, 18 on y=1, 18 on y=2
    points4 = np.zeros((54, 2), dtype=np.int64)
    for i in range(18):
        points4[i] = [i + 1, 0]
        points4[i + 18] = [i + 1, 1]
        points4[i + 36] = [i + 1, 2]
    
    inc4, lines4 = _find_best_lines_for_points(points4)
    if inc4 > best_inc:
        best_inc = inc4
        best_points = points4.copy()
        best_lines = lines4.copy()
    
    # Strategy 5: 9 points on each of 6 lines
    points5 = np.zeros((54, 2), dtype=np.int64)
    for line_idx in range(6):
        for i in range(9):
            points5[line_idx * 9 + i] = [i + 1, line_idx]
    
    inc5, lines5 = _find_best_lines_for_points(points5)
    if inc5 > best_inc:
        best_inc = inc5
        best_points = points5.copy()
        best_lines = lines5.copy()
    
    # Strategy 6: 3x18 grid with optimized lines
    # Already covered in Strategy 2 with X=3, Y=18
    
    # Strategy 7: 6x9 grid
    # Already covered in Strategy 2 with X=6, Y=9
    
    # Strategy 8: 9x6 grid
    # Already covered in Strategy 2 with X=9, Y=6
    
    # Strategy 9: Try non-uniform distributions
    # Put points at positions that maximize collinearities
    # E.g., use a set where many points are collinear
    
    # Try: points at (i, j) for i in 1..6, j in 1..9 but shifted
    # Or use modular arithmetic patterns
    
    # Strategy 10: 2x27 grid
    # Already covered in Strategy 2
    
    # Strategy 11: 1x54 (all on one line)
    # Already covered in Strategy 2 with X=1, Y=54
    
    # Strategy 12: 54x1
    # Already covered in Strategy 2 with X=54, Y=1
    
    # Let's also try: points on a circle or parabola
    # Parabola: y = x^2 for x in 1..54, but that's 54 distinct points
    # Lines through pairs of parabola points
    
    points12 = np.zeros((54, 2), dtype=np.int64)
    for i in range(54):
        points12[i] = [i + 1, (i + 1) ** 2]
    
    inc12, lines12 = _find_best_lines_for_points(points12)
    if inc12 > best_inc:
        best_inc = inc12
        best_points = points12.copy()
        best_lines = lines12.copy()
    
    # Strategy 13: Try 3 parallel lines with 18 points each, but offset
    points13 = np.zeros((54, 2), dtype=np.int64)
    for line_idx in range(3):
        for i in range(18):
            points13[line_idx * 18 + i] = [i + 1 + line_idx * 5, line_idx * 10]
    
    inc13, lines13 = _find_best_lines_for_points(points13)
    if inc13 > best_inc:
        best_inc = inc13
        best_points = points13.copy()
        best_lines = lines13.copy()
    
    # Strategy 14: 2 lines with 27 points each, but interleaved to create more cross-lines
    points14 = np.zeros((54, 2), dtype=np.int64)
    for i in range(27):
        points14[i] = [2 * i + 1, 0]  # odd x on y=0
        points14[i + 27] = [2 * i + 2, 1]  # even x on y=1
    
    inc14, lines14 = _find_best_lines_for_points(points14)
    if inc14 > best_inc:
        best_inc = inc14
        best_points = points14.copy()
        best_lines = lines14.copy()
    
    # Strategy 15: 4 lines with 13-14 points each
    points15 = np.zeros((54, 2), dtype=np.int64)
    counts = [14, 14, 13, 13]
    idx = 0
    for line_idx in range(4):
        for i in range(counts[line_idx]):
            points15[idx] = [i + 1, line_idx * 3]
            idx += 1
    
    inc15, lines15 = _find_best_lines_for_points(points15)
    if inc15 > best_inc:
        best_inc = inc15
        best_points = points15.copy()
        best_lines = lines15.copy()
    
    # Strategy 16: 5 lines with ~11 points each
    points16 = np.zeros((54, 2), dtype=np.int64)
    counts = [11, 11, 11, 11, 10]
    idx = 0
    for line_idx in range(5):
        for i in range(counts[line_idx]):
            points16[idx] = [i + 1, line_idx * 4]
            idx += 1
    
    inc16, lines16 = _find_best_lines_for_points(points16)
    if inc16 > best_inc:
        best_inc = inc16
        best_points = points16.copy()
        best_lines = lines16.copy()
    
    # Strategy 17: 7 lines with 8 points each (56 total, use 54)
    points17 = np.zeros((54, 2), dtype=np.int64)
    counts = [8, 8, 8, 8, 8, 8, 6]
    idx = 0
    for line_idx in range(7):
        for i in range(counts[line_idx]):
            points17[idx] = [i + 1, line_idx * 5]
            idx += 1
    
    inc17, lines17 = _find_best_lines_for_points(points17)
    if inc17 > best_inc:
        best_inc = inc17
        best_points = points17.copy()
        best_lines = lines17.copy()
    
    # Strategy 18: 9 lines with 6 points each
    points18 = np.zeros((54, 2), dtype=np.int64)
    for line_idx in range(9):
        for i in range(6):
            points18[line_idx * 6 + i] = [i + 1, line_idx * 7]
    
    inc18, lines18 = _find_best_lines_for_points(points18)
    if inc18 > best_inc:
        best_inc = inc18
        best_points = points18.copy()
        best_lines = lines18.copy()
    
    # Strategy 19: 18 lines with 3 points each
    points19 = np.zeros((54, 2), dtype=np.int64)
    for line_idx in range(18):
        for i in range(3):
            points19[line_idx * 3 + i] = [i + 1, line_idx * 11]
    
    inc19, lines19 = _find_best_lines_for_points(points19)
    if inc19 > best_inc:
        best_inc = inc19
        best_points = points19.copy()
        best_lines = lines19.copy()
    
    # Strategy 20: 27 lines with 2 points each
    points20 = np.zeros((54, 2), dtype=np.int64)
    for line_idx in range(27):
        for i in range(2):
            points20[line_idx * 2 + i] = [i + 1, line_idx * 13]
    
    inc20, lines20 = _find_best_lines_for_points(points20)
    if inc20 > best_inc:
        best_inc = inc20
        best_points = points20.copy()
        best_lines = lines20.copy()
    
    # Strategy 21: Mixed - some dense clusters, some spread
    # 3 clusters of 18 points on parallel lines, plus cross connections
    points21 = np.zeros((54, 2), dtype=np.int64)
    for line_idx in range(3):
        for i in range(18):
            points21[line_idx * 18 + i] = [i + 1, line_idx * 20]
    
    inc21, lines21 = _find_best_lines_for_points(points21)
    if inc21 > best_inc:
        best_inc = inc21
        best_points = points21.copy()
        best_lines = lines21.copy()
    
    # Strategy 22: 6x9 grid with different spacing
    points22 = np.zeros((54, 2), dtype=np.int64)
    idx = 0
    for i in range(6):
        for j in range(9):
            points22[idx] = [i * 3 + 1, j * 3 + 1]
            idx += 1
    
    inc22, lines22 = _find_best_lines_for_points(points22)
    if inc22 > best_inc:
        best_inc = inc22
        best_points = points22.copy()
        best_lines = lines22.copy()
    
    # Strategy 23: 9x6 grid with different spacing
    points23 = np.zeros((54, 2), dtype=np.int64)
    idx = 0
    for i in range(9):
        for j in range(6):
            points23[idx] = [i * 3 + 1, j * 3 + 1]
            idx += 1
    
    inc23, lines23 = _find_best_lines_for_points(points23)
    if inc23 > best_inc:
        best_inc = inc23
        best_points = points23.copy()
        best_lines = lines23.copy()
    
    # Strategy 24: 3x18 grid with different spacing
    points24 = np.zeros((54, 2), dtype=np.int64)
    idx = 0
    for i in range(3):
        for j in range(18):
            points24[idx] = [i * 5 + 1, j * 2 + 1]
            idx += 1
    
    inc24, lines24 = _find_best_lines_for_points(points24)
    if inc24 > best_inc:
        best_inc = inc24
        best_points = points24.copy()
        best_lines = lines24.copy()
    
    # Strategy 25: 18x3 grid with different spacing
    points25 = np.zeros((54, 2), dtype=np.int64)
    idx = 0
    for i in range(18):
        for j in range(3):
            points25[idx] = [i * 2 + 1, j * 5 + 1]
            idx += 1
    
    inc25, lines25 = _find_best_lines_for_points(points25)
    if inc25 > best_inc:
        best_inc = inc25
        best_points = points25.copy()
        best_lines = lines25.copy()
    
    # Strategy 26: 2x27 grid with different spacing
    points26 = np.zeros((54, 2), dtype=np.int64)
    idx = 0
    for i in range(2):
        for j in range(27):
            points26[idx] = [i * 7 + 1, j * 2 + 1]
            idx += 1
    
    inc26, lines26 = _find_best_lines_for_points(points26)
    if inc26 > best_inc:
        best_inc = inc26
        best_points = points26.copy()
        best_lines = lines26.copy()
    
    # Strategy 27: 27x2 grid with different spacing
    points27 = np.zeros((54, 2), dtype=np.int64)
    idx = 0
    for i in range(27):
        for j in range(2):
            points27[idx] = [i * 2 + 1, j * 7 + 1]
            idx += 1
    
    inc27, lines27 = _find_best_lines_for_points(points27)
    if inc27 > best_inc:
        best_inc = inc27
        best_points = points27.copy()
        best_lines = lines27.copy()
    
    # Validate final result
    if best_points is None or best_lines is None:
        # Fallback to original
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
