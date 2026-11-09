# EVOLVE-BLOCK-START
import time
import numpy as np
from math import gcd
from functools import reduce


def _gcd3(a, b, c):
    return reduce(gcd, (abs(int(a)), abs(int(b)), abs(int(c))))


def _normalize(A, B, C):
    """Return a canonical integer triple for the line A*x + B*y + C = 0."""
    A, B, C = int(A), int(B), int(C)
    g = _gcd3(A, B, C)
    if g != 0:
        A //= g
        B //= g
        C //= g
    # Make the first nonzero entry positive
    for v in (A, B, C):
        if v != 0:
            if v < 0:
                A, B, C = -A, -B, -C
            break
    return (A, B, C)


def szemeredi_trotter_incidences():
    """
    Model-and-optimize: parameterize by the choice of 27 lines; optimize directly
    against the incidence objective.

    Strategy:
      1. Fix a 54-point lattice grid (6 x 9 = 54) that admits many collinear
         subsets (horizontal lines hit 6 points, steep diagonals hit several).
      2. Enumerate every line that passes through at least 2 of the grid points.
      3. Count, for each such line, how many of the 54 grid points it contains.
      4. Select the 27 distinct lines with the largest counts (ties broken
         deterministically). The total incidence count is the sum of those 27
         counts, which directly maximizes the objective for this point set.

    A deadline guard returns the best result found within the wall-clock budget.
    """
    t_start = time.time()
    deadline = 630.0
    margin = 30.0  # seconds of safety before the 630s kill

    # --- Point set: try multiple grid shapes, pick the one with best top-27 sum
    # For an m x n grid (m*n = 54), horizontal lines have m incidences,
    # vertical have n, and various diagonals contribute.
    # 54 = 1*54, 2*27, 3*18, 6*9
    # Try 6x9 (current) and also 3x18, 2*27 to see which gives better top-27 sum.
    
    def _build_grid(m, n):
        xs = np.arange(1, m + 1, dtype=np.int64)
        ys = np.arange(1, n + 1, dtype=np.int64)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    def _score_grid(pts):
        """Compute sum of top-27 line incidence counts for a point set."""
        line_counts = {}
        n_pts = len(pts)
        for i in range(n_pts):
            xi, yi = int(pts[i, 0]), int(pts[i, 1])
            for j in range(i + 1, n_pts):
                xj, yj = int(pts[j, 0]), int(pts[j, 1])
                A = yi - yj
                B = xj - xi
                C = -(xi * yj - xj * yi)
                key = _normalize(A, B, C)
                if key not in line_counts:
                    line_counts[key] = 0
        
        for key in line_counts:
            A, B, C = key
            vals = A * pts[:, 0] + B * pts[:, 1] + C
            line_counts[key] = int(np.count_nonzero(vals == 0))
        
        candidates = [(cnt, key) for key, cnt in line_counts.items() if cnt >= 2]
        candidates.sort(key=lambda t: (-t[0], t[1]))
        
        total = 0
        seen = set()
        count = 0
        for cnt, key in candidates:
            if key in seen:
                continue
            seen.add(key)
            total += cnt
            count += 1
            if count == 27:
                break
        return total
    
    # Try different grid shapes
    best_points = None
    best_score = -1
    for m, n in [(6, 9), (3, 18)]:
        pts = _build_grid(m, n)
        s = _score_grid(pts)
        if s > best_score:
            best_score = s
            best_points = pts
        if time.time() > deadline - margin:
            break
    
    points = best_points

    # --- Enumerate all lines through >=2 grid points, count incidences ------
    line_counts = {}

    n = len(points)
    for i in range(n):
        xi, yi = int(points[i, 0]), int(points[i, 1])
        for j in range(i + 1, n):
            xj, yj = int(points[j, 0]), int(points[j, 1])
            A = yi - yj
            B = xj - xi
            C = -(xi * yj - xj * yi)
            key = _normalize(A, B, C)
            if key not in line_counts:
                line_counts[key] = 0

    # Count incidences for each candidate line
    for key in list(line_counts.keys()):
        A, B, C = key
        vals = A * points[:, 0] + B * points[:, 1] + C
        line_counts[key] = int(np.count_nonzero(vals == 0))

    # Keep only lines with >= 2 incidences (valid, non-degenerate by construction)
    candidates = [(cnt, key) for key, cnt in line_counts.items() if cnt >= 2]

    # Sort by count descending, then by the normalized triple for determinism
    candidates.sort(key=lambda t: (-t[0], t[1]))

    # Select the top 27 distinct lines
    selected = []
    seen = set()
    for cnt, key in candidates:
        if key in seen:
            continue
        seen.add(key)
        selected.append((cnt, key))
        if len(selected) == 27:
            break

    # If (unexpectedly) we have fewer than 27, top up with any remaining valid lines
    if len(selected) < 27:
        for cnt, key in candidates:
            if key in seen:
                continue
            seen.add(key)
            selected.append((cnt, key))
            if len(selected) == 27:
                break

    if len(selected) < 27:
        # Should not happen (grid has far more than 27 valid lines), but guard.
        # Fall back to the Szemeredi-Trotter seed construction.
        k = 3
        xs2 = np.arange(1, k + 1)
        ys2 = np.arange(1, 2 * k * k + 1)
        px2, py2 = np.meshgrid(xs2, ys2, indexing="ij")
        points = np.stack([px2.ravel(), py2.ravel()], axis=1).astype(np.int64)
        rows = []
        for a in range(1, k + 1):
            for b in range(1, k * k + 1):
                rows.append((a, -1, b))
        lines = np.array(rows, dtype=np.int64)
        return points, lines

    lines = np.array([key for _, key in selected], dtype=np.int64)

    # --- Deadline guard -----------------------------------------------------
    if time.time() > deadline - margin:
        # Return the best result found (which is already the full construction).
        pass

    return points, lines


# EVOLVE-BLOCK-END
