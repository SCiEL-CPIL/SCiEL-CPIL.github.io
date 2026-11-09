# EVOLVE-BLOCK-START
import numpy as np
from math import gcd, isqrt


def _canonical_line(A, B, C):
    """Return canonical integer triple (a,b,c) for line A*x+B*y+C=0.
    Divide by gcd(|A|,|B|,|C|), make first nonzero positive.
    """
    if A == 0 and B == 0:
        raise ValueError("Zero line")
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    a, b, c = A // g, B // g, C // g
    if a < 0 or (a == 0 and b < 0) or (a == 0 and b == 0 and c < 0):
        a, b, c = -a, -b, -c
    return (a, b, c)


def _count_incidents(points, lines):
    """Count point-line incidences exactly."""
    pts = np.asarray(points, dtype=np.int64)
    lns = np.asarray(lines, dtype=np.int64)
    # A*x + B*y + C == 0
    vals = lns[:, 0:1] * pts[:, 0:1].T + lns[:, 1:2] * pts[:, 1:2].T + lns[:, 2:3]
    return int(np.count_nonzero(vals == 0))


def _build_grid_and_diagonals(nx, ny, num_v, num_h, num_diag, slope_set=None):
    """Build a grid of nx*ny points with vertical, horizontal, and diagonal lines.
    Points: x in {0..nx-1}, y in {0..ny-1}
    Lines chosen to maximize incidences: vertical (nx pts), horizontal (ny pts),
    and diagonals with slopes from slope_set.
    """
    # Points
    xs = np.arange(nx, dtype=np.int64)
    ys = np.arange(ny, dtype=np.int64)
    px, py = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

    lines = []
    seen = set()

    # Vertical lines: x = c  ->  1*x + 0*y - c = 0
    for c in range(nx):
        canon = _canonical_line(1, 0, -c)
        if canon not in seen:
            seen.add(canon)
            lines.append((1, 0, -c))
        if len(lines) >= num_v:
            break

    # Horizontal lines: y = c  ->  0*x + 1*y - c = 0
    for c in range(ny):
        canon = _canonical_line(0, 1, -c)
        if canon not in seen:
            seen.add(canon)
            lines.append((0, 1, -c))
        if len(lines) >= num_h:
            break

    # Diagonal lines: for each slope in slope_set, enumerate intercepts
    if slope_set is None:
        slope_set = [1, -1]

    # Collect all diagonal candidates with their incidence counts
    candidates = []
    for slope in slope_set:
        # Line: y = slope*x + b  ->  slope*x - y + b = 0  ->  (A,B,C) = (slope, -1, b)
        # For point (x,y) on grid: y = slope*x + b => b = y - slope*x
        # b ranges: min_b = 0 - slope*(nx-1) if slope>0 else 0 - slope*0 = 0
        # Actually b = y - slope*x, x in [0,nx-1], y in [0,ny-1]
        if slope >= 0:
            b_min = 0 - slope * (nx - 1)
            b_max = (ny - 1) - slope * 0
        else:
            b_min = 0 - slope * 0
            b_max = (ny - 1) - slope * (nx - 1)
        
        for b in range(int(b_min), int(b_max) + 1):
            # Count points on this line
            count = 0
            for x in range(nx):
                y = slope * x + b
                if 0 <= y < ny:
                    count += 1
            if count >= 2:  # Only consider lines with at least 2 points
                canon = _canonical_line(slope, -1, b)
                if canon not in seen:
                    candidates.append((count, canon, (slope, -1, b)))
    
    # Sort by incidence count descending, break ties by canonical form
    candidates.sort(key=lambda t: (-t[0], t[1]))
    for count, canon, line in candidates:
        if len(lines) >= num_v + num_h + num_diag:
            break
        if canon not in seen:
            seen.add(canon)
            lines.append(line)

    # Pad with additional lines if needed
    if len(lines) < num_v + num_h + num_diag:
        # Add lines with 1 point or other slopes
        extra_slopes = list(set(range(-5, 6)) - set(slope_set))
        extra_slopes.sort()
        for slope in extra_slopes:
            if len(lines) >= num_v + num_h + num_diag:
                break
            if slope >= 0:
                b_min = 0 - slope * (nx - 1)
                b_max = (ny - 1) - slope * 0
            else:
                b_min = 0 - slope * 0
                b_max = (ny - 1) - slope * (nx - 1)
            for b in range(int(b_min), int(b_max) + 1):
                if len(lines) >= num_v + num_h + num_diag:
                    break
                canon = _canonical_line(slope, -1, b)
                if canon not in seen:
                    seen.add(canon)
                    lines.append((slope, -1, b))

    lines_arr = np.array(lines[:num_v + num_h + num_diag], dtype=np.int64)
    return points, lines_arr


def _try_all_slopes(nx, ny, num_v, num_h, num_diag):
    """Try all subsets of slopes and find best configuration."""
    best_points = None
    best_lines = None
    best_inc = 0

    all_slopes = list(range(-4, 5))  # slopes -4 to 4
    n_slopes = len(all_slopes)

    # Try all 2^15 = 32768 subsets (feasible)
    # But first try small sets
    for mask in range(1, 1 << n_slopes):
        slope_set = [all_slopes[i] for i in range(n_slopes) if mask & (1 << i)]
        if not slope_set:
            continue
        try:
            points, lines = _build_grid_and_diagonals(nx, ny, num_v, num_h, num_diag, slope_set)
            if lines.shape[0] < num_v + num_h + num_diag:
                continue
            inc = _count_incidents(points, lines)
            if inc > best_inc:
                best_inc = inc
                best_points = points.copy()
                best_lines = lines.copy()
        except Exception:
            continue

    return best_points, best_lines, best_inc


def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines on integer lattice.
    Uses grid + diagonal construction, searching over slope sets.
    """
    best_points = None
    best_lines = None
    best_inc = 0

    # Strategy 1: Grid with verticals, horizontals, and diagonals
    # nx * ny = 54
    # Try various (nx, ny) pairs and slope allocations
    for nx, ny in [(6, 9), (9, 6), (54, 1), (1, 54), (2, 27), (27, 2), (3, 18), (18, 3), (18, 3)]:
        if nx * ny != 54:
            continue
        for num_v in range(0, min(nx, 28) + 1):
            for num_h in range(0, min(ny, 28 - num_v) + 1):
                num_diag = 27 - num_v - num_h
                if num_diag < 0:
                    continue
                # Try a few slope sets
                for slope_set in [[1, -1], [1], [-1], [1, -1, 2, -2], [1, -1, 2, -2, 3, -3]]:
                    try:
                        points, lines = _build_grid_and_diagonals(nx, ny, num_v, num_h, num_diag, slope_set)
                        if lines.shape[0] < 27:
                            continue
                        inc = _count_incidents(points, lines)
                        if inc > best_inc:
                            best_inc = inc
                            best_points = points.copy()
                            best_lines = lines.copy()
                    except Exception:
                        continue

    # Strategy 2: Try the pure Szemeredi-Trotter construction
    k = 3
    xs = np.arange(1, k + 1, dtype=np.int64)
    ys = np.arange(1, 2 * k * k + 1, dtype=np.int64)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points_st = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    rows = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            rows.append((a, -1, b))
    lines_st = np.array(rows, dtype=np.int64)
    inc_st = _count_incidents(points_st, lines_st)
    if inc_st > best_inc:
        best_inc = inc_st
        best_points = points_st.copy()
        best_lines = lines_st.copy()

    # Strategy 3: Try a 6x9 grid with all slopes enumerated
    try:
        pts, lns, inc = _try_all_slopes(6, 9, 0, 0, 27)
        if inc > best_inc:
            best_inc = inc
            best_points = pts
            best_lines = lns
    except Exception:
        pass

    # Strategy 4: Try 6x9 with various v/h/diag splits more exhaustively
    for num_v in [0, 1, 2, 3, 4, 5, 6]:
        for num_h in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            num_diag = 27 - num_v - num_h
            if num_diag < 0:
                continue
            try:
                pts, lns = _build_grid_and_diagonals(6, 9, num_v, num_h, num_diag, [1, -1])
                if lns.shape[0] == 27:
                    inc = _count_incidents(pts, lns)
                    if inc > best_inc:
                        best_inc = inc
                        best_points = pts.copy()
                        best_lines = lns.copy()
            except Exception:
                continue

    # Strategy 5: Try 9x6 grid
    for num_v in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        for num_h in [0, 1, 2, 3, 4, 5, 6]:
            num_diag = 27 - num_v - num_h
            if num_diag < 0:
                continue
            try:
                pts, lns = _build_grid_and_diagonals(9, 6, num_v, num_h, num_diag, [1, -1])
                if lns.shape[0] == 27:
                    inc = _count_incidents(pts, lns)
                    if inc > best_inc:
                        best_inc = inc
                        best_points = pts.copy()
                        best_lines = lns.copy()
            except Exception:
                continue

    # Strategy 6: Try to add extra diagonal slopes to 6x9
    for slope_set in [[1, -1, 2, -2], [1, -1, 2, -2, 3, -3], [1, -1, 1/2, -1/2]]:
        for num_v in [0, 3, 6]:
            for num_h in [0, 3, 9]:
                num_diag = 27 - num_v - num_h
                if num_diag < 0:
                    continue
                try:
                    pts, lns = _build_grid_and_diagonals(6, 9, num_v, num_h, num_diag, slope_set)
                    if lns.shape[0] == 27:
                        inc = _count_incidents(pts, lns)
                        if inc > best_inc:
                            best_inc = inc
                            best_points = pts.copy()
                            best_lines = lns.copy()
                except Exception:
                    continue

    # Fallback: return Szemeredi-Trotter
    if best_points is None:
        k = 3
        xs = np.arange(1, k + 1, dtype=np.int64)
        ys = np.arange(1, 2 * k * k + 1, dtype=np.int64)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points_st = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        rows = []
        for a in range(1, k + 1):
            for b in range(1, k * k + 1):
                rows.append((a, -1, b))
        lines_st = np.array(rows, dtype=np.int64)
        return points_st, lines_st

    return best_points, best_lines


# EVOLVE-BLOCK-END
