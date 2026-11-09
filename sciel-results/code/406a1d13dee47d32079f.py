# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from itertools import combinations
from collections import Counter


def _canonical_line(a, b, c):
    """Return canonical form of line (a,b,c): divide by gcd, make first nonzero positive."""
    if a == 0 and b == 0 and c == 0:
        return (0, 0, 0)
    g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    a //= g
    b //= g
    c //= g
    # Make first nonzero entry positive
    if a < 0:
        a, b, c = -a, -b, -c
    elif a == 0 and b < 0:
        a, b, c = -a, -b, -c
    return (a, b, c)


def _count_incidences(points, lines):
    """Count point-line incidences exactly using vectorized numpy."""
    # points: (n,2) int array, lines: (m,3) int array
    # Compute A*x + B*y + C for all pairs at once
    # points: (n, 2) -> x: (n,1), y: (n,1)
    # lines: (m, 3) -> A: (1,m), B: (1,m), C: (1,m)
    x = points[:, 0].astype(np.int64).reshape(-1, 1)
    y = points[:, 1].astype(np.int64).reshape(-1, 1)
    A = lines[:, 0].astype(np.int64).reshape(1, -1)
    B = lines[:, 1].astype(np.int64).reshape(1, -1)
    C = lines[:, 2].astype(np.int64).reshape(1, -1)
    vals = A * x + B * y + C
    return int(np.sum(vals == 0))


def _top_lines_for_points(points, max_lines=27):
    """Find the top max_lines lines by incidence count with the given points."""
    n = len(points)
    if n < 2:
        return []
    
    # Count all lines through pairs of points
    line_counts = Counter()
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Line: dy*x - dx*y + (dx*y1 - dy*x1) = 0
            a = dy
            b = -dx
            c = dx * y1 - dy * x1
            key = _canonical_line(a, b, c)
            line_counts[key] += 1
    
    # Also consider single-point lines? No, we need at least 2 points for a unique line.
    # But a line through 1 point could be optimal if no pair gives more... 
    # Actually, for 54 points, the top lines will have many points.
    
    # Get top lines
    top = line_counts.most_common(max_lines)
    if len(top) < max_lines:
        # Pad with lines through single points (arbitrary)
        # This shouldn't happen with 54 points
        pass
    
    return [key for key, cnt in top]


def _evaluate(points, max_lines=27):
    """Evaluate a point configuration: find best max_lines lines and count incidences."""
    lines = _top_lines_for_points(points, max_lines)
    if not lines:
        return 0, []
    lines_arr = np.array(lines, dtype=np.int64)
    count = _count_incidences(points, lines_arr)
    return count, lines_arr


def _local_search(points, max_lines=27, max_iters=500, time_limit=580):
    """Local search: move one point at a time to improve incidence count."""
    start_time = time.time()
    best_points = points.copy()
    best_score, best_lines = _evaluate(best_points, max_lines)
    
    n = len(best_points)
    
    # Precompute all pair-lines for fast evaluation
    def _get_top_lines(pts):
        """Fast: compute top lines for a point set."""
        line_counts = {}
        for i in range(n):
            x1, y1 = pts[i]
            for j in range(i + 1, n):
                x2, y2 = pts[j]
                dx = x2 - x1
                dy = y2 - y1
                a = dy
                b = -dx
                c = dx * y1 - dy * x1
                key = _canonical_line(a, b, c)
                line_counts[key] = line_counts.get(key, 0) + 1
        # Get top max_lines
        top = sorted(line_counts.items(), key=lambda x: -x[1])[:max_lines]
        if len(top) < max_lines:
            # Pad
            seen_keys = set(k for k, _ in top)
            # Add lines through single points with arbitrary orientation
            for i in range(n):
                if len(top) >= max_lines:
                    break
                x1, y1 = pts[i]
                for a, b, c in [(1, 0, -x1), (0, 1, -y1), (1, 1, -(x1+y1))]:
                    key = _canonical_line(a, b, c)
                    if key not in seen_keys:
                        top.append((key, 0))
                        seen_keys.add(key)
                        if len(top) >= max_lines:
                            break
        return np.array([k for k, _ in top], dtype=np.int64)
    
    # Use a set for duplicate checking
    def _make_point_set(pts, exclude_idx):
        s = set()
        for j in range(n):
            if j != exclude_idx:
                s.add((pts[j][0], pts[j][1]))
        return s
    
    # Phase 1: Small moves
    move_ranges = [
        list(range(-2, 3)),   # small
        list(range(-5, 6)),   # medium
        list(range(-10, 11)), # large
    ]
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            
            orig_x, orig_y = best_points[i]
            existing = _make_point_set(best_points, i)
            
            # Try moves from small to large
            for move_range in move_ranges:
                for dx in move_range:
                    if time.time() - start_time > time_limit:
                        break
                    for dy in move_range:
                        if dx == 0 and dy == 0:
                            continue
                        new_x = orig_x + dx
                        new_y = orig_y + dy
                        if abs(new_x) > 60 or abs(new_y) > 60:
                            continue
                        if (new_x, new_y) in existing:
                            continue
                        
                        trial = best_points.copy()
                        trial[i] = [new_x, new_y]
                        score, _ = _evaluate(trial, max_lines)
                        
                        if score > best_score:
                            best_points = trial
                            best_score = score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    # Final polish: re-optimize line selection
    final_lines = _get_top_lines(best_points)
    final_score = _count_incidences(best_points, final_lines)
    
    return best_points, final_score, final_lines


def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization using local search.
    Starts from a grid configuration and iteratively moves points to maximize
    the number of incidences with the top-27 lines.
    """
    deadline = 620
    start_time = time.time()
    
    # Initial configuration: 3x18 grid
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Run local search with more iterations
    best_points, best_score, best_lines = _local_search(points, max_lines=27, max_iters=300, time_limit=deadline - 60)
    
    # If time remains, try a different initial configuration
    remaining = deadline - (time.time() - start_time) - 15
    if remaining > 40:
        # Try a different grid: 6x9
        xs2 = np.arange(1, 7)
        ys2 = np.arange(1, 10)
        px2, py2 = np.meshgrid(xs2, ys2, indexing="ij")
        points2 = np.stack([px2.ravel(), py2.ravel()], axis=1).astype(np.int64)
        
        p2, s2, l2 = _local_search(points2, max_lines=27, max_iters=150, time_limit=remaining)
        if s2 > best_score:
            best_points, best_score, best_lines = p2, s2, l2
    
    # Try yet another configuration if time allows
    remaining = deadline - (time.time() - start_time) - 15
    if remaining > 40:
        # 9x6 grid
        xs3 = np.arange(1, 10)
        ys3 = np.arange(1, 7)
        px3, py3 = np.meshgrid(xs3, ys3, indexing="ij")
        points3 = np.stack([px3.ravel(), py3.ravel()], axis=1).astype(np.int64)
        
        p3, s3, l3 = _local_search(points3, max_lines=27, max_iters=100, time_limit=remaining)
        if s3 > best_score:
            best_points, best_score, best_lines = p3, s3, l3
    
    # Ensure we have exactly 27 lines
    if len(best_lines) < 27:
        # Pad with dummy lines (shouldn't happen)
        dummy = np.zeros((27 - len(best_lines), 3), dtype=np.int64)
        best_lines = np.vstack([best_lines, dummy])
    else:
        best_lines = best_lines[:27]
    
    # Validate: ensure no duplicate points
    seen = set()
    valid_points = []
    for p in best_points:
        key = (int(p[0]), int(p[1]))
        if key not in seen:
            seen.add(key)
            valid_points.append(key)
    
    if len(valid_points) < 54:
        # Fallback to original grid
        k = 3
        xs = np.arange(1, k + 1)
        ys = np.arange(1, 2 * k * k + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        # Use the lines from the original grid
        rows = []
        for a in range(1, k + 1):
            for b in range(1, k * k + 1):
                rows.append((a, -1, b))
        lines = np.array(rows, dtype=np.int64)
        return points, lines
    
    points_out = np.array(valid_points[:54], dtype=np.int64)
    lines_out = best_lines.astype(np.int64)
    
    # Final validation: check for duplicate lines
    seen_lines = set()
    final_lines = []
    for line in lines_out:
        key = _canonical_line(int(line[0]), int(line[1]), int(line[2]))
        if key not in seen_lines and key != (0, 0, 0):
            seen_lines.add(key)
            final_lines.append(key)
    
    # Pad if needed
    while len(final_lines) < 27:
        # Add a line that doesn't pass through any points
        found = False
        for a in range(-20, 21):
            for b in range(-20, 21):
                for c in range(-200, 201, 7):
                    key = _canonical_line(a, b, c)
                    if key not in seen_lines and key != (0, 0, 0):
                        seen_lines.add(key)
                        final_lines.append(key)
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            break
    
    lines_out = np.array(final_lines[:27], dtype=np.int64)
    
    return points_out, lines_out


# EVOLVE-BLOCK-END
