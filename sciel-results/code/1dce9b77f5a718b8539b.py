# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd
from collections import defaultdict
from itertools import combinations


def _normalize_line(A, B, C):
    """Normalize a line (A, B, C) so that gcd(|A|,|B|,|C|)=1 and first nonzero is positive."""
    if A == 0 and B == 0 and C == 0:
        return None
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g > 0:
        A //= g
        B //= g
        C //= g
    if A < 0:
        A, B, C = -A, -B, -C
    elif A == 0 and B < 0:
        A, B, C = -A, -B, -C
    return (A, B, C)


def _count_incidences(points, lines):
    """Count total incidences between points and lines."""
    I = 0
    for p in points:
        x, y = p
        for l in lines:
            A, B, C = l
            if A * x + B * y + C == 0:
                I += 1
    return I


def _get_all_lines_through_points(points):
    """Get all unique lines through at least 2 points, with their point counts."""
    n = len(points)
    line_to_points = defaultdict(set)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            norm = _normalize_line(A, B, C)
            if norm is not None:
                line_to_points[norm].add(i)
    return {line: pts for line, pts in line_to_points.items() if len(pts) >= 2}


def _evaluate_configuration(points, lines):
    """Evaluate a configuration: return total incidences."""
    return _count_incidences(points, lines)


def _local_search(points, lines, deadline, max_iters=50):
    """Local search: try swapping one point or one line to improve incidences."""
    best_points = points.copy()
    best_lines = lines.copy()
    best_score = _evaluate_configuration(best_points, best_lines)
    
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        improved = False
        
        # Try swapping one point with a candidate position
        # Generate candidate positions in a neighborhood
        for i in range(len(best_points)):
            if time.time() > deadline - 5:
                break
            x, y = best_points[i]
            # Try moving this point to nearby positions
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if dx == 0 and dy == 0:
                        continue
                    new_x, new_y = x + dx, y + dy
                    # Check if this position is already occupied
                    if any(best_points[k][0] == new_x and best_points[k][1] == new_y for k in range(len(best_points)) if k != i):
                        continue
                    new_points = best_points.copy()
                    new_points[i] = (new_x, new_y)
                    score = _evaluate_configuration(new_points, best_lines)
                    if score > best_score:
                        best_points = new_points
                        best_score = score
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            # Try swapping one line with a better candidate
            all_lines = _get_all_lines_through_points(best_points)
            # Sort by count descending
            sorted_lines = sorted(all_lines.items(), key=lambda t: -len(t[1]))
            current_line_set = set(map(tuple, best_lines.tolist()))
            
            for candidate_line, pts in sorted_lines:
                if time.time() > deadline - 5:
                    break
                if candidate_line in current_line_set:
                    continue
                # Try replacing the worst line with this candidate
                # Find the line with minimum marginal contribution
                worst_idx = -1
                worst_gain = float('inf')
                for j in range(len(best_lines)):
                    A, B, C = best_lines[j]
                    # Count how many points are on this line
                    cnt = sum(1 for p in best_points if A * p[0] + B * p[1] + C == 0)
                    if cnt < worst_gain:
                        worst_gain = cnt
                        worst_idx = j
                if worst_idx >= 0 and len(pts) > worst_gain:
                    new_lines = best_lines.copy()
                    new_lines[worst_idx] = candidate_line
                    score = _evaluate_configuration(best_points, new_lines)
                    if score > best_score:
                        best_lines = new_lines
                        best_score = score
                        current_line_set = set(map(tuple, best_lines.tolist()))
                        improved = True
                        break
        
        if not improved:
            break
    
    return best_points, best_lines, best_score


def szemeredi_trotter_incidences():
    """
    Point-line incidence maximization with n=54 points and m=27 lines.
    
    Uses a 9x6 grid construction with local search polish to maximize incidences.
    """
    start = time.time()
    deadline = start + 620.0  # safety margin

    # ---- Points: 9x6 grid ----
    xs = np.arange(9, dtype=np.int64)   # 0..8
    ys = np.arange(6, dtype=np.int64)   # 0..5
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    assert points.shape == (54, 2)

    # ---- Enumerate all candidate lines ----
    point_set = set(map(tuple, points.tolist()))
    line_to_points = defaultdict(set)

    pts = points.tolist()
    n_pts = len(pts)
    for i in range(n_pts):
        x1, y1 = pts[i]
        for j in range(i + 1, n_pts):
            x2, y2 = pts[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            norm = _normalize_line(A, B, C)
            if norm is not None:
                line_to_points[norm].add(i)

    # Filter lines with at least 2 points
    candidate_lines = [(key, pts_set) for key, pts_set in line_to_points.items() if len(pts_set) >= 2]

    # Sort by number of points descending
    candidate_lines.sort(key=lambda t: -len(t[1]))

    # Take top 27 lines
    selected = candidate_lines[:27]
    lines = np.array([key for key, _ in selected], dtype=np.int64)
    assert lines.shape == (27, 3)

    # Verify no line has A=B=0
    assert not np.all(lines[:, :2] == 0, axis=1).any()

    # Compute initial incidences
    I = _count_incidences(points, lines)
    # print(f"Initial incidences: {I}")

    # ---- Local search polish ----
    if time.time() < deadline - 30:
        points_list = points.tolist()
        lines_list = lines.tolist()
        best_points, best_lines, best_score = _local_search(points_list, lines_list, deadline, max_iters=20)
        
        # Convert back to arrays
        points = np.array(best_points, dtype=np.int64)
        lines = np.array(best_lines, dtype=np.int64)
        
        # Verify validity
        assert points.shape == (54, 2)
        assert lines.shape == (27, 3)
        assert not np.all(lines[:, :2] == 0, axis=1).any()
        
        # Check for duplicate points
        point_tuples = set(map(tuple, points.tolist()))
        assert len(point_tuples) == 54, "Duplicate points found"
        
        # Check for duplicate lines
        line_tuples = set(map(tuple, lines.tolist()))
        assert len(line_tuples) == 27, "Duplicate lines found"
        
        I = _count_incidences(points, lines)
        # print(f"After local search: {I}")

    return points, lines


# EVOLVE-BLOCK-END
