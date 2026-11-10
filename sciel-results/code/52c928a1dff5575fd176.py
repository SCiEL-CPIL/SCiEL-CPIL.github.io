# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd

_start_time = time.time()
_DEADLINE = 600


def _line_key(A, B, C):
    """Canonical form for a line."""
    if A == 0:
        if B == 0:
            return (0, 0, 0)
        if B < 0:
            A, B, C = -A, -B, -C
    else:
        if A < 0:
            A, B, C = -A, -B, -C
    g = gcd(gcd(abs(A), abs(B)), abs(C))
    if g > 0:
        A, B, C = A // g, B // g, C // g
    return (A, B, C)


def _incidence_count_fast(points, lines):
    """Vectorized incidence count."""
    vals = points[:, 0:1].T @ lines[:, 0:1] + points[:, 1:2].T @ lines[:, 1:2] + lines[:, 2:3].T
    return int(np.sum(vals == 0))


def _valid_lines(lines):
    seen = set()
    for A, B, C in lines:
        if A == 0 and B == 0:
            return False
        key = _line_key(A, B, C)
        if key in seen:
            return False
        seen.add(key)
    return True


def _valid_points(points):
    seen = set()
    for x, y in points:
        key = (int(x), int(y))
        if key in seen:
            return False
        seen.add(key)
    return True


def _find_best_lines_for_points(points, m=27, time_limit=20):
    """
    Given fixed points, find m lines maximizing incidences.
    Uses greedy + local search with efficient scoring.
    """
    pts = np.array(points, dtype=np.int64)
    n = len(pts)
    
    # Generate all candidate lines through pairs of points
    candidate_lines = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = pts[i]
            x2, y2 = pts[j]
            dx = x2 - x1
            dy = y2 - y1
            A = dy
            B = -dx
            C = dx * y1 - dy * x1
            key = _line_key(A, B, C)
            if key not in candidate_lines:
                candidate_lines[key] = (A, B, C)
    
    cand_keys = list(candidate_lines.keys())
    cand_lines = [candidate_lines[k] for k in cand_keys]
    num_cands = len(cand_lines)
    
    if num_cands == 0:
        # Fallback
        lines = [(1, 0, 0)] * m
        return 0, pts, np.array(lines, dtype=np.int64)
    
    # Precompute incidence vectors
    inc_vecs = np.zeros((num_cands, n), dtype=np.int8)
    for i, (A, B, C) in enumerate(cand_lines):
        inc_vecs[i] = (A * pts[:, 0] + B * pts[:, 1] + C == 0).astype(np.int8)
    
    line_scores = inc_vecs.sum(axis=1)
    
    # Greedy: pick top m
    top_m = np.argsort(-line_scores)[:m]
    current_inc = int(line_scores[top_m].sum())
    
    # Local search: single swap
    start = time.time()
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        for idx in range(m):
            if (time.time() - start) > time_limit:
                break
            cur_line = top_m[idx]
            cur_score = line_scores[cur_line]
            # Try replacing with any other line
            for j in range(num_cands):
                if j in top_m:
                    continue
                new_score = line_scores[j]
                if new_score > cur_score:
                    # Check if this line is already in the set
                    # (it shouldn't be since we skip j in top_m)
                    new_inc = current_inc - cur_score + new_score
                    if new_inc > current_inc:
                        current_inc = new_inc
                        top_m[idx] = j
                        improved = True
                        break
    
    selected = [cand_lines[i] for i in top_m]
    
    # Validate
    if not _valid_lines(selected):
        # Try to fix by replacing duplicates
        seen = set()
        fixed = []
        for line in selected:
            key = _line_key(*line)
            if key not in seen:
                seen.add(key)
                fixed.append(line)
        # Pad if needed
        for line in cand_lines:
            if len(fixed) >= m:
                break
            key = _line_key(*line)
            if key not in seen:
                seen.add(key)
                fixed.append(line)
        selected = fixed[:m]
        if len(selected) < m:
            selected = [(1, 0, 0)] * m
    
    return current_inc, pts, np.array(selected, dtype=np.int64)


def _grid_points(nx, ny, x0=1, y0=1):
    pts = []
    for x in range(x0, x0 + nx):
        for y in range(y0, y0 + ny):
            pts.append((x, y))
    return pts


def _search_all_configs():
    """Search over various point configurations and find best line set."""
    best_inc = 0
    best_points = None
    best_lines = None
    
    # Pure grid shapes that multiply to 54
    grid_shapes = []
    for nx in range(1, 55):
        if 54 % nx == 0:
            ny = 54 // nx
            grid_shapes.append((nx, ny))
    
    for nx, ny in grid_shapes:
        if time.time() - _start_time > 550:
            break
        points = _grid_points(nx, ny)
        pts = np.array(points, dtype=np.int64)
        
        if not _valid_points(pts):
            continue
        
        inc, pts_out, lines = _find_best_lines_for_points(pts, 27, time_limit=15)
        if inc > best_inc:
            best_inc = inc
            best_points = pts_out
            best_lines = lines
    
    # Hybrid: vertical line + grid
    for k in range(2, 30):
        if time.time() - _start_time > 580:
            break
        remaining = 54 - k
        if remaining < 4:
            continue
        for nx in range(2, min(remaining + 1, 20)):
            if remaining % nx != 0:
                continue
            ny = remaining // nx
            if ny > 20:
                continue
            
            # Points: k on x=0, rest on grid at x=10..
            points = []
            for y in range(1, k + 1):
                points.append((0, y))
            for x in range(10, 10 + nx):
                for y in range(1, ny + 1):
                    points.append((x, y))
            
            pts = np.array(points, dtype=np.int64)
            if not _valid_points(pts):
                continue
            
            inc, pts_out, lines = _find_best_lines_for_points(pts, 27, time_limit=10)
            if inc > best_inc:
                best_inc = inc
                best_points = pts_out
                best_lines = lines
    
    return best_inc, best_points, best_lines


def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization.
    """
    global _start_time
    
    best_inc, best_points, best_lines = _search_all_configs()
    
    if best_points is None:
        # Fallback
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
    
    best_points = np.array(best_points, dtype=np.int64)
    best_lines = np.array(best_lines, dtype=np.int64)
    
    assert best_points.shape == (54, 2)
    assert best_lines.shape == (27, 3)
    
    return best_points, best_lines


# EVOLVE-BLOCK-END
