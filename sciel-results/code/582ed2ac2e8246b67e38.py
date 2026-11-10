import numpy as np
import time
import math
import random
from itertools import combinations

def _normalize_line(A, B, C):
    A, B, C = int(A), int(B), int(C)
    g = math.gcd(abs(A), abs(B))
    g = math.gcd(g, abs(C))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidence(points, lines):
    """Count incidences between points (N,2) and lines (M,3)."""
    if len(points) == 0 or len(lines) == 0:
        return 0
    px = points[:, 0]
    py = points[:, 1]
    A = lines[:, 0]
    B = lines[:, 1]
    C = lines[:, 2]
    vals = A[:, None] * px[None, :] + B[:, None] * py[None, :] + C[:, None]
    return int(np.sum(vals == 0))

def _best_lines_for_points(points, k=27):
    """
    Find the k lines with the highest incidence count for the given points.
    """
    n = len(points)
    if n < 2:
        return np.zeros((k, 3), dtype=np.int64)
    
    xs = points[:, 0]
    ys = points[:, 1]
    
    seen = set()
    candidate_lines = []
    
    # Generate candidate lines from all pairs
    for i in range(n):
        x1, y1 = int(xs[i]), int(ys[i])
        for j in range(i + 1, n):
            x2, y2 = int(xs[j]), int(ys[j])
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            key = _normalize_line(A, B, C)
            if key not in seen:
                seen.add(key)
                candidate_lines.append(key)
    
    # Add some generic lines just in case (though pairs should cover all grid lines)
    # But for sparse sets, generic lines might be better if they hit many points?
    # Unlikely for random, but for grid, pairs cover it.
    
    if not candidate_lines:
         # Fallback
         return np.array([(0, 1, -i) for i in range(k)], dtype=np.int64)

    # Vectorized counting
    lines_arr = np.array(candidate_lines, dtype=np.int64)
    A = lines_arr[:, 0]
    B = lines_arr[:, 1]
    C = lines_arr[:, 2]
    
    # Efficient counting
    # vals shape: (M, N)
    vals = A[:, None] * xs[None, :] + B[:, None] * ys[None, :] + C[:, None]
    counts = np.sum(vals == 0, axis=1)
    
    # Get top k
    if len(counts) >= k:
        top_indices = np.argsort(counts)[::-1][:k]
    else:
        top_indices = np.argsort(counts)[::-1]
        
    top_lines = lines_arr[top_indices]
    
    # Pad if necessary
    if len(top_lines) < k:
        pad = np.zeros((k - len(top_lines), 3), dtype=np.int64)
        pad[:, 1] = 1
        pad[:, 2] = -10000 - np.arange(k - len(top_lines))
        top_lines = np.vstack([top_lines, pad])
        
    return top_lines.astype(np.int64)

def _generate_grid_points(k, m, dx=1, dy=1, ox=0, oy=0):
    """Generate a k x m grid of points."""
    xs = np.arange(k) * dx + ox
    ys = np.arange(m) * dy + oy
    px, py = np.meshgrid(xs, ys, indexing='ij')
    return np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

def _evaluate_configuration(points):
    """Evaluate a point configuration and return (points, lines, score)."""
    lines = _best_lines_for_points(points, 27)
    score = _count_incidence(points, lines)
    return points, lines, score

def szemeredi_trotter_incidences():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_points = None
    best_lines = None
    best_score = -1
    
    # Candidate 1: 6x9 Grid
    pts = _generate_grid_points(6, 9)
    if len(pts) == 54:
        p, l, s = _evaluate_configuration(pts)
        if s > best_score:
            best_score = s
            best_points = p.copy()
            best_lines = l.copy()
            
    # Candidate 2: 9x6 Grid
    pts = _generate_grid_points(9, 6)
    if len(pts) == 54:
        p, l, s = _evaluate_configuration(pts)
        if s > best_score:
            best_score = s
            best_points = p.copy()
            best_lines = l.copy()
            
    # Candidate 3: 7x8 Grid with 2 points removed
    # Try removing corners and some random points
    full_grid = _generate_grid_points(7, 8)
    # Remove (0,0) and (6,7)
    mask = np.ones(len(full_grid), dtype=bool)
    # Indices: row-major. (0,0) is 0. (6,7) is 6*8+7 = 55.
    mask[0] = False
    mask[55] = False
    pts = full_grid[mask]
    if len(pts) == 54:
        p, l, s = _evaluate_configuration(pts)
        if s > best_score:
            best_score = s
            best_points = p.copy()
            best_lines = l.copy()
            
    # Try removing (0,7) and (6,0)
    mask = np.ones(len(full_grid), dtype=bool)
    mask[7] = False
    mask[48] = False
    pts = full_grid[mask]
    if len(pts) == 54:
        p, l, s = _evaluate_configuration(pts)
        if s > best_score:
            best_score = s
            best_points = p.copy()
            best_lines = l.copy()

    # Candidate 4: 5x11 Grid with 1 point removed
    full_grid = _generate_grid_points(5, 11)
    mask = np.ones(len(full_grid), dtype=bool)
    mask[0] = False
    pts = full_grid[mask]
    if len(pts) == 54:
        p, l, s = _evaluate_configuration(pts)
        if s > best_score:
            best_score = s
            best_points = p.copy()
            best_lines = l.copy()
            
    # Candidate 5: 4x14 Grid with 2 points removed
    full_grid = _generate_grid_points(4, 14)
    mask = np.ones(len(full_grid), dtype=bool)
    mask[0] = False
    mask[55] = False # 4*14=56, last index 55
    pts = full_grid[mask]
    if len(pts) == 54:
        p, l, s = _evaluate_configuration(pts)
        if s > best_score:
            best_score = s
            best_points = p.copy()
            best_lines = l.copy()

    # Candidate 6: 3x18 Grid
    pts = _generate_grid_points(3, 18)
    if len(pts) == 54:
        p, l, s = _evaluate_configuration(pts)
        if s > best_score:
            best_score = s
            best_points = p.copy()
            best_lines = l.copy()

    # Fallback
    if best_points is None:
        best_points = _generate_grid_points(6, 9)
        best_lines = _best_lines_for_points(best_points, 27)
        best_score = _count_incidence(best_points, best_lines)
        
    # Final Validation
    final_points = np.array(best_points, dtype=np.int64)
    final_lines = np.array(best_lines, dtype=np.int64)
    
    if final_points.shape != (54, 2):
        final_points = _generate_grid_points(6, 9)
        final_lines = _best_lines_for_points(final_points, 27)
        
    if final_lines.shape != (27, 3):
        final_lines = final_lines[:27]
        if len(final_lines) < 27:
            pad = np.zeros((27 - len(final_lines), 3), dtype=np.int64)
            pad[:, 1] = 1
            pad[:, 2] = -10000
            final_lines = np.vstack([final_lines, pad])
            
    # Ensure uniqueness
    unique_pts = np.unique(final_points, axis=0)
    if len(unique_pts) < 54:
        # This shouldn't happen with our generators, but just in case
        offset = 100
        while len(unique_pts) < 54:
            pt = np.array([offset, offset], dtype=np.int64)
            if np.sum(np.all(unique_pts == pt, axis=1)) == 0:
                unique_pts = np.vstack([unique_pts, pt[None, :]])
            offset += 1
        final_points = unique_pts[:54]
        final_lines = _best_lines_for_points(final_points, 27)

    return final_points.astype(np.int64), final_lines.astype(np.int64)
