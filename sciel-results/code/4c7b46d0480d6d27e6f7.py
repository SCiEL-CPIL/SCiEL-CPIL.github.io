"""Rewritten program for Heilbronn triangles task."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations

N = 11
# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])


def _tri_area(p1, p2, p3):
    """Area of triangle formed by three points."""
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def _min_tri_area(points):
    """Minimum area among all triangles formed by the points."""
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _tri_area(points[i], points[j], points[k])
                if a < min_a:
                    min_a = a
    return min_a


def _inside_triangle(p):
    """Check if point is inside or on the equilateral triangle."""
    # Barycentric coordinates
    v0 = V2 - V0
    v1 = V1 - V0
    v2 = p - V0
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d02 = np.dot(v0, v2)
    d11 = np.dot(v1, v1)
    d12 = np.dot(v1, v2)
    den = d00 * d11 - d01 * d01
    u = (d11 * d02 - d01 * d12) / den
    v = (d00 * d12 - d01 * d02) / den
    return u >= -1e-9 and v >= -1e-9 and u + v <= 1 + 1e-9


def _clip_to_triangle(p):
    """Clip a point to the triangle domain."""
    # If already inside, return as is
    if _inside_triangle(p):
        return p
    # Otherwise, find closest point on triangle boundary
    # Simple approach: project onto each edge and pick the one that minimizes distance
    edges = [(V0, V1), (V1, V2), (V2, V0)]
    best_p = None
    best_d = float('inf')
    for a, b in edges:
        ab = b - a
        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        d = np.linalg.norm(p - proj)
        if d < best_d:
            best_d = d
            best_p = proj
    return best_p


def _objective(x):
    """Objective for minimization: negative of min triangle area (we maximize min area)."""
    points = np.array(x).reshape(N, 2)
    # Penalty for points outside triangle
    penalty = 0.0
    for p in points:
        if not _inside_triangle(p):
            # Distance outside
            penalty += np.linalg.norm(p - _clip_to_triangle(p)) ** 2
    if penalty > 1e-6:
        return 1.0 + penalty  # Penalize heavily
    min_a = _min_tri_area(points)
    return -min_a


def _local_polish(points, max_iter=500):
    """Local optimization using Nelder-Mead or similar."""
    points = np.array(points, dtype=float)
    x0 = points.flatten()
    
    # Use Nelder-Mead for local refinement
    res = minimize(_objective, x0, method='Nelder-Mead',
                   options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12})
    best_points = res.x.reshape(N, 2)
    
    # Clip to triangle
    for i in range(N):
        best_points[i] = _clip_to_triangle(best_points[i])
    
    return best_points


def _largest_empty_triangle_heuristic(points, n_iter=100):
    """
    Iteratively improve points by:
    1. Finding the minimum-area triangle
    2. Computing centroid of convex hull of other points
    3. Moving the least-contributing point toward that centroid
    """
    points = np.array(points, dtype=float)
    
    for it in range(n_iter):
        # Find min area triangle
        min_a = float('inf')
        min_idx = None
        for i in range(N):
            for j in range(i + 1, N):
                for k in range(j + 1, N):
                    a = _tri_area(points[i], points[j], points[k])
                    if a < min_a:
                        min_a = a
                        min_idx = (i, j, k)
        
        i, j, k = min_idx
        
        # Compute centroid of other points
        other_indices = [m for m in range(N) if m not in (i, j, k)]
        centroid = np.mean(points[other_indices], axis=0)
        
        # Determine which of i, j, k to move
        # Move the point that, when moved toward centroid, most improves min area
        best_gain = -float('inf')
        best_move_idx = None
        best_new_point = None
        
        for idx in (i, j, k):
            # Try moving this point toward centroid
            for t in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
                new_point = points[idx] + t * (centroid - points[idx])
                new_point = _clip_to_triangle(new_point)
                
                # Test improvement
                test_points = points.copy()
                test_points[idx] = new_point
                test_min_a = _min_tri_area(test_points)
                
                gain = test_min_a - min_a
                if gain > best_gain:
                    best_gain = gain
                    best_move_idx = idx
                    best_new_point = new_point
        
        if best_move_idx is not None and best_gain > 1e-12:
            points[best_move_idx] = best_new_point
    
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    best_points = None
    best_score = -float('inf')
    
    # Strategy 1: Differential evolution for global search
    # Bounds: x in [0, 1], y in [0, sqrt(3)/2]
    bounds = [(0.0, 1.0), (0.0, np.sqrt(3) / 2)] * N
    
    # Use a smaller population for speed
    try:
        result = differential_evolution(
            _objective, bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-8,
            polish=True,
            init='sobol'
        )
        if time.time() < deadline:
            candidate = result.x.reshape(N, 2)
            for i in range(N):
                candidate[i] = _clip_to_triangle(candidate[i])
            score = _min_tri_area(candidate)
            if score > best_score:
                best_score = score
                best_points = candidate
    except Exception:
        pass
    
    # Strategy 2: Local polish from best found
    if best_points is not None and time.time() < deadline:
        polished = _local_polish(best_points, max_iter=300)
        score = _min_tri_area(polished)
        if score > best_score:
            best_score = score
            best_points = polished
    
    # Strategy 3: Largest empty triangle heuristic from multiple random starts
    if time.time() < deadline:
        np.random.seed(123)
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            # Random initial points
            pts = []
            for _ in range(N):
                u, v = np.random.random(), np.random.random()
                while u + v > 1:
                    u, v = np.random.random(), np.random.random()
                p = V0 + u * (V1 - V0) + v * (V2 - V0)
                pts.append(p)
            pts = np.array(pts)
            
            # Apply heuristic
            improved = _largest_empty_triangle_heuristic(pts, n_iter=50)
            
            # Local polish
            if time.time() < deadline - 5:
                improved = _local_polish(improved, max_iter=200)
            
            score = _min_tri_area(improved)
            if score > best_score:
                best_score = score
                best_points = improved
    
    # Strategy 4: Symmetric quasi-lattice initialization + optimization
    if time.time() < deadline - 10:
        # Create a symmetric arrangement
        # 3 corners, 3 midpoints, 1 center, 2 additional
        sym_points = [
            V0 + 0.05 * (V1 - V0),  # near corner 0
            V1 - 0.05 * (V1 - V0),  # near corner 1
            V2 + 0.05 * (V0 - V2),  # near corner 2
            0.5 * (V0 + V1),  # midpoint 0-1
            0.5 * (V1 + V2),  # midpoint 1-2
            0.5 * (V2 + V0),  # midpoint 2-0
            (V0 + V1 + V2) / 3,  # center
            0.33 * V0 + 0.33 * V1 + 0.34 * V2,
            0.34 * V0 + 0.33 * V1 + 0.33 * V2,
            0.33 * V0 + 0.34 * V1 + 0.33 * V2,
            (V0 + V1 + V2) / 3 + np.array([0.05, 0.03])
        ]
        sym_points = np.array(sym_points)
        for i in range(N):
            sym_points[i] = _clip_to_triangle(sym_points[i])
        
        if time.time() < deadline - 5:
            polished = _local_polish(sym_points, max_iter=300)
            score = _min_tri_area(polished)
            if score > best_score:
                best_score = score
                best_points = polished
    
    if best_points is None:
        # Fallback: return simple valid configuration
        best_points = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3) / 2],
            [0.25, np.sqrt(3) / 4],
            [0.75, np.sqrt(3) / 4],
            [0.5, 3 * np.sqrt(3) / 4],
            [0.5, np.sqrt(3) / 2 * 0.5],
            [0.3, 0.1],
            [0.7, 0.1],
            [0.5, 0.3],
            [0.4, 0.2]
        ])
        for i in range(N):
            best_points[i] = _clip_to_triangle(best_points[i])
    
    return best_points
