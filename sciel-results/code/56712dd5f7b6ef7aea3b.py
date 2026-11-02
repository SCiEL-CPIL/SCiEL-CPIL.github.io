"""Heilbronn triangles: maximize minimum triangle area for 11 points in a unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize
import itertools
import sys

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2.0
TRI_AREA = S3 / 4.0  # area of equilateral triangle with side 1

# The triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])


def _point_in_triangle(p):
    """Check if point p is inside the equilateral triangle."""
    # Using barycentric coordinates
    v0 = V0
    v1 = V1
    v2 = V2
    p = np.asarray(p)
    d0 = np.cross(v1 - v0, p - v0)
    d1 = np.cross(v2 - v1, p - v1)
    d2 = np.cross(v0 - v2, p - v2)
    has_neg = (d0 < 0) or (d1 < 0) or (d2 < 0)
    has_pos = (d0 > 0) or (d1 > 0) or (d2 > 0)
    return not (has_neg and has_pos)


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points.
    Returns the area (not normalized).
    """
    n = len(points)
    min_a = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            for k in range(j + 1, n):
                pk = points[k]
                # Area = 0.5 * |cross(pj-pi, pk-pi)|
                cross = abs((pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0]))
                area = 0.5 * cross
                if area < min_a:
                    min_a = area
    return min_a


def _min_triangle_area_normalized(points):
    """Minimum triangle area divided by the triangle area (sqrt(3)/4)."""
    return _min_triangle_area(points) / TRI_AREA


def _negative_min_area(points):
    """Objective for minimization: negative of minimum triangle area."""
    return -_min_triangle_area(points)


def _project_to_triangle(p):
    """Project a point to the closest point inside the triangle (approximate)."""
    # Simple approach: if inside, return as-is. Otherwise, clamp using barycentric coords.
    v0 = V0
    v1 = V1
    v2 = V2
    p = np.asarray(p, dtype=float)
    
    # Barycentric coordinates
    v01 = v1 - v0
    v02 = v2 - v0
    v0p = p - v0
    
    d00 = np.dot(v01, v01)
    d01 = np.dot(v01, v02)
    d02 = np.dot(v01, v0p)
    d11 = np.dot(v02, v02)
    d12 = np.dot(v02, v0p)
    
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return p
    
    v = (d11 * d02 - d01 * d12) / denom
    w = (d00 * d12 - d01 * d02) / denom
    u = 1.0 - v - w
    
    # Clamp to barycentric simplex
    if u < 0:
        u = 0.0
        v = v / (v + w)
        w = w / (v + w)
    elif v < 0:
        v = 0.0
        u = u / (u + w)
        w = w / (u + w)
    elif w < 0:
        w = 0.0
        u = u / (u + v)
        v = v / (u + v)
    
    # Normalize if needed
    s = u + v + w
    if s > 1e-12:
        u /= s
        v /= s
        w /= s
    
    return u * v0 + v * v1 + w * v2


def _random_point_in_triangle(rng):
    """Generate a random point inside the triangle using rejection sampling."""
    while True:
        u = rng.uniform(0, 1)
        v = rng.uniform(0, 1)
        if u + v <= 1:
            p = u * V1 + v * V2
            return p


def construct():
    import random
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    best_points = None
    best_score = -float('inf')
    
    # Initial good configuration: use a grid-like pattern
    # Start with points spread across the triangle
    def make_initial_config(seed):
        rng = random.Random(seed)
        points = []
        # Place some points near vertices and edges
        # Then fill interior
        for _ in range(N):
            p = _random_point_in_triangle(rng)
            points.append(p)
        return np.array(points)
    
    # Try multiple random initializations with L-BFGS-B
    n_starts = 0
    seed = 42
    
    for attempt in range(200):
        if time.time() > deadline - 5.0:
            break
        
        seed += 1
        rng = random.Random(seed)
        
        # Create initial random configuration
        points = make_initial_config(seed)
        
        # Flatten to 2D array of 2N values
        x0 = points.flatten()
        
        # Define objective with penalty for points outside triangle
        def objective(x):
            pts = x.reshape(-1, 2)
            # Check if all points are inside triangle
            penalty = 0.0
            for p in pts:
                if not _point_in_triangle(p):
                    # Penalty: distance to triangle
                    proj = _project_to_triangle(p)
                    dist = np.linalg.norm(p - proj)
                    penalty += dist * 100.0
            if penalty > 1e-6:
                return -_min_triangle_area(pts) + penalty
            return -_min_triangle_area(pts)
        
        # Run L-BFGS-B
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if time.time() > deadline - 3.0:
                break
            
            # Check if result is valid
            pts = result.x.reshape(-1, 2)
            valid = True
            for p in pts:
                if not _point_in_triangle(p):
                    valid = False
                    break
            
            if valid:
                score = _min_triangle_area(pts) / TRI_AREA
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
                    n_starts += 1
        except Exception:
            continue
    
    # If no valid result from optimization, fall back to random
    if best_points is None:
        rng = random.Random(7)
        points = []
        while len(points) < N:
            p = _random_point_in_triangle(rng)
            points.append(p)
        best_points = np.array(points)
    
    # Final refinement: try a few more local optimizations from the best point
    if best_points is not None and time.time() < deadline - 10.0:
        for refine_attempt in range(10):
            if time.time() > deadline - 5.0:
                break
            
            # Perturb the best points slightly
            rng = random.Random(1000 + refine_attempt)
            perturbed = best_points.copy()
            for i in range(N):
                perturbed[i] += rng.gauss(0, 0.01)
            
            # Project to triangle
            for i in range(N):
                if not _point_in_triangle(perturbed[i]):
                    perturbed[i] = _project_to_triangle(perturbed[i])
            
            x0 = perturbed.flatten()
            
            def objective(x):
                pts = x.reshape(-1, 2)
                penalty = 0.0
                for p in pts:
                    if not _point_in_triangle(p):
                        proj = _project_to_triangle(p)
                        dist = np.linalg.norm(p - proj)
                        penalty += dist * 100.0
                if penalty > 1e-6:
                    return -_min_triangle_area(pts) + penalty
                return -_min_triangle_area(pts)
            
            try:
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                pts = result.x.reshape(-1, 2)
                valid = True
                for p in pts:
                    if not _point_in_triangle(p):
                        valid = False
                        break
                
                if valid:
                    score = _min_triangle_area(pts) / TRI_AREA
                    if score > best_score:
                        best_score = score
                        best_points = pts.copy()
            except Exception:
                continue
    
    return best_points


# Keep the original interface
N = 11
