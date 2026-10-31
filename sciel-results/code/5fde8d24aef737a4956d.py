"""Heilbronn triangles: place 11 points to maximize the minimum triangle area."""
import numpy as np
from itertools import combinations
import time
from scipy.optimize import differential_evolution, minimize

N = 11
# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Pre-compute triangle index combinations
tri_indices = list(combinations(range(N), 3))
tri_idx_arr = np.array(tri_indices)  # shape (165, 3)


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    p1 = points[tri_idx_arr[:, 0]]
    p2 = points[tri_idx_arr[:, 1]]
    p3 = points[tri_idx_arr[:, 2]]
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    return 0.5 * np.min(np.abs(cross))


def _points_inside_triangle(points):
    """Check if all points are inside or on the boundary of the triangle."""
    # For each point, check barycentric coordinates
    # A point P is inside triangle ABC if it can be written as
    # P = a*A + b*B + c*C with a+b+c=1 and a,b,c >= 0
    for p in points:
        # Compute barycentric coordinates
        # Using the formula with cross products
        v0 = V1 - V0
        v1 = V2 - V0
        v2 = p - V0
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d02 = np.dot(v0, v2)
        d11 = np.dot(v1, v1)
        d12 = np.dot(v1, v2)
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-12:
            return False
        a = (d11 * d02 - d01 * d12) / denom
        b = (d00 * d12 - d01 * d02) / denom
        c = 1.0 - a - b
        if a < -1e-9 or b < -1e-9 or c < -1e-9:
            return False
    return True


def _objective(x):
    """Objective for optimization: negative of min triangle area (to minimize)."""
    points = x.reshape(N, 2)
    # Penalty if points are outside the triangle
    if not _points_inside_triangle(points):
        return 1e6
    return -_min_triangle_area(points)


def _objective_fast(x):
    """Faster objective that clips points to triangle for DE (penalty approach)."""
    points = x.reshape(N, 2)
    # Soft penalty for being outside
    penalty = 0.0
    for p in points:
        # Check barycentric coordinates
        v0 = V1 - V0
        v1 = V2 - V0
        v2 = p - V0
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d02 = np.dot(v0, v2)
        d11 = np.dot(v1, v1)
        d12 = np.dot(v1, v2)
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-12:
            penalty += 1e6
            continue
        a = (d11 * d02 - d01 * d12) / denom
        b = (d00 * d12 - d01 * d02) / denom
        c = 1.0 - a - b
        if a < 0:
            penalty += 100 * a * a
        if b < 0:
            penalty += 100 * b * b
        if c < 0:
            penalty += 100 * c * c
    if penalty > 0:
        return 1e6 + penalty
    return -_min_triangle_area(points)


def _local_polish(points, max_iter=10000):
    """Local polish using Nelder-Mead on the 22-dimensional parameter space."""
    x0 = points.flatten()
    res = minimize(_objective, x0, method='Nelder-Mead',
                   options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12})
    polished = res.x.reshape(N, 2)
    if _points_inside_triangle(polished):
        return polished
    return points


def _coordinate_descent(points, max_iter=5000):
    """Simple coordinate descent for fine-tuning."""
    pts = points.copy()
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            best_pt = pts[i].copy()
            best_area = -_min_triangle_area(pts)
            # Try small perturbations
            for dx in [-0.005, -0.002, 0.002, 0.005]:
                for dy in [-0.005, -0.002, 0.002, 0.005]:
                    candidate = pts[i].copy()
                    candidate[0] += dx
                    candidate[1] += dy
                    pts[i] = candidate
                    if _points_inside_triangle(pts):
                        area = _min_triangle_area(pts)
                        if area > best_area:
                            best_area = area
                            best_pt = candidate.copy()
                    pts[i] = best_pt
            pts[i] = best_pt
        if not improved:
            break
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill

    # Initial population: grid-like placement
    # Place points at vertices and along edges, then interior
    initial = np.zeros((N, 2))
    # Vertices
    initial[0] = V0
    initial[1] = V1
    initial[2] = V2
    # Midpoints of edges
    initial[3] = (V0 + V1) / 2
    initial[4] = (V1 + V2) / 2
    initial[5] = (V2 + V0) / 2
    # Centroid
    initial[6] = (V0 + V1 + V2) / 3
    # Additional interior points
    # Use barycentric coordinates for interior
    interior_bary = [
        (0.2, 0.2, 0.6),
        (0.6, 0.2, 0.2),
        (0.2, 0.6, 0.2),
        (0.3, 0.3, 0.4),
        (0.4, 0.3, 0.3),
    ]
    for idx, (a, b, c) in enumerate(interior_bary):
        initial[7 + idx] = a * V0 + b * V1 + c * V2

    best_points = initial.copy()
    best_score = _min_triangle_area(best_points)

    # Differential evolution for global search
    # Bounds: x in [0, 1], y in [0, sqrt(3)/2]
    bounds = [(0.0, 1.0)] * N + [(0.0, np.sqrt(3) / 2)] * N

    def _de_objective(x):
        points = x.reshape(N, 2)
        if not _points_inside_triangle(points):
            return 1e6
        return -_min_triangle_area(points)

    elapsed = time.time() - start_time
    if elapsed < 60:
        # Run DE with moderate settings
        de_result = differential_evolution(
            _de_objective,
            bounds=bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=False,
            init='latinhypercube'
        )
        if de_result.fun < -best_score:
            candidate = de_result.x.reshape(N, 2)
            if _points_inside_triangle(candidate):
                score = _min_triangle_area(candidate)
                if score > best_score:
                    best_score = score
                    best_points = candidate.copy()

    elapsed = time.time() - start_time
    if elapsed < 90:
        # Local polish from DE result
        polished = _local_polish(best_points, max_iter=5000)
        if _points_inside_triangle(polished):
            score = _min_triangle_area(polished)
            if score > best_score:
                best_score = score
                best_points = polished.copy()

    # Additional local polish with coordinate descent
    elapsed = time.time() - start_time
    if elapsed < 110:
        refined = _coordinate_descent(best_points, max_iter=2000)
        if _points_inside_triangle(refined):
            score = _min_triangle_area(refined)
            if score > best_score:
                best_score = score
                best_points = refined.copy()

    # Final Nelder-Mead polish
    elapsed = time.time() - start_time
    if elapsed < 125:
        final = _local_polish(best_points, max_iter=3000)
        if _points_inside_triangle(final):
            score = _min_triangle_area(final)
            if score > best_score:
                best_score = score
                best_points = final.copy()

    return best_points
