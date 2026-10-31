"""Heilbronn triangles: place 11 points in equilateral triangle to maximize min triangle area."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import math

N = 11

# Triangle vertices
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, math.sqrt(3) / 2])
TRI_AREA = math.sqrt(3) / 4  # Area of unit base equilateral triangle

def barycentric_coords(p, a, b, c):
    """Compute barycentric coordinates of point p with respect to triangle abc."""
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w])

def barycentric_to_cartesian(u, v, w, a, b, c):
    """Convert barycentric coords to Cartesian."""
    return u * a + v * b + w * c

def in_triangle(p, a, b, c, tol=1e-10):
    """Check if point p is inside triangle abc."""
    bc = barycentric_coords(p, a, b, c)
    return (bc[0] >= -tol) and (bc[1] >= -tol) and (bc[2] >= -tol)

def triangle_area(p1, p2, p3):
    """Compute area of triangle formed by three points."""
    # 2D cross product (scalar)
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))

def min_triangle_area(points):
    """Compute the minimum area of any triangle formed by three of the points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area

def soft_min_triangle_area(points, beta=50.0):
    """
    Smooth surrogate for min triangle area using log-sum-exp.
    Returns a smooth approximation of the min area.
    """
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = triangle_area(points[i], points[j], points[k])
                areas.append(area)
    areas = np.array(areas)
    # Use log-sum-exp with negative sign to approximate min
    # min(x) ≈ -logsumexp(-x * beta) / beta
    neg_areas = -beta * areas
    max_neg = np.max(neg_areas)
    logsum = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
    return -logsum / beta

def project_to_triangle(p, a, b, c):
    """Project point p onto the triangle (a,b,c) if outside."""
    bc = barycentric_coords(p, a, b, c)
    if bc[0] < 0 or bc[1] < 0 or bc[2] < 0:
        # Project by clamping barycentric coords
        bc_clamped = np.clip(bc, 0, 1)
        s = np.sum(bc_clamped)
        if s > 0:
            bc_clamped /= s
        return barycentric_to_cartesian(bc_clamped[0], bc_clamped[1], bc_clamped[2], a, b, c)
    return p

def objective(x, beta=50.0):
    """
    Objective function for optimization.
    x is a flat array of 2N coordinates.
    Returns negative of soft-min area (we minimize negative = maximize area).
    """
    points = x.reshape(-1, 2)
    # Soft penalty for points outside triangle
    penalty = 0.0
    for p in points:
        bc = barycentric_coords(p, A, B, C)
        if bc[0] < 0 or bc[1] < 0 or bc[2] < 0:
            penalty += 100.0 * (max(0, -bc[0])**2 + max(0, -bc[1])**2 + max(0, -bc[2])**2)
    
    if penalty > 1e-6:
        return -soft_min_triangle_area(points, beta) + penalty
    
    return -soft_min_triangle_area(points, beta)

def gradient_objective(x, beta=50.0):
    """Compute gradient of the objective function using finite differences."""
    points = x.reshape(-1, 2)
    f0 = objective(x, beta)
    grad = np.zeros_like(x)
    h = 1e-7
    for i in range(len(x)):
        x_pert = x.copy()
        x_pert[i] += h
        grad[i] = (objective(x_pert, beta) - f0) / h
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s wall-clock limit, with safety margin
    
    best_points = None
    best_area = 0.0
    
    # Multiple restarts with different initializations
    rng = np.random.RandomState(42)
    
    for restart in range(50):
        if time.time() > deadline - 30.0:
            break
        
        # Initialize points
        if restart == 0:
            # Start with a reasonable heuristic: vertices + centroid + others
            points = []
            points.append(A.copy())
            points.append(B.copy())
            points.append(C.copy())
            centroid = (A + B + C) / 3
            points.append(centroid)
            # Add remaining points randomly inside triangle
            for _ in range(N - 4):
                u = rng.random()
                v = rng.random()
                while u + v > 1:
                    u = rng.random()
                    v = rng.random()
                p = A + u * (B - A) + v * (C - A)
                points.append(p)
            points = np.array(points)
        else:
            # Random initialization
            points = np.zeros((N, 2))
            for i in range(N):
                u = rng.random()
                v = rng.random()
                while u + v > 1:
                    u = rng.random()
                    v = rng.random()
                points[i] = A + u * (B - A) + v * (C - A)
        
        # Evaluate initial
        init_area = min_triangle_area(points)
        if init_area > best_area:
            best_area = init_area
            best_points = points.copy()
        
        # Local optimization using Nelder-Mead
        x0 = points.flatten()
        
        # Use Nelder-Mead with the smooth surrogate
        result = None
        try:
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                },
                beta=30.0
            )
        except Exception:
            pass
        
        if result is not None and result.success:
            opt_points = result.x.reshape(-1, 2)
            # Project any points outside triangle back in
            for i in range(len(opt_points)):
                opt_points[i] = project_to_triangle(opt_points[i], A, B, C)
            
            opt_area = min_triangle_area(opt_points)
            if opt_area > best_area:
                best_area = opt_area
                best_points = opt_points.copy()
        
        # Additional local refinement with higher beta for precision
        if best_points is not None:
            x1 = best_points.flatten()
            try:
                result2 = minimize(
                    objective,
                    x1,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 3000,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    },
                    beta=100.0
                )
                if result2.success:
                    opt_points2 = result2.x.reshape(-1, 2)
                    for i in range(len(opt_points2)):
                        opt_points2[i] = project_to_triangle(opt_points2[i], A, B, C)
                    opt_area2 = min_triangle_area(opt_points2)
                    if opt_area2 > best_area:
                        best_area = opt_area2
                        best_points = opt_points2.copy()
            except Exception:
                pass
        
        if time.time() > deadline - 60.0:
            break
    
    # Final polish pass only if we have enough time left
    if best_points is not None and time.time() < deadline - 10.0:
        x_final = best_points.flatten()
        try:
            result_final = minimize(
                objective,
                x_final,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                },
                beta=100.0
            )
            if result_final.success:
                opt_final = result_final.x.reshape(-1, 2)
                for i in range(len(opt_final)):
                    opt_final[i] = project_to_triangle(opt_final[i], A, B, C)
                final_area = min_triangle_area(opt_final)
                if final_area > best_area:
                    best_points = opt_final
                    best_area = final_area
        except Exception:
            pass
    
    # Ensure all points are strictly inside the triangle
    if best_points is not None:
        for i in range(len(best_points)):
            best_points[i] = project_to_triangle(best_points[i], A, B, C)
    
    return best_points

# EVOLVE-BLOCK-END
