"""Rewritten program for Heilbronn triangles (11 points in unit equilateral triangle)."""
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2  # area of the equilateral triangle

DEADLINE = 115.0  # seconds; safety margin before 120 s limit
START_TIME = time.time()


def elapsed():
    return time.time() - START_TIME


def project_to_triangle(points):
    """Project each point into the triangle using barycentric coordinates with clipping."""
    # Barycentric coordinates relative to V0, V1, V2
    # p = a*V0 + b*V1 + c*V2, a+b+c=1
    # Solve for b, c: p - V0 = b*(V1-V0) + c*(V2-V0)
    A = np.array([[V1[0] - V0[0], V2[0] - V0[0]],
                  [V1[1] - V0[1], V2[1] - V0[1]]])
    A_inv = np.linalg.inv(A)
    pts = points - V0
    bc = (A_inv @ pts.T).T  # shape (n, 2): b, c
    a = 1.0 - bc[:, 0] - bc[:, 1]
    # Clip barycentric coords
    bary = np.stack([a, bc[:, 0], bc[:, 1]], axis=1)
    bary = np.clip(bary, 0.0, 1.0)
    # Renormalize
    s = bary.sum(axis=1, keepdims=True)
    s = np.where(s < 1e-12, 1.0, s)
    bary = bary / s
    # Convert back to Cartesian
    return bary[:, 0:1] * V0 + bary[:, 1:2] * V1 + bary[:, 2:3] * V2


def min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(points)
    # Precompute all pairwise differences
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) -
                                 (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                if area < min_area:
                    min_area = area
    return min_area


def objective_flat(x_flat):
    """Objective: negative of min triangle area. Points are projected to stay in triangle."""
    points = x_flat.reshape(N, 2)
    points = project_to_triangle(points)
    area = min_triangle_area(points)
    return -area


def generate_initial_points():
    """Generate diverse starting configurations."""
    starts = []

    # Start 1: Structured placement - corners, edge midpoints, centroid, and interior points
    # Corners (slightly inset to avoid boundary issues)
    eps = 0.02
    c0 = V0 + eps * (V1 - V0) + eps * (V2 - V0)
    c1 = V1 - eps * (V1 - V0) + eps * (V2 - V0)
    c2 = V2 + eps * (V1 - V2) + eps * (V0 - V2)
    
    # Edge midpoints (slightly inset)
    m01 = 0.5 * (V0 + V1) + eps * (V2 - 0.5 * (V0 + V1))
    m12 = 0.5 * (V1 + V2) + eps * (V0 - 0.5 * (V1 + V2))
    m20 = 0.5 * (V2 + V0) + eps * (V1 - 0.5 * (V2 + V0))
    
    # Centroid
    centroid = (V0 + V1 + V2) / 3.0
    
    # 4 interior points: Fermat-like positions
    # Place at fractions along medians
    p1 = centroid + 0.15 * (V0 - centroid)
    p2 = centroid + 0.15 * (V1 - centroid)
    p3 = centroid + 0.15 * (V2 - centroid)
    p4 = centroid
    
    starts.append(np.array([c0, c1, c2, m01, m12, m20, centroid, p1, p2, p3, p4]))
    
    # Start 2: Grid-based with jitter
    rng = np.random.RandomState(42)
    grid_pts = []
    # 3x3 grid in barycentric coords
    for i in range(3):
        for j in range(3):
            a = (i + 0.5) / 3.0
            b = (j + 0.5) / 3.0
            c = 1.0 - a - b
            if c > 0:
                pt = a * V0 + b * V1 + c * V2
                grid_pts.append(pt)
    # Take first 11
    if len(grid_pts) >= 11:
        starts.append(np.array(grid_pts[:11]))
    
    # Start 3: Random points
    rng = np.random.RandomState(123)
    while True:
        u = rng.random()
        v = rng.random()
        if u + v < 1.0:
            pt = V0 + u * (V1 - V0) + v * (V2 - V0)
            if len(starts[2]) < N if len(starts) > 2 else False:
                pass
            break
    # Generate 11 random points
    rand_pts = []
    for _ in range(N):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v < 1.0:
                pt = V0 + u * (V1 - V0) + v * (V2 - V0)
                rand_pts.append(pt)
                break
    starts.append(np.array(rand_pts))
    
    # Start 4: Slightly perturbed structured
    base = starts[0].copy()
    base = base + rng.normal(0, 0.02, size=base.shape)
    base = project_to_triangle(base)
    starts.append(base)
    
    return starts


def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_points = None
    best_area = 0.0
    
    starts = generate_initial_points()
    
    for idx, start_pts in enumerate(starts):
        if elapsed() > DEADLINE - 5.0:
            break
        
        x0 = start_pts.flatten()
        
        try:
            result = minimize(
                objective_flat,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if elapsed() > DEADLINE - 2.0:
                break
            
            # Evaluate the result
            result_pts = result.x.reshape(N, 2)
            result_pts = project_to_triangle(result_pts)
            area = min_triangle_area(result_pts)
            
            if area > best_area:
                best_area = area
                best_points = result_pts.copy()
                
        except Exception:
            continue
    
    # If no optimization succeeded, use the best initial
    if best_points is None:
        for start_pts in starts:
            pts = project_to_triangle(start_pts)
            area = min_triangle_area(pts)
            if area > best_area:
                best_area = area
                best_points = pts.copy()
    
    return best_points
