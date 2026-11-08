"""Heilbronn triangles: maximize min triangle area for 11 points in equilateral triangle."""
import numpy as np
import time
from itertools import combinations

N = 11
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
TRI_AREA = np.sqrt(3) / 4.0


def _min_triangle_area(pts):
    """Compute minimum area of any triangle formed by 3 of the points."""
    n = len(pts)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = 0.5 * abs((pts[j][0] - pts[i][0]) * (pts[k][1] - pts[i][1]) -
                              (pts[j][1] - pts[i][1]) * (pts[k][0] - pts[i][0]))
                if a < min_a:
                    min_a = a
    return min_a


def _is_inside_triangle(p, eps=1e-9):
    """Check if point p is inside or on the boundary of the equilateral triangle."""
    x, y = p
    if y < -eps or y > np.sqrt(3) / 2 + eps:
        return False
    if x < -eps or x > 1 + eps:
        return False
    # Check below the line from (0,0) to (0.5, sqrt(3)/2): y <= sqrt(3)*x
    if y > np.sqrt(3) * x + eps:
        return False
    # Check below the line from (1,0) to (0.5, sqrt(3)/2): y <= -sqrt(3)*(x-1)
    if y > -np.sqrt(3) * (x - 1) + eps:
        return False
    return True


def _clamp_to_triangle(p):
    """Clamp a point to be inside the triangle."""
    x, y = p
    # Clamp to bounding box first
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(np.sqrt(3) / 2, y))
    # Check and fix for the two slanted edges
    # Edge from (0,0) to (0.5, sqrt(3)/2): y = sqrt(3)*x
    if y > np.sqrt(3) * x:
        # Project onto this edge or move inward
        # The constraint is y <= sqrt(3)*x, so if violated, reduce y or increase x
        # Simplest: set y = sqrt(3)*x (on the edge)
        y = np.sqrt(3) * x
        if y > np.sqrt(3) / 2:
            y = np.sqrt(3) / 2
            x = 0.5
    # Edge from (1,0) to (0.5, sqrt(3)/2): y = -sqrt(3)*(x-1) = sqrt(3)*(1-x)
    if y > np.sqrt(3) * (1 - x):
        y = np.sqrt(3) * (1 - x)
        if y > np.sqrt(3) / 2:
            y = np.sqrt(3) / 2
            x = 0.5
    return np.array([x, y])


def _local_optimize(pts, max_iter=500, step0=0.05, lr_decay=0.95):
    """Gradient-free local optimization using coordinate descent with random perturbations."""
    pts = pts.copy()
    best_pts = pts.copy()
    best_val = _min_triangle_area(pts)
    step = step0
    rng = np.random.RandomState(42)
    
    for it in range(max_iter):
        improved = False
        for i in range(N):
            # Try random perturbations around current position
            best_local = pts[i].copy()
            best_local_val = best_val
            
            for _ in range(30):
                candidate = pts[i].copy()
                candidate += rng.normal(0, step, 2)
                candidate = _clamp_to_triangle(candidate)
                
                test_pts = pts.copy()
                test_pts[i] = candidate
                val = _min_triangle_area(test_pts)
                
                if val > best_local_val:
                    best_local_val = val
                    best_local = candidate
                    improved = True
        
        if improved:
            pts[i] = best_local
            if best_local_val > best_val:
                best_val = best_local_val
                best_pts = pts.copy()
        
        step *= lr_decay
        if step < 1e-6:
            break
    
    return best_pts, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Start with a structured grid-like configuration
    # Place points at vertices and along edges, then fill interior
    # Use a hexagonal/triangular lattice pattern
    
    # Initial structured placement: 3 corners + 3 edge midpoints + 5 interior
    base_points = [
        [0.0, 0.0],           # V0
        [1.0, 0.0],           # V1
        [0.5, np.sqrt(3)/2],  # V2
        [0.25, np.sqrt(3)/4],  # midpoint of V0-V2
        [0.75, np.sqrt(3)/4],  # midpoint of V1-V2
        [0.5, 0.0],           # midpoint of V0-V1
        # Interior points - centroid and surrounding
        [0.5, np.sqrt(3)/6],   # centroid
        [0.35, np.sqrt(3)/6],
        [0.65, np.sqrt(3)/6],
        [0.5, np.sqrt(3)/3],
    ]
    
    pts = np.array(base_points, dtype=float)
    
    # Ensure all points are inside
    for i in range(N):
        pts[i] = _clamp_to_triangle(pts[i])
    
    best_pts = pts.copy()
    best_val = _min_triangle_area(pts)
    
    # Run local optimization with multiple restarts
    rng = np.random.RandomState(123)
    
    # First, optimize the initial structured configuration
    if time.time() < deadline - 5:
        opt_pts, opt_val = _local_optimize(pts, max_iter=300, step0=0.03)
        if opt_val > best_val:
            best_val = opt_val
            best_pts = opt_pts.copy()
    
    # Try additional restarts with perturbed starting points
    n_restarts = 0
    while time.time() < deadline - 5:
        n_restarts += 1
        
        # Generate a new starting configuration
        if n_restarts % 3 == 1:
            # Perturb the best found
            new_pts = best_pts + rng.normal(0, 0.02, (N, 2))
            for i in range(N):
                new_pts[i] = _clamp_to_triangle(new_pts[i])
        elif n_restarts % 3 == 2:
            # Structured grid with jitter
            # Use a triangular lattice
            rows = [3, 4, 4]  # points per row
            new_pts = []
            y_positions = [0.05, 0.25, 0.45]
            for r, y in enumerate(y_positions):
                n_in_row = rows[r]
                x_start = 0.05 + y / 2
                x_end = 0.95 - y / 2
                for c in range(n_in_row):
                    if n_in_row == 1:
                        x = 0.5
                    else:
                        x = x_start + c * (x_end - x_start) / (n_in_row - 1)
                    new_pts.append([x, y])
            new_pts = np.array(new_pts, dtype=float)
            for i in range(N):
                new_pts[i] = _clamp_to_triangle(new_pts[i])
        else:
            # Random but with minimum distance enforcement
            new_pts = []
            min_dist = 0.1
            attempts = 0
            while len(new_pts) < N and attempts < 500:
                attempts += 1
                u, v = rng.random(), rng.random()
                if u + v < 1.0:
                    x = u + 0.5 * v
                    y = np.sqrt(3) / 2 * v
                    # Check minimum distance
                    if all(np.linalg.norm([x, y] - p) > min_dist for p in new_pts):
                        new_pts.append([x, y])
            if len(new_pts) < N:
                # Fill remaining with structured positions
                for i in range(len(new_pts), N):
                    new_pts.append(base_points[i % len(base_points)].copy())
            new_pts = np.array(new_pts, dtype=float)
            for i in range(N):
                new_pts[i] = _clamp_to_triangle(new_pts[i])
        
        if time.time() < deadline - 5:
            opt_pts, opt_val = _local_optimize(new_pts, max_iter=200, step0=0.02)
            if opt_val > best_val:
                best_val = opt_val
                best_pts = opt_pts.copy()
    
    # Final check: ensure all points are inside
    for i in range(N):
        best_pts[i] = _clamp_to_triangle(best_pts[i])
    
    return best_pts
