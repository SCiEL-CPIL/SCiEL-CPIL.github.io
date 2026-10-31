"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Equilateral triangle vertices
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    
    def point_in_triangle(p):
        """Check if point p is inside or on the triangle ABC."""
        # Using barycentric coordinates
        v0 = B - A
        v1 = C - A
        v2 = p - A
        
        # Compute dot products
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)
        
        # Compute barycentric coordinates
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-10:
            return False
        
        u = (d11 * d20 - d01 * d21) / denom
        v = (d00 * d21 - d01 * d20) / denom
        
        return u >= -1e-9 and v >= -1e-9 and (u + v) <= 1 + 1e-9
    
    def triangle_area(p1, p2, p3):
        """Compute area of triangle formed by three points."""
        return 0.5 * abs(np.cross(p2 - p1, p3 - p1))
    
    def min_triangle_area(points):
        """Compute the minimum area of any triangle formed by three points."""
        min_area = float('inf')
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    area = triangle_area(points[i], points[j], points[k])
                    if area < min_area:
                        min_area = area
        return min_area
    
    def objective(x):
        """Objective function to maximize (we minimize the negative)."""
        points = x.reshape(-1, 2)
        
        # Check if all points are inside the triangle
        for p in points:
            if not point_in_triangle(p):
                return -1000  # Penalty for invalid points
        
        return -min_triangle_area(points)
    
    # Hexagonal packing-inspired initialization
    def hex_init(seed):
        np.random.seed(seed)
        points = []
        
        # Start with vertices and centroid
        points.append(A.copy())
        points.append(B.copy())
        points.append(C.copy())
        points.append((A + B + C) / 3)
        
        # Add points in hexagonal pattern
        # The centroid is at (0.5, sqrt(3)/6)
        centroid = (A + B + C) / 3
        h = np.sqrt(3)/2
        
        # Create a hexagonal grid within the triangle
        # Scale factor for hexagonal packing
        scale = 0.3
        
        # Generate hexagonal points
        for i in range(-3, 4):
            for j in range(-3, 4):
                # Hexagonal coordinates to Cartesian
                x = centroid[0] + scale * (i + 0.5 * j)
                y = centroid[1] + scale * (np.sqrt(3)/2 * j)
                
                if point_in_triangle(np.array([x, y])):
                    # Check if point is not too close to existing points
                    too_close = False
                    for p in points:
                        if np.linalg.norm(np.array([x, y]) - p) < 0.05:
                            too_close = True
                            break
                    if not too_close:
                        points.append(np.array([x, y]))
        
        # If we don't have enough points, add random ones
        while len(points) < N:
            u, v = np.random.random(), np.random.random()
            if u + v < 1:
                p = A + u * (B - A) + v * (C - A)
                if point_in_triangle(p):
                    # Check distance from existing points
                    too_close = False
                    for q in points:
                        if np.linalg.norm(p - q) < 0.05:
                            too_close = True
                            break
                    if not too_close:
                        points.append(p)
        
        # If we still don't have enough, just add random points
        while len(points) < N:
            u, v = np.random.random(), np.random.random()
            if u + v < 1:
                p = A + u * (B - A) + v * (C - A)
                points.append(p)
        
        return np.array(points[:N])
    
    # Nelder-Mead optimization with bounds handling
    def optimize(points_init, max_iter=2000):
        x0 = points_init.flatten()
        
        # Use Nelder-Mead
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8})
        
        return result.x.reshape(-1, 2), -result.fun
    
    # Try multiple initializations and keep the best
    best_points = None
    best_area = -float('inf')
    
    # Try several different seeds for hexagonal init
    for seed in range(20):
        if time.time() > deadline:
            break
        
        points_init = hex_init(seed)
        points_opt, area = optimize(points_init, max_iter=1000)
        
        # Verify the result is valid
        valid = all(point_in_triangle(p) for p in points_opt)
        if valid and area > best_area:
            best_area = area
            best_points = points_opt
        
        # Also try a simple local search from the initial points
        if time.time() > deadline:
            break
        
        # Simple gradient-free local search
        current_points = points_init.copy()
        current_area = min_triangle_area(current_points)
        
        for iteration in range(500):
            if time.time() > deadline:
                break
            
            # Try perturbing each point
            improved = False
            for i in range(N):
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    
                    # Random perturbation
                    delta = np.random.randn(2) * 0.05
                    new_point = current_points[i] + delta
                    
                    # Check if new point is inside triangle
                    if not point_in_triangle(new_point):
                        continue
                    
                    # Check minimum distance to other points
                    too_close = False
                    for j in range(N):
                        if i != j and np.linalg.norm(new_point - current_points[j]) < 0.01:
                            too_close = True
                            break
                    if too_close:
                        continue
                    
                    # Try the perturbation
                    old_point = current_points[i].copy()
                    current_points[i] = new_point
                    new_area = min_triangle_area(current_points)
                    
                    if new_area > current_area:
                        current_area = new_area
                        improved = True
                    else:
                        current_points[i] = old_point
            
            if not improved:
                break
        
        valid = all(point_in_triangle(p) for p in current_points)
        if valid and current_area > best_area:
            best_area = current_area
            best_points = current_points
    
    # If we didn't find anything, return a basic configuration
    if best_points is None:
        best_points = hex_init(0)
    
    return best_points
# EVOLVE-BLOCK-END
