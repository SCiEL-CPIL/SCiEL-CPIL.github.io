"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill
    
    # Triangle vertices
    A = np.array([0.0, 0.0])
    B = np.array([1.0, 0.0])
    C = np.array([0.5, np.sqrt(3)/2])
    H = np.sqrt(3)/2  # height of equilateral triangle
    
    def is_inside_triangle(p):
        """Check if point p is inside or on the boundary of the triangle."""
        # Using barycentric coordinates
        v0 = B - A
        v1 = C - A
        v2 = p - A
        
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d02 = np.dot(v0, v2)
        d11 = np.dot(v1, v1)
        d12 = np.dot(v1, v2)
        
        denom = d00 * d11 - d01 * d01
        if denom == 0:
            return False
            
        u = (d11 * d02 - d01 * d12) / denom
        v = (d00 * d12 - d01 * d02) / denom
        
        return (u >= -1e-10) and (v >= -1e-10) and (u + v <= 1 + 1e-10)
    
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
    
    def objective_flat(x):
        """Objective function: negative of min triangle area (to minimize)."""
        points = x.reshape(N, 2)
        # Penalty for points outside triangle
        penalty = 0.0
        for p in points:
            if not is_inside_triangle(p):
                # Distance to nearest edge (approximate)
                # For simplicity, use a large penalty
                penalty += 1.0
        if penalty > 0:
            return -min_triangle_area(points) + penalty * 10.0
        return -min_triangle_area(points)
    
    def smooth_min_triangle_area(points, beta=10.0):
        """Smooth approximation of min triangle area using log-sum-exp."""
        areas = []
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    area = triangle_area(points[i], points[j], points[k])
                    areas.append(area)
        areas = np.array(areas)
        if len(areas) == 0:
            return 0.0
        # Use log-sum-exp for soft-min
        m = np.min(areas)
        if m < 1e-15:
            return m
        log_areas = np.log(areas)
        log_sum_exp = m + np.log(np.sum(np.exp(log_areas - m)))
        # Soft-min = -1/beta * log(sum(exp(-beta * areas)))
        # = -1/beta * (-beta * m - log(sum(exp(-beta*(areas-m)))))
        # = m + 1/beta * log(sum(exp(-beta*(areas-m))))
        soft_min = m + (1.0/beta) * np.log(np.sum(np.exp(-beta * (areas - m))))
        return soft_min
    
    def objective_smooth(x, beta=10.0):
        """Smooth objective: negative of soft-min triangle area."""
        points = x.reshape(N, 2)
        # Check if all points are inside triangle
        for p in points:
            if not is_inside_triangle(p):
                return -min_triangle_area(points) + 10.0
        return -smooth_min_triangle_area(points, beta)
    
    def gradient_smooth(x, beta=10.0):
        """Numerical gradient of smooth objective."""
        eps = 1e-6
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (objective_smooth(x_plus, beta) - objective_smooth(x_minus, beta)) / (2 * eps)
        return grad
    
    # Initial points: use a structured initialization
    # Place points in a grid-like pattern within the triangle
    def initial_points():
        pts = []
        # Corners
        pts.append(A + 0.02 * (B - A) + 0.02 * (C - A))
        pts.append(B - 0.02 * (B - A) + 0.02 * (C - B))
        pts.append(C + 0.02 * (A - C) + 0.02 * (B - C))
        
        # Midpoints of edges
        pts.append(0.5 * (A + B))
        pts.append(0.5 * (B + C))
        pts.append(0.5 * (C + A))
        
        # Center
        pts.append((A + B + C) / 3.0)
        
        # Additional points
        pts.append(0.25 * A + 0.5 * B + 0.25 * C)
        pts.append(0.5 * A + 0.25 * B + 0.25 * C)
        pts.append(0.25 * A + 0.25 * B + 0.5 * C)
        
        return np.array(pts)
    
    # Start with initial points
    best_points = initial_points()
    best_score = min_triangle_area(best_points)
    
    # Try multiple restarts with different initializations
    n_restarts = 5
    rng = np.random.RandomState(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            x0 = best_points.flatten()
        else:
            # Random initialization with some structure
            x0 = best_points.flatten().copy()
            # Add random perturbation
            x0 += rng.normal(0, 0.05, size=x0.shape)
        
        # Use L-BFGS-B with bounds to keep points in triangle
        # Bounds: x in [0, 1], y in [0, H]
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))
            bounds.append((0.0, H))
        
        # Optimization with smooth objective
        try:
            result = minimize(
                objective_smooth,
                x0,
                args=(10.0,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success:
                opt_points = result.x.reshape(N, 2)
                # Project points back into triangle if needed
                for i in range(N):
                    if not is_inside_triangle(opt_points[i]):
                        # Move to nearest point in triangle (simple projection)
                        # For now, just clamp
                        opt_points[i, 0] = np.clip(opt_points[i, 0], 0, 1)
                        opt_points[i, 1] = np.clip(opt_points[i, 1], 0, H)
                        # Additional check
                        if not is_inside_triangle(opt_points[i]):
                            # Use barycentric projection
                            v0 = B - A
                            v1 = C - A
                            v2 = opt_points[i] - A
                            d00 = np.dot(v0, v0)
                            d01 = np.dot(v0, v1)
                            d02 = np.dot(v0, v2)
                            d11 = np.dot(v1, v1)
                            d12 = np.dot(v1, v2)
                            denom = d00 * d11 - d01 * d01
                            if denom != 0:
                                u = (d11 * d02 - d01 * d12) / denom
                                v = (d00 * d12 - d01 * d02) / denom
                                # Clamp u, v
                                u = np.clip(u, 0, 1)
                                v = np.clip(v, 0, 1)
                                if u + v > 1:
                                    # Scale back
                                    scale = 1.0 / (u + v)
                                    u *= scale
                                    v *= scale
                                opt_points[i] = A + u * v0 + v * v1
                
                score = min_triangle_area(opt_points)
                if score > best_score:
                    best_score = score
                    best_points = opt_points.copy()
        except Exception as e:
            pass
    
    # Final polish with exact objective
    if time.time() < deadline - 5:
        x0 = best_points.flatten()
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0))
            bounds.append((0.0, H))
        
        try:
            result = minimize(
                objective_flat,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success:
                opt_points = result.x.reshape(N, 2)
                for i in range(N):
                    if not is_inside_triangle(opt_points[i]):
                        v0 = B - A
                        v1 = C - A
                        v2 = opt_points[i] - A
                        d00 = np.dot(v0, v0)
                        d01 = np.dot(v0, v1)
                        d02 = np.dot(v0, v2)
                        d11 = np.dot(v1, v1)
                        d12 = np.dot(v1, v2)
                        denom = d00 * d11 - d01 * d01
                        if denom != 0:
                            u = (d11 * d02 - d01 * d12) / denom
                            v = (d00 * d12 - d01 * d02) / denom
                            u = np.clip(u, 0, 1)
                            v = np.clip(v, 0, 1)
                            if u + v > 1:
                                scale = 1.0 / (u + v)
                                u *= scale
                                v *= scale
                            opt_points[i] = A + u * v0 + v * v1
                
                score = min_triangle_area(opt_points)
                if score > best_score:
                    best_score = score
                    best_points = opt_points.copy()
        except Exception as e:
            pass
    
    return best_points
# EVOLVE-BLOCK-END
