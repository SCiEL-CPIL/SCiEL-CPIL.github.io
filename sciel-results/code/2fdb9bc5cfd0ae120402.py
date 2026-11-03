"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2  # area of unit equilateral triangle


def point_in_triangle(p, eps=1e-9):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, V0, V1)
    d2 = sign(p, V1, V2)
    d3 = sign(p, V2, V0)
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    return not (has_neg and has_pos)


def min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = 0.5 * abs((points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) -
                                 (points[k][0] - points[i][0]) * (points[j][1] - points[i][1]))
                if area < min_area:
                    min_area = area
    return min_area


def negative_min_area_flat(flat_points):
    points = flat_points.reshape(-1, 2)
    # Check if all points are inside the triangle
    for p in points:
        if not point_in_triangle(p):
            return 1.0  # penalty
    return -min_triangle_area(points)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Generate diverse initial points using structured grid with jitter
    def generate_initial_points(seed):
        rng = np.random.RandomState(seed)
        points = []
        # Use a grid-based approach with random perturbation
        for i in range(N):
            # Place points roughly evenly spaced with some randomness
            u = rng.uniform(0.05, 0.95)
            v = rng.uniform(0.05, 0.95)
            # Ensure u + v < 1 for triangle constraint
            while u + v >= 0.95:
                u = rng.uniform(0.05, 0.45)
                v = rng.uniform(0.05, 0.45)
            x = u + 0.5 * v
            y = (3 ** 0.5) / 2 * v
            points.append([x, y])
        return np.array(points)
    
    # Try multiple initial configurations
    best_points = None
    best_score = float('-inf')
    
    # First, try a few deterministic structured configurations
    configs = []
    
    # Configuration 1: Evenly spaced grid-like
    for seed in range(5):
        configs.append(generate_initial_points(seed))
    
    # Configuration 2: More spread out
    for seed in range(5):
        rng = np.random.RandomState(100 + seed)
        points = []
        for i in range(N):
            # Try to spread points more
            angle = 2 * np.pi * i / N + rng.uniform(-0.3, 0.3)
            r = rng.uniform(0.2, 0.7)
            x = 0.5 + r * np.cos(angle)
            y = (np.sqrt(3)/4) + r * 0.8 * np.sin(angle)
            # Clamp to triangle
            if x < 0: x = 0.05
            if x > 1: x = 0.95
            if y < 0: y = 0.05
            if y > np.sqrt(3)/2: y = np.sqrt(3)/2 - 0.05
            # Check if in triangle
            if point_in_triangle(np.array([x, y])):
                points.append([x, y])
            else:
                # Fallback to random
                u, v = rng.uniform(0.05, 0.45), rng.uniform(0.05, 0.45)
                points.append([u + 0.5*v, (3**0.5)/2*v])
        configs.append(np.array(points))
    
    # Configuration 3: Include some vertices and edge midpoints
    special_points = [
        [0.05, 0.05],  # near V0
        [0.95, 0.05],  # near V1
        [0.5, np.sqrt(3)/2 - 0.05],  # near V2
        [0.5, 0.05],  # midpoint of base
        [0.25, np.sqrt(3)/4],  # mid-left
        [0.75, np.sqrt(3)/4],  # mid-right
        [0.5, np.sqrt(3)/8],  # center-bottom
        [0.5, 3*np.sqrt(3)/8],  # center-top
    ]
    for seed in range(3):
        rng = np.random.RandomState(200 + seed)
        pts = [p.copy() for p in special_points]
        while len(pts) < N:
            u, v = rng.uniform(0.05, 0.45), rng.uniform(0.05, 0.45)
            pts.append([u + 0.5*v, (3**0.5)/2*v])
        # Add jitter
        for i in range(len(pts)):
            pts[i][0] += rng.uniform(-0.02, 0.02)
            pts[i][1] += rng.uniform(-0.02, 0.02)
            # Ensure still in triangle
            if not point_in_triangle(np.array(pts[i])):
                pts[i] = special_points[i % len(special_points)].copy()
        configs.append(np.array(pts[:N]))
    
    # Optimize each configuration
    for config in configs:
        if time.time() > deadline - 10:
            break
        
        flat = config.flatten()
        
        # Use Nelder-Mead optimization
        result = minimize(
            negative_min_area_flat,
            flat,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'fatol': 1e-10, 'xatol': 1e-8}
        )
        
        if result.success or result.fun < 0:
            opt_points = result.x.reshape(-1, 2)
            # Verify all points are in triangle
            valid = True
            for p in opt_points:
                if not point_in_triangle(p):
                    valid = False
                    break
            
            if valid:
                score = min_triangle_area(opt_points)
                if score > best_score:
                    best_score = score
                    best_points = opt_points
        
        # Also try a few local refinements from the initial config
        if time.time() > deadline - 10:
            break
        
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            
            # Add small perturbation
            perturbed = config + np.random.uniform(-0.01, 0.01, size=config.shape)
            # Clamp to triangle
            for i in range(len(perturbed)):
                if not point_in_triangle(perturbed[i]):
                    # Revert to original
                    perturbed[i] = config[i]
            
            flat = perturbed.flatten()
            result = minimize(
                negative_min_area_flat,
                flat,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'fatol': 1e-10, 'xatol': 1e-8}
            )
            
            if result.success or result.fun < 0:
                opt_points = result.x.reshape(-1, 2)
                valid = True
                for p in opt_points:
                    if not point_in_triangle(p):
                        valid = False
                        break
                
                if valid:
                    score = min_triangle_area(opt_points)
                    if score > best_score:
                        best_score = score
                        best_points = opt_points
    
    # If no valid points found, return the initial random points
    if best_points is None:
        best_points = configs[0]
    
    return best_points
# EVOLVE-BLOCK-END
