"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import itertools

# Unit equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
CENTROID = (V0 + V1 + V2) / 3.0

def point_in_triangle(p, a=V0, b=V1, c=V2, eps=1e-10):
    """Check if point p is inside or on the boundary of triangle abc."""
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    
    return not (has_neg and has_pos)

def min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def negative_min_area_flat(x):
    """Objective function for minimization: negative of minimum triangle area.
    Uses penalty for points outside the triangle."""
    points = x.reshape(-1, 2)
    
    # Penalty for points outside the triangle
    penalty = 0.0
    for p in points:
        if not point_in_triangle(p):
            # Compute distance to nearest edge
            dist = 0.0
            edges = [(V0, V1), (V1, V2), (V2, V0)]
            for a, b in edges:
                ab = b - a
                t = np.dot(p - a, ab) / np.dot(ab, ab)
                t = np.clip(t, 0, 1)
                proj = a + t * ab
                d = np.linalg.norm(p - proj)
                dist = max(dist, d)
            penalty += dist * 100.0
    
    if penalty > 0:
        return -1e-10 + penalty
    
    return -min_triangle_area(points)

def project_to_triangle(p):
    """Project point p to the closest point inside the triangle."""
    if point_in_triangle(p):
        return p
    
    # Find closest point on the triangle boundary
    best = p
    best_dist = float('inf')
    
    edges = [(V0, V1), (V1, V2), (V2, V0)]
    for a, b in edges:
        # Project p onto line segment ab
        ab = b - a
        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        proj = a + t * ab
        dist = np.linalg.norm(p - proj)
        if dist < best_dist:
            best_dist = dist
            best = proj
    
    # Move slightly inward toward centroid
    best = 0.995 * best + 0.005 * CENTROID
    
    return best

def project_all_to_triangle(points):
    """Project all points to be inside the triangle."""
    for i in range(len(points)):
        points[i] = project_to_triangle(points[i])
    return points

def generate_structured_starts():
    """Generate diverse starting configurations using structured patterns."""
    starts = []
    
    # Pattern 1: Grid-based configurations
    n_grid = 4
    for i in range(n_grid):
        for j in range(n_grid - i):
            u = (i + 0.5) / n_grid
            v = (j + 0.5) / (n_grid - i)
            center = np.array([u * (1 - v) + 0.5 * v, (np.sqrt(3) / 2) * v])
            
            config = np.zeros((N, 2))
            for k in range(N):
                angle = 2 * np.pi * k / N + 0.3 * (k % 2)
                r = 0.12 + 0.04 * (k % 3)
                config[k] = center + r * np.array([np.cos(angle), np.sin(angle)])
                config[k] = project_to_triangle(config[k])
            starts.append(config.flatten())
    
    # Pattern 2: Hexagonal packing
    hex_centers = [
        (0.2, 0.1), (0.5, 0.1), (0.8, 0.1),
        (0.35, 0.3), (0.65, 0.3),
        (0.5, 0.5),
        (0.3, 0.6), (0.7, 0.6),
        (0.4, 0.75), (0.6, 0.75),
        (0.5, 0.85)
    ]
    config = np.zeros((N, 2))
    for k, (x, y) in enumerate(hex_centers):
        config[k] = np.array([x, y])
    config = project_all_to_triangle(config)
    starts.append(config.flatten())
    
    # Pattern 3: Vertices + edges + center
    config = np.zeros((N, 2))
    # 3 vertices
    config[0] = V0 + 0.05 * (CENTROID - V0)
    config[1] = V1 + 0.05 * (CENTROID - V1)
    config[2] = V2 + 0.05 * (CENTROID - V2)
    # 3 edge midpoints
    config[3] = (V0 + V1) / 2 + 0.03 * (CENTROID - (V0 + V1) / 2)
    config[4] = (V1 + V2) / 2 + 0.03 * (CENTROID - (V1 + V2) / 2)
    config[5] = (V2 + V0) / 2 + 0.03 * (CENTROID - (V2 + V0) / 2)
    # 3 inner ring
    for k in range(3):
        angle = 2 * np.pi * k / 3
        r = 0.2
        config[6 + k] = CENTROID + r * np.array([np.cos(angle), np.sin(angle)])
    # 2 more inner
    config[9] = CENTROID + 0.1 * np.array([0.5, 0.5])
    config[10] = CENTROID + 0.1 * np.array([-0.5, 0.5])
    config = project_all_to_triangle(config)
    starts.append(config.flatten())
    
    # Pattern 4: Random with seed
    np.random.seed(123)
    for _ in range(3):
        config = np.zeros((N, 2))
        for k in range(N):
            u = np.random.random()
            v = np.random.random()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            config[k] = np.array([u + 0.5 * v, (np.sqrt(3) / 2) * v])
        config = project_all_to_triangle(config)
        starts.append(config.flatten())
    
    return starts

def nelder_mead_optimize(x0, maxiter=200, tol=1e-8):
    """Run Nelder-Mead optimization with tight tolerance."""
    try:
        result = minimize(
            negative_min_area_flat,
            x0,
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': tol, 'fatol': tol, 'adaptive': True}
        )
        return result.x
    except Exception:
        return x0

def local_refine(points, n_iter=50):
    """Local refinement by moving points to increase minimum area."""
    points = points.copy()
    best_area = min_triangle_area(points)
    
    for _ in range(n_iter):
        improved = False
        # Find the triangle with minimum area
        min_area = float('inf')
        min_idx = None
        for i in range(N):
            for j in range(i + 1, N):
                for k in range(j + 1, N):
                    area = 0.5 * abs(
                        (points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) -
                        (points[k][0] - points[i][0]) * (points[j][1] - points[i][1])
                    )
                    if area < min_area:
                        min_area = area
                        min_idx = (i, j, k)
        
        if min_area >= best_area * (1 + 1e-10):
            break
        
        i, j, k = min_idx
        # Try moving each of the three points slightly
        for idx in [i, j, k]:
            p = points[idx].copy()
            # Try small perturbations
            for dx in [-0.02, -0.01, 0.01, 0.02]:
                for dy in [-0.02, -0.01, 0.01, 0.02]:
                    new_p = p + np.array([dx, dy])
                    new_p = project_to_triangle(new_p)
                    test_points = points.copy()
                    test_points[idx] = new_p
                    new_area = min_triangle_area(test_points)
                    if new_area > best_area + 1e-10:
                        points[idx] = new_p
                        best_area = new_area
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
    return points, best_area

def construct():
    """Construct 11 points inside the unit equilateral triangle to maximize the minimum triangle area."""
    
    # Generate diverse starting points
    starts = generate_structured_starts()
    
    best_x = None
    best_score = -float('inf')
    
    # Run multi-start optimization
    for x0 in starts:
        # Ensure starting points are valid
        x0_points = x0.reshape(-1, 2)
        x0_points = project_all_to_triangle(x0_points)
        x0 = x0_points.flatten()
        
        # Run Nelder-Mead optimization
        x_opt = nelder_mead_optimize(x0, maxiter=150)
        
        # Evaluate
        points = x_opt.reshape(-1, 2)
        points = project_all_to_triangle(points)
        
        score = min_triangle_area(points)
        
        if score > best_score:
            best_score = score
            best_x = points.copy()
    
    # Local refinement on the best solution
    if best_x is not None:
        refined_points, refined_score = local_refine(best_x, n_iter=30)
        if refined_score > best_score:
            best_x = refined_points
            best_score = refined_score
    
    return best_x
# EVOLVE-BLOCK-END
