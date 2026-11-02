"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy: 
1. Model the configuration as 10 interior points plus the 3 fixed vertices of the equilateral triangle.
2. Apply a repulsive force field where each pair of points (including vertices) exerts a force 
   inversely proportional to the distance raised to a high power.
3. Combine with a constraint force pushing points toward the triangle center if they approach the boundary.
4. Optimize the total potential energy using L-BFGS-B, starting from multiple diverse initial configurations.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

# Fixed vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
    return min_area

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    for p in pts:
        x, y = p
        if y < -eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
        if x < -eps or x > 1 + eps:
            return False
    return True

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0:
            y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y:
            y = max_y
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        pts[i] = [x, y]
    return pts

def _random_point_in_triangle(rng):
    """Generate a random point inside the equilateral triangle."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    x = u + 0.5 * v
    y = S3H * v
    return [x, y]

def _potential_energy(interior_pts):
    """
    Compute the total potential energy of the configuration.
    interior_pts: array of shape (10, 2) for the 10 interior points.
    The 3 vertices are fixed.
    
    The potential energy consists of:
    1. Repulsive terms between all pairs of points (13 points total: 10 interior + 3 vertices).
       Force is inversely proportional to distance^3, so potential is ~1/d^2.
    2. Constraint penalty for points outside the triangle.
    """
    # All 13 points: 10 interior + 3 vertices
    all_pts = np.vstack([interior_pts, V0, V1, V2])
    n = len(all_pts)
    
    # Repulsive potential: sum over all pairs of 1/d^2
    # This encourages points to be far apart
    repulsive = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(all_pts[i] - all_pts[j])
            if d < 1e-6:
                d = 1e-6
            repulsive += 1.0 / (d * d)
    
    # Constraint penalty: push interior points inside the triangle
    penalty = 0.0
    for p in interior_pts:
        x, y = p
        d1 = y  # distance to bottom edge
        d2 = np.sqrt(3) * x - y  # distance to left edge
        d3 = np.sqrt(3) * (1 - x) - y  # distance to right edge
        if d1 < 0:
            penalty += d1 * d1
        if d2 < 0:
            penalty += d2 * d2
        if d3 < 0:
            penalty += d3 * d3
    
    # Total potential energy: minimize this
    # We want to MINIMIZE the repulsive energy (which means maximizing distances)
    # and MINIMIZE the penalty (which means staying inside)
    return repulsive + 100.0 * penalty

def _potential_energy_grad(interior_pts):
    """
    Compute the gradient of the potential energy with respect to the interior points.
    interior_pts: array of shape (10, 2)
    Returns: array of shape (10, 2) with the gradient.
    """
    all_pts = np.vstack([interior_pts, V0, V1, V2])
    n = len(all_pts)
    n_interior = len(interior_pts)
    
    grad = np.zeros((n_interior, 2))
    
    # Repulsive gradient: for each pair, the force on point i is proportional to (p_i - p_j) / d^3
    for i in range(n):
        for j in range(i + 1, n):
            d_vec = all_pts[i] - all_pts[j]
            d = np.linalg.norm(d_vec)
            if d < 1e-6:
                d = 1e-6
            # Force magnitude: 2/d^3 (derivative of 1/d^2 is -2/d^3, but we want repulsion)
            # Actually, potential is 1/d^2, gradient is -2*(p_i-p_j)/d^3
            # So force on i is -2*(p_i-p_j)/d^3, force on j is +2*(p_i-p_j)/d^3
            force = -2.0 * d_vec / (d * d * d)
            
            if i < n_interior:
                grad[i] += force
            if j < n_interior:
                grad[j] -= force
    
    # Constraint gradient
    for i in range(n_interior):
        x, y = interior_pts[i]
        d1 = y
        d2 = np.sqrt(3) * x - y
        d3 = np.sqrt(3) * (1 - x) - y
        
        if d1 < 0:
            grad[i, 1] += 2.0 * 100.0 * d1
        if d2 < 0:
            grad[i, 0] += 2.0 * 100.0 * d2 * np.sqrt(3)
            grad[i, 1] += -2.0 * 100.0 * d2
        if d3 < 0:
            grad[i, 0] += -2.0 * 100.0 * d3 * np.sqrt(3)
            grad[i, 1] += -2.0 * 100.0 * d3
    
    return grad

def _initial_guess(rng, method='random'):
    """Generate an initial guess for the 10 interior points."""
    if method == 'random':
        pts = np.array([_random_point_in_triangle(rng) for _ in range(10)])
    elif method == 'grid':
        # Place points on a rough grid inside the triangle
        pts = []
        # 3x3 grid, skip corners that are too close to vertices
        for i in range(1, 4):
            for j in range(1, 4):
                x = 0.2 + 0.15 * i
                y = 0.1 + 0.15 * j * 0.8
                if y > np.sqrt(3) * x and y > np.sqrt(3) * (1 - x):
                    pts.append([x, y])
        # Pad if needed
        while len(pts) < 10:
            pts.append(_random_point_in_triangle(rng))
        pts = np.array(pts[:10])
    elif method == 'centroid':
        # Start near the centroid with small random perturbations
        centroid = np.array([0.5, S3H / 3])
        pts = []
        for i in range(10):
            angle = 2 * np.pi * i / 10
            r = 0.15 + 0.05 * rng.random()
            x = centroid[0] + r * np.cos(angle)
            y = centroid[1] + r * np.sin(angle)
            pts.append([x, y])
        pts = np.array(pts)
    else:
        pts = np.array([_random_point_in_triangle(rng) for _ in range(10)])
    
    return _project_to_triangle(pts)

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses a repulsive force field model with L-BFGS-B optimization.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = 0.0
    
    # Try multiple initial configurations
    methods = ['random', 'grid', 'centroid', 'random', 'random', 'grid', 'centroid', 'random']
    
    for method in methods:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        # Generate initial guess
        interior_pts = _initial_guess(rng, method)
        
        # Optimize using L-BFGS-B
        def obj(x_flat):
            pts = x_flat.reshape(10, 2)
            return _potential_energy(pts)
        
        def grad(x_flat):
            pts = x_flat.reshape(10, 2)
            g = _potential_energy_grad(pts)
            return g.flatten()
        
        try:
            result = minimize(
                obj,
                interior_pts.flatten(),
                jac=grad,
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 500}
            )
            
            if result.success or result.status == 0:
                optimized_interior = result.x.reshape(10, 2)
                optimized_interior = _project_to_triangle(optimized_interior)
                
                # Form the full 11-point set
                full_pts = np.vstack([optimized_interior, V0, V1, V2])
                
                if _check_inside(full_pts):
                    score = _min_triangle_area(full_pts)
                    if score > best_score:
                        best_score = score
                        best_pts = full_pts.copy()
        except Exception:
            pass
    
    # If no valid result, return a random configuration
    if best_pts is None:
        interior_pts = np.array([_random_point_in_triangle(rng) for _ in range(10)])
        interior_pts = _project_to_triangle(interior_pts)
        best_pts = np.vstack([interior_pts, V0, V1, V2])
        best_score = _min_triangle_area(best_pts)
    
    return best_pts
