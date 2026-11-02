import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0

def _bary_to_cart(b):
    """Convert barycentric coords (a,b,c) summing to 1 to cartesian in unit equilateral triangle."""
    x = b + 0.5 * c
    y = S3H * c
    return np.array([x, y])

def _cart_to_bary(p):
    """Convert cartesian to barycentric coords."""
    x, y = p
    c = y / S3H
    b = x - 0.5 * c
    a = 1.0 - b - c
    return np.array([a, b, c])

def _min_triangle_area(pts):
    """Compute minimum area of any triangle formed by 3 points."""
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
    """Check if all points are inside or on boundary of equilateral triangle."""
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

def _soft_min_area(pts, beta=10.0):
    """
    Smooth approximation of the minimum triangle area.
    Uses log-sum-exp trick: -1/beta * log(sum(exp(-beta * area_i)))
    approximates min(area_i) for large beta.
    """
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                areas.append(area)
    areas = np.array(areas)
    # Log-sum-exp for stability
    max_area = np.max(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - max_area)))) + (-beta * max_area)
    # -1/beta * log_sum approximates min
    return -1.0 / beta * log_sum

def _barycentric_penalty(bary_pts, w=1000.0):
    """
    Penalty for barycentric coords going outside [0,1].
    bary_pts: (N, 3) array of barycentric coords.
    """
    # Penalty for a < 0, b < 0, c < 0
    a = bary_pts[:, 0]
    b = bary_pts[:, 1]
    c = bary_pts[:, 2]
    penalty = 0.0
    for coord in [a, b, c]:
        neg = np.maximum(0, -coord)
        penalty += np.sum(neg ** 2)
    return w * penalty

def _objective_bary(bary_flat, beta=10.0):
    """
    Objective function in barycentric coordinates.
    bary_flat: (33,) array of 11 barycentric triplets.
    Returns negative soft-min-area + penalty.
    """
    bary_pts = bary_flat.reshape(N, 3)
    # Convert to cartesian
    pts = np.array([_bary_to_cart(b) for b in bary_pts])
    
    # Soft min area
    area = _soft_min_area(pts, beta=beta)
    
    # Penalty for being outside triangle
    pen = _barycentric_penalty(bary_pts)
    
    return -area + pen

def _objective_grad_bary(bary_flat, beta=10.0):
    """
    Numerical gradient for the barycentric objective.
    """
    eps = 1e-6
    grad = np.zeros_like(bary_flat)
    f0 = _objective_bary(bary_flat, beta)
    for i in range(len(bary_flat)):
        bary_flat[i] += eps
        f1 = _objective_bary(bary_flat, beta)
        bary_flat[i] -= 2 * eps
        f2 = _objective_bary(bary_flat, beta)
        bary_flat[i] += eps
        grad[i] = (f1 - f2) / (2 * eps)
    return grad

def _project_bary_to_valid(bary_pts):
    """
    Project barycentric coords to valid region (a,b,c >= 0, sum=1).
    Simple projection: clip negatives to 0, renormalize.
    """
    bary_pts = bary_pts.copy()
    for i in range(N):
        # Clip negatives
        bary_pts[i] = np.maximum(0, bary_pts[i])
        s = np.sum(bary_pts[i])
        if s > 1e-12:
            bary_pts[i] /= s
        else:
            bary_pts[i] = np.array([1.0/3, 1.0/3, 1.0/3])
    return bary_pts

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = 0.0
    
    # Strategy: Topological layering
    # Divide 11 points into layers based on distance from centroid
    # Layer 0: 1 point (centroid-ish)
    # Layer 1: 3 points (inner triangle)
    # Layer 2: 3 points (middle)
    # Layer 3: 4 points (outer)
    
    # But for optimization, we'll use a structured initialization and then optimize
    
    # Initial structured guess in barycentric coords
    # Centroid
    bary_centroid = np.array([1/3, 1/3, 1/3])
    
    # Inner triangle (3 points)
    r_inner = 0.25
    angles_inner = [0, 2*np.pi/3, 4*np.pi/3]
    bary_inner = []
    for ang in angles_inner:
        # In barycentric, radial distance corresponds to deviation from centroid
        # Simplified: place at fixed barycentric offsets
        b = r_inner * np.cos(ang)
        c = r_inner * np.sin(ang)
        a = 1 - b - c
        # Ensure valid
        if a < 0 or b < 0 or c < 0:
            a, b, c = 1/3, 1/3, 1/3
        bary_inner.append([a, b, c])
    
    # Middle ring (3 points)
    r_mid = 0.5
    angles_mid = [np.pi/3, np.pi, 5*np.pi/3]
    bary_mid = []
    for ang in angles_mid:
        b = r_mid * np.cos(ang)
        c = r_mid * np.sin(ang)
        a = 1 - b - c
        if a < 0 or b < 0 or c < 0:
            a, b, c = 1/3, 1/3, 1/3
        bary_mid.append([a, b, c])
    
    # Outer ring (4 points)
    r_outer = 0.75
    angles_outer = [np.pi/6, np.pi/2, 7*np.pi/6, 3*np.pi/2]
    bary_outer = []
    for ang in angles_outer:
        b = r_outer * np.cos(ang)
        c = r_outer * np.sin(ang)
        a = 1 - b - c
        if a < 0 or b < 0 or c < 0:
            a, b, c = 1/3, 1/3, 1/3
        bary_outer.append([a, b, c])
    
    # Combine all barycentric coords
    bary_init = np.array([bary_centroid] + bary_inner + bary_mid + bary_outer)
    bary_init = _project_bary_to_valid(bary_init)
    bary_flat_init = bary_init.flatten()
    
    # Optimize using BFGS
    try:
        result = minimize(
            _objective_bary,
            bary_flat_init,
            method='BFGS',
            options={'maxiter': 200, 'gtol': 1e-6}
        )
        
        if result.success:
            bary_res = _project_bary_to_valid(result.x.reshape(N, 3))
            pts_res = np.array([_bary_to_cart(b) for b in bary_res])
            if _check_inside(pts_res):
                score = _min_triangle_area(pts_res)
                if score > best_score:
                    best_score = score
                    best_pts = pts_res.copy()
    except Exception:
        pass
    
    # Multiple random restarts with local optimization
    num_restarts = 15
    for r in range(num_restarts):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Random barycentric initialization
        bary_rand = np.random.random((N, 3))
        bary_rand /= bary_rand.sum(axis=1, keepdims=True)
        bary_rand = _project_bary_to_valid(bary_rand)
        bary_flat_rand = bary_rand.flatten()
        
        try:
            result = minimize(
                _objective_bary,
                bary_flat_rand,
                method='BFGS',
                options={'maxiter': 100, 'gtol': 1e-5}
            )
            
            if result.success:
                bary_res = _project_bary_to_valid(result.x.reshape(N, 3))
                pts_res = np.array([_bary_to_cart(b) for b in bary_res])
                if _check_inside(pts_res):
                    score = _min_triangle_area(pts_res)
                    if score > best_score:
                        best_score = score
                        best_pts = pts_res.copy()
        except Exception:
            continue
    
    # Local polish: nudge points in minimal triangles
    if best_pts is not None:
        for _ in range(50):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Find minimal triangle
            pts = best_pts
            min_area = float('inf')
            min_triplet = None
            n = len(pts)
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
                            min_triplet = (i, j, k)
            
            if min_triplet is None:
                break
            
            i, j, k = min_triplet
            # Nudge one of the points slightly away from the other two
            # Pick the point with most room to move
            for idx in [i, j, k]:
                # Try small random perturbation
                for _ in range(10):
                    pts_new = pts.copy()
                    pts_new[idx] += np.random.normal(0, 0.02, 2)
                    # Project back inside triangle
                    x, y = pts_new[idx]
                    if x < 0.01: x = 0.01
                    if x > 0.99: x = 0.99
                    if y < 0.01: y = 0.01
                    max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
                    if y > max_y - 0.01: y = max_y - 0.01
                    pts_new[idx] = [x, y]
                    
                    if _check_inside(pts_new):
                        new_score = _min_triangle_area(pts_new)
                        if new_score > best_score:
                            best_score = new_score
                            best_pts = pts_new.copy()
    
    # Fallback
    if best_pts is None:
        # Simple random points
        best_pts = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            while u + v > 1:
                u = rng.random()
                v = rng.random()
            best_pts[i] = [u + 0.5 * v, S3H * v]
    
    return best_pts

if __name__ == "__main__":
    construct()
