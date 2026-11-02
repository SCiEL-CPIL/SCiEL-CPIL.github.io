"""Combined program for Heilbronn-11: Barycentric optimization with soft-min annealing."""
import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11

# EVOLVE-BLOCK-START

def _bary_to_cart(bary):
    """Convert barycentric coordinates to Cartesian.
    bary has shape (n, 3) with each row summing to 1.
    Vertices: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    """
    h = np.sqrt(3) / 2
    # bary[:, 0] * A + bary[:, 1] * B + bary[:, 2] * C
    x = bary[:, 1] * 1.0 + bary[:, 2] * 0.5
    y = bary[:, 2] * h
    return np.column_stack([x, y])

def _cart_to_bary(cart):
    """Convert Cartesian to barycentric coordinates.
    cart has shape (n, 2).
    """
    h = np.sqrt(3) / 2
    x = cart[:, 0]
    y = cart[:, 1]
    # bary_C = y / h
    bary_c = y / h
    # bary_B = x - 0.5 * bary_C
    bary_b = x - 0.5 * bary_c
    # bary_A = 1 - bary_B - bary_C
    bary_a = 1.0 - bary_b - bary_c
    return np.column_stack([bary_a, bary_b, bary_c])

def _vectorized_min_area_cart(points):
    """Compute all triangle areas vectorized. Returns array of shape (n, n, n)."""
    n = len(points)
    # Compute cross products for all triples
    # For points i, j, k: area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    # Using broadcasting
    p1 = points[:, np.newaxis, np.newaxis, :]  # (n, 1, 1, 2)
    p2 = points[np.newaxis, :, np.newaxis, :]  # (1, n, 1, 2)
    p3 = points[np.newaxis, np.newaxis, :, :]  # (1, 1, n, 2)
    
    # v1 = p2 - p1, v2 = p3 - p1
    v1 = p2 - p1  # (n, n, 1, 2)
    v2 = p3 - p1  # (n, n, n, 2)
    
    # Cross product z-component
    cross = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    areas = 0.5 * np.abs(cross)
    
    # Only keep i < j < k
    # Create mask
    idx = np.arange(n)
    mask = (idx[:, np.newaxis, np.newaxis] < idx[np.newaxis, :, np.newaxis]) & \
           (idx[np.newaxis, :, np.newaxis] < idx[np.newaxis, np.newaxis, :])
    
    return areas, mask

def _soft_min_area_cart(points, temperature=0.01):
    """Compute soft-min of triangle areas using log-sum-exp."""
    n = len(points)
    areas, mask = _vectorized_min_area_cart(points)
    
    # Extract valid areas (i < j < k)
    valid_areas = areas[mask]
    
    if len(valid_areas) == 0:
        return 0.0
    
    # Soft-min: -logsumexp(-x/t) * t
    neg_areas = -valid_areas / temperature
    max_neg = np.max(neg_areas)
    logsumexp_val = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
    softmin_val = -logsumexp_val * temperature
    
    return softmin_val

def _project_cartesian(points):
    """Project points to be inside the triangle."""
    h = np.sqrt(3) / 2
    projected = points.copy()
    
    for i in range(len(projected)):
        x, y = projected[i]
        
        # Check and project to triangle
        # y >= 0
        if y < 0:
            y = 0
        
        # y <= h * x
        if y > h * x:
            # Project onto line y = h*x
            dist = (h * x - y) / (h * h + 1)
            x = x - dist * h
            y = y + dist
        
        # y <= h * (1-x)
        if y > h * (1 - x):
            # Project onto line y = h*(1-x) => hx + y = h
            dist = (h * x + y - h) / (h * h + 1)
            x = x - dist * h
            y = y - dist
        
        # x in [0, 1]
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        
        projected[i] = [x, y]
    
    return projected

def _is_valid_cart(points, eps=1e-9):
    """Check if all points are inside the triangle."""
    h = np.sqrt(3) / 2
    for p in points:
        x, y = p
        if y < -eps or x < -eps or x > 1 + eps:
            return False
        if y > h * x + eps or y > h * (1 - x) + eps:
            return False
    return True

def _random_bary(rng, n=N):
    """Generate random barycentric coordinates."""
    # Generate random points in barycentric space
    # Use Dirichlet distribution
    bary = np.zeros((n, 3))
    for i in range(n):
        # Random uniform in triangle
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        bary[i] = [1 - u - v, u, v]
    return bary

def _bary_objective(bary_flat, temperature=0.01):
    """Objective function in barycentric coordinates."""
    n = len(bary_flat) // 3
    bary = bary_flat.reshape(n, 3)
    
    # Normalize to sum to 1
    bary = bary / np.sum(bary, axis=1, keepdims=True)
    
    # Convert to Cartesian
    cart = _bary_to_cart(bary)
    
    # Compute soft-min area
    softmin = _soft_min_area_cart(cart, temperature)
    
    # Penalty for points outside triangle (should be zero if barycentric is valid)
    penalty = 0.0
    for i in range(n):
        if bary[i, 0] < -1e-6 or bary[i, 1] < -1e-6 or bary[i, 2] < -1e-6:
            penalty += (min(bary[i]) + 1e-6) ** 2
    
    return -softmin + penalty

def _cartesian_objective(cart_flat, temperature=0.01):
    """Objective function in Cartesian coordinates."""
    n = len(cart_flat) // 2
    cart = cart_flat.reshape(n, 2)
    
    # Compute soft-min area
    softmin = _soft_min_area_cart(cart, temperature)
    
    # Penalty for points outside triangle
    h = np.sqrt(3) / 2
    penalty = 0.0
    for p in cart:
        x, y = p
        if y < 0:
            penalty += y ** 2
        if x < 0:
            penalty += x ** 2
        if x > 1:
            penalty += (x - 1) ** 2
        if y > h * x:
            penalty += (y - h * x) ** 2
        if y > h * (1 - x):
            penalty += (y - h * (1 - x)) ** 2
    
    return -softmin + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    best_points = None
    best_score = -float('inf')
    
    rng = np.random.RandomState(42)
    
    # Generate multiple initial configurations
    n_configs = 30
    configs = []
    
    # Config 1: Symmetric arrangement
    h = np.sqrt(3) / 2
    v1 = [0, 0]
    v2 = [1, 0]
    v3 = [0.5, h]
    m1 = [0.5, 0]
    m2 = [0.75, h/2]
    m3 = [0.25, h/2]
    cx, cy = 0.5, h/3
    configs.append([v1, v2, v3, m1, m2, m3, [cx, cy],
                    [0.5, h/6], [0.5, h/2], [0.25, h/3], [0.75, h/3]])
    
    # Random configurations
    for _ in range(n_configs - 1):
        bary = _random_bary(rng)
        cart = _bary_to_cart(bary)
        configs.append(cart)
    
    # Evaluate all initial configs
    for cfg in configs:
        pts = np.array(cfg, dtype=float)
        if len(pts) != N:
            continue
        pts = _project_cartesian(pts)
        score = _soft_min_area_cart(pts, temperature=0.001)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Refine with optimization in Cartesian space
    for cfg_idx, cfg in enumerate(configs):
        if time.time() > deadline - 10:
            break
        
        pts = np.array(cfg, dtype=float)
        if len(pts) != N:
            continue
        
        flat = pts.flatten()
        
        # Try optimization with different temperatures
        for temp in [0.05, 0.02, 0.01, 0.005, 0.002]:
            if time.time() > deadline - 10:
                break
            
            try:
                result = minimize(
                    _cartesian_objective,
                    flat,
                    args=(temp,),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                opt_pts = result.x.reshape(N, 2)
                opt_pts = _project_cartesian(opt_pts)
                score = _soft_min_area_cart(opt_pts, temperature=0.001)
                if score > best_score:
                    best_score = score
                    best_points = opt_pts.copy()
            except Exception:
                pass
    
    # Final polish: use scipy with tight tolerances on the best solution
    if best_points is not None and time.time() < deadline - 15:
        flat = best_points.flatten()
        for temp in [0.001, 0.0005, 0.0002]:
            if time.time() > deadline - 10:
                break
            try:
                result = minimize(
                    _cartesian_objective,
                    flat,
                    args=(temp,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                opt_pts = result.x.reshape(N, 2)
                opt_pts = _project_cartesian(opt_pts)
                score = _soft_min_area_cart(opt_pts, temperature=0.001)
                if score > best_score:
                    best_score = score
                    best_points = opt_pts.copy()
                    flat = best_points.flatten()
            except Exception:
                pass
    
    if best_points is None:
        # Fallback: return a valid configuration
        h = np.sqrt(3) / 2
        best_points = np.array([[0.5, h/3]] * N)
    
    # Final validation and projection
    best_points = _project_cartesian(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
