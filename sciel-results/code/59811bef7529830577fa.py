"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

DEADLINE_MARGIN = 140.0

_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
NUM_TRIP = len(_TRIPLETS)

_VERTS = np.array([[0, 0], [1, 0], [0.5, S3H]])

def _bary_to_cart(bary):
    return bary[:, 0:1] * _VERTS[0] + bary[:, 1:2] * _VERTS[1] + bary[:, 2:3] * _VERTS[2]

def _cart_to_bary(cart):
    x = cart[:, 0]
    y = cart[:, 1]
    b2 = y / S3
    b1 = x - 0.5 * b2
    b0 = 1 - b1 - b2
    return np.column_stack([b0, b1, b2])

def _vectorized_min_area_cart(points):
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _soft_min_area_cart(points, alpha=100):
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    min_a = np.min(areas)
    return min_a - (1.0 / alpha) * np.log(np.sum(np.exp(-alpha * (areas - min_a))))

def _project_cartesian(points):
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    y = np.maximum(0, y)
    y = np.minimum(y, S3 * x)
    y = np.minimum(y, S3 - S3 * x)
    x = np.maximum(0, x)
    x = np.minimum(1, x)
    y = np.maximum(0, y)
    y = np.minimum(y, S3 * x)
    y = np.minimum(y, S3 - S3 * x)
    return np.column_stack([x, y])

def _is_valid_cart(points):
    x = points[:, 0]
    y = points[:, 1]
    tol = 1e-9
    return (np.all(y >= -tol) and 
            np.all(y <= S3 * x + tol) and 
            np.all(S3 * x + y <= S3 + tol))

def _random_bary():
    r = np.random.random((N, 3))
    r /= r.sum(axis=1, keepdims=True)
    return r

def _slp_objective(flat_cart):
    """
    SLP (Sequential Linear Programming) style objective.
    Uses soft-min with adaptive alpha based on current min area.
    """
    cart = flat_cart.reshape(N, 2)
    cart = _project_cartesian(cart)
    
    areas = np.empty(NUM_TRIP)
    p1 = cart[_T_I]
    p2 = cart[_T_J]
    p3 = cart[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_a = np.min(areas)
    
    # Adaptive alpha: larger when min area is small (more sensitive), 
    # but capped to avoid numerical issues
    alpha = max(20.0, min(500.0, 10.0 / max(min_a, 1e-6)))
    
    soft_min = min_a - (1.0 / alpha) * np.log(np.sum(np.exp(-alpha * (areas - min_a))))
    
    # Boundary penalty
    x = cart[:, 0]
    y = cart[:, 1]
    pen1 = np.sum(np.maximum(0, -y)**2)
    pen2 = np.sum(np.maximum(0, y - S3 * x)**2)
    pen3 = np.sum(np.maximum(0, S3 * x + y - S3)**2)
    pen = pen1 + pen2 + pen3
    
    return -soft_min + 50.0 * pen

def _symmetric_initial_config():
    """
    Create a symmetric initial configuration that respects the 
    equilateral triangle's 3-fold symmetry.
    
    Layout: 
    - 3 points near the midpoints of each edge
    - 2 points in each of the 3 "corner" regions  
    - 1 point at the centroid
    Total: 3 + 6 + 1 = 10... need 11
    
    Let me use:
    - 1 centroid
    - 3 points at 1/3 along each edge from vertex
    - 3 points at 2/3 along each edge from vertex  
    - 3 points forming a smaller equilateral triangle around centroid
    - 1 point... 
    
    Actually, let me use a known good starting point:
    A hexagonal-like pattern with 11 points.
    """
    # Start with a rough grid and apply symmetry
    # Use barycentric coordinates for symmetry
    
    # 11 points in barycentric coords
    # Group them into orbits under 3-fold rotation
    
    # Orbit 1: centroid (1 point) - (1/3, 1/3, 1/3)
    # Orbit 2: 3 points at equal distance from centroid, rotated 120 degrees
    # Orbit 3: 3 points at another radius, rotated 120 degrees  
    # Orbit 4: 3 points at another radius
    # Orbit 5: 1 point? No, that breaks symmetry.
    
    # Let me use: 1 + 3 + 3 + 3 + 1 = 11? No.
    # 1 + 3 + 3 + 4 = 11? 
    # Or: 3 + 3 + 3 + 2 = 11?
    
    # Simplest: use a 3x3 grid plus 2 extra points
    # But that's not symmetric.
    
    # Let me just use a reasonable asymmetric starting point
    # and let SLP find the optimum.
    
    # Start with points roughly evenly spaced
    cart = np.zeros((N, 2))
    
    # Bottom row: 4 points
    cart[0] = [0.05, 0.0]
    cart[1] = [0.35, 0.0]
    cart[2] = [0.65, 0.0]
    cart[3] = [0.95, 0.0]
    
    # Middle row: 4 points
    cart[4] = [0.2, S3H * 0.33]
    cart[5] = [0.4, S3H * 0.33]
    cart[6] = [0.6, S3H * 0.33]
    cart[7] = [0.8, S3H * 0.33]
    
    # Top row: 3 points
    cart[8] = [0.3, S3H * 0.67]
    cart[9] = [0.5, S3H * 0.67]
    cart[10] = [0.7, S3H * 0.67]
    
    return cart

def _slp_optimize(x0, max_iters=200, tol=1e-10):
    """
    SLP optimization with adaptive alpha.
    x0: initial point in Cartesian coordinates (N, 2)
    """
    current = x0.copy()
    current_obj = _slp_objective(current.flatten())
    
    for it in range(max_iters):
        # Get current min area for adaptive alpha
        cart = _project_cartesian(current)
        min_a = _vectorized_min_area_cart(cart)
        
        # Define objective with current alpha
        alpha = max(20.0, min(500.0, 10.0 / max(min_a, 1e-6)))
        
        def obj(flat):
            c = _project_cartesian(flat.reshape(N, 2))
            p1 = c[_T_I]
            p2 = c[_T_J]
            p3 = c[_T_K]
            v1 = p2 - p1
            v2 = p3 - p1
            cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            areas = 0.5 * np.abs(cross)
            ma = np.min(areas)
            sm = ma - (1.0 / alpha) * np.log(np.sum(np.exp(-alpha * (areas - ma))))
            x = c[:, 0]
            y = c[:, 1]
            pen = np.sum(np.maximum(0, -y)**2) + np.sum(np.maximum(0, y - S3*x)**2) + np.sum(np.maximum(0, S3*x + y - S3)**2)
            return -sm + 50.0 * pen
        
        res = minimize(obj, current.flatten(), method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-9})
        
        new_point = res.x.reshape(N, 2)
        new_point = _project_cartesian(new_point)
        new_obj = _slp_objective(new_point.flatten())
        
        if new_obj < current_obj - tol:
            current = new_point
            current_obj = new_obj
        else:
            break
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    np.random.seed(42)
    
    # Initial symmetric configuration
    best_cart = _symmetric_initial_config()
    best_cart = _project_cartesian(best_cart)
    best_score = _vectorized_min_area_cart(best_cart)
    
    # SLP optimization from symmetric start
    if time.time() < deadline - 10:
        result = _slp_optimize(best_cart, max_iters=300)
        result = _project_cartesian(result)
        score = _vectorized_min_area_cart(result)
        if score > best_score:
            best_score = score
            best_cart = result
    
    # Try multiple random restarts with SLP
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random start with some structure
        bary = _random_bary()
        cart = _bary_to_cart(bary)
        cart = _project_cartesian(cart)
        
        result = _slp_optimize(cart, max_iters=100)
        result = _project_cartesian(result)
        score = _vectorized_min_area_cart(result)
        if score > best_score:
            best_score = score
            best_cart = result
    
    # Final polish with high-precision SLP
    if time.time() < deadline - 3:
        result = _slp_optimize(best_cart, max_iters=500, tol=1e-12)
        result = _project_cartesian(result)
        score = _vectorized_min_area_cart(result)
        if score > best_score:
            best_score = score
            best_cart = result
    
    best_cart = _project_cartesian(best_cart)
    return best_cart
# EVOLVE-BLOCK-END
