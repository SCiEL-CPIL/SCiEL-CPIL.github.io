"""Rewritten program for Heilbronn triangles with Lagrangian relaxation and subgradient optimization."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of equilateral triangle with side 1

DEADLINE_MARGIN = 140.0

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
NUM_TRIP = len(_TRIPLETS)  # 165

# Triangle vertices in Cartesian
_VERTS = np.array([[0, 0], [1, 0], [0.5, S3H]])

def _bary_to_cart(bary):
    """Convert barycentric coordinates (N, 3) to Cartesian (N, 2)."""
    return bary[:, 0:1] * _VERTS[0] + bary[:, 1:2] * _VERTS[1] + bary[:, 2:3] * _VERTS[2]

def _cart_to_bary(cart):
    """Convert Cartesian (N, 2) to barycentric coordinates (N, 3)."""
    x = cart[:, 0]
    y = cart[:, 1]
    b2 = y / S3
    b1 = x - 0.5 * b2
    b0 = 1 - b1 - b2
    return np.column_stack([b0, b1, b2])

def _all_areas_cart(points):
    """Calculate all triangle areas for 165 triplets."""
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _vectorized_min_area_cart(points):
    """Calculate the minimum area of all triangles formed by the points in Cartesian."""
    return np.min(_all_areas_cart(points))

def _project_cartesian(points):
    """Project points to be inside the triangle."""
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    
    for _ in range(3):
        y = np.maximum(0, y)
        y = np.minimum(y, S3 * x)
        y = np.minimum(y, S3 - S3 * x)
        x = np.maximum(0, x)
        x = np.minimum(1, x)
    
    return np.column_stack([x, y])

def _is_valid_cart(points):
    """Check if all points are inside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    tol = 1e-9
    return (np.all(y >= -tol) and 
            np.all(y <= S3 * x + tol) and 
            np.all(S3 * x + y <= S3 + tol))

def _symmetric_initial():
    """Create a symmetric initial configuration for 11 points."""
    # Place points in a structured pattern:
    # 3 corners, 3 midpoints of sides, and 5 interior points
    pts = np.zeros((N, 2))
    
    # Corners
    pts[0] = [0.0, 0.0]
    pts[1] = [1.0, 0.0]
    pts[2] = [0.5, S3H]
    
    # Midpoints of sides
    pts[3] = [0.5, 0.0]
    pts[4] = [0.75, S3H * 0.5]
    pts[5] = [0.25, S3H * 0.5]
    
    # Interior points - arranged in a smaller triangle
    cx, cy = 0.5, S3H / 3
    r = 0.15
    for i in range(3, 8):
        angle = 2 * np.pi * (i - 3) / 5
        pts[i] = [cx + r * np.cos(angle), cy + r * np.sin(angle) * 0.577]
    
    return _project_cartesian(pts)

def _random_bary():
    """Generate random barycentric coordinates."""
    r = np.random.random((N, 3))
    r /= r.sum(axis=1, keepdims=True)
    return r

def _lagrangian_objective(flat_cart, multipliers, t):
    """
    Lagrangian objective: minimize -min_area + sum(multipliers * (t - area_i))
    subject to points in triangle.
    
    We maximize min(area_i) by solving:
    max_t t s.t. area_i >= t for all i
    Lagrangian: L = t + sum(m_i * (area_i - t)) = (1 - sum(m_i)) * t + sum(m_i * area_i)
    
    For fixed multipliers, we minimize over points:
    -sum(m_i * area_i) + penalty for being outside triangle
    
    Actually, let's use a different formulation. We want to maximize the minimum area.
    Let f(x) = min_i area_i(x). We want to maximize f.
    
    Subgradient method: at each step, find the active constraint (minimum area),
    and move points to increase that area.
    
    Alternative: use a penalty method with adaptive weights.
    """
    cart = flat_cart.reshape(N, 2)
    cart = _project_cartesian(cart)
    
    areas = _all_areas_cart(cart)
    min_a = np.min(areas)
    
    # Soft penalty: weighted sum of (min_a - area_i) for areas below min_a
    # This is equivalent to maximizing a weighted sum that emphasizes the minimum
    deficit = np.maximum(0, min_a - areas)
    penalty = np.sum(multipliers * deficit)
    
    # Boundary penalty
    x = cart[:, 0]
    y = cart[:, 1]
    pen1 = np.sum(np.maximum(0, -y)**2)
    pen2 = np.sum(np.maximum(0, y - S3 * x)**2)
    pen3 = np.sum(np.maximum(0, S3 * x + y - S3)**2)
    bpen = pen1 + pen2 + pen3
    
    return -min_a + penalty + 100 * bpen

def _subgradient_step(cart, step_size=0.01):
    """
    One subgradient step: identify the minimum area triangle and move its vertices
    to increase that area.
    """
    areas = _all_areas_cart(cart)
    min_idx = np.argmin(areas)
    min_a = areas[min_idx]
    
    # Get the three points forming the minimum area triangle
    i, j, k = _TRIPLETS[min_idx]
    
    # Compute the gradient of area with respect to each point
    # Area = 0.5 * |cross(P2-P1, P3-P1)|
    # For fixed P2, P3, gradient wrt P1:
    # d/dP1 (0.5 * |cross(P2-P1, P3-P1)|)
    
    P1, P2, P3 = cart[i], cart[j], cart[k]
    
    v1 = P2 - P1
    v2 = P3 - P1
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    
    # Gradient of 0.5 * cross wrt P1:
    # cross = (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
    # d(cross)/dx1 = -(y3-y1) + (y2-y1) = y2 - y3
    # d(cross)/dy1 = (x2-x1) - (x3-x1) = x2 - x3
    
    # We want to increase the area, so we move against the gradient of -area
    # Gradient of area wrt P1:
    grad_P1 = np.array([-(y2 - y3), (x2 - x3)])  # This is for cross, need to handle abs
    
    # Actually, let's use a simpler approach: move P1, P2, P3 slightly to increase area
    # For a triangle, increasing area means moving vertices apart
    
    # Move P1 away from the centroid of P2, P3
    centroid_23 = (P2 + P3) / 2
    dir_P1 = P1 - centroid_23
    norm = np.linalg.norm(dir_P1)
    if norm > 1e-10:
        dir_P1 /= norm
    
    # Move P2 away from centroid of P1, P3
    centroid_13 = (P1 + P3) / 2
    dir_P2 = P2 - centroid_13
    norm = np.linalg.norm(dir_P2)
    if norm > 1e-10:
        dir_P2 /= norm
    
    # Move P3 away from centroid of P1, P2
    centroid_12 = (P1 + P2) / 2
    dir_P3 = P3 - centroid_12
    norm = np.linalg.norm(dir_P3)
    if norm > 1e-10:
        dir_P3 /= norm
    
    # Apply small moves
    new_cart = cart.copy()
    new_cart[i] += step_size * dir_P1
    new_cart[j] += step_size * dir_P2
    new_cart[k] += step_size * dir_P3
    
    new_cart = _project_cartesian(new_cart)
    
    return new_cart

def _lagrangian_optimize(cart, max_iter=500, step_size_init=0.02):
    """
    Lagrangian relaxation with subgradient optimization.
    """
    best_cart = cart.copy()
    best_score = _vectorized_min_area_cart(best_cart)
    
    # Initialize multipliers: give higher weight to constraints that are more active
    multipliers = np.ones(NUM_TRIP) / NUM_TRIP
    
    step_size = step_size_init
    
    for it in range(max_iter):
        # Primal step: minimize the Lagrangian with respect to points
        def objective(flat):
            c = flat.reshape(N, 2)
            c = _project_cartesian(c)
            areas = _all_areas_cart(c)
            
            # Weighted sum: emphasize smaller areas
            w = multipliers / np.sum(multipliers)
            obj = -np.sum(w * areas)
            
            # Boundary penalty
            x = c[:, 0]
            y = c[:, 1]
            pen = (np.sum(np.maximum(0, -y)**2) + 
                   np.sum(np.maximum(0, y - S3 * x)**2) + 
                   np.sum(np.maximum(0, S3 * x + y - S3)**2))
            return obj + 100 * pen
        
        res = minimize(objective, best_cart.flatten(), method='L-BFGS-B',
                      options={'maxiter': 200, 'gtol': 1e-5})
        
        new_cart = res.x.reshape(N, 2)
        new_cart = _project_cartesian(new_cart)
        
        # Evaluate
        areas = _all_areas_cart(new_cart)
        new_score = np.min(areas)
        
        if new_score > best_score:
            best_score = new_score
            best_cart = new_cart.copy()
        
        # Dual step: update multipliers based on which constraints are active
        # Increase weight for constraints that are close to the minimum
        deficit = np.maximum(0, np.max(areas) - areas)
        multipliers = np.maximum(0, multipliers + step_size * deficit / (np.max(deficit) + 1e-10))
        multipliers /= np.sum(multipliers)
        
        # Decay step size
        step_size *= 0.95
    
    return best_cart, best_score

def _soft_min_objective(flat_cart, alpha=100):
    """Soft minimum objective for smooth optimization."""
    cart = flat_cart.reshape(N, 2)
    cart = _project_cartesian(cart)
    
    areas = _all_areas_cart(cart)
    min_a = np.min(areas)
    soft_min = min_a - (1.0 / alpha) * np.log(np.sum(np.exp(-alpha * (areas - min_a))))
    
    x = cart[:, 0]
    y = cart[:, 1]
    pen1 = np.sum(np.maximum(0, -y)**2)
    pen2 = np.sum(np.maximum(0, y - S3 * x)**2)
    pen3 = np.sum(np.maximum(0, S3 * x + y - S3)**2)
    pen = pen1 + pen2 + pen3
    
    return -soft_min + 100 * pen

def _hill_climbing(cart, n_iter=5000, step_size=0.005):
    """Hill climbing with random perturbations."""
    best_cart = cart.copy()
    best_score = _vectorized_min_area_cart(best_cart)
    
    for i in range(n_iter):
        new_cart = best_cart + np.random.randn(N, 2) * step_size
        new_cart = _project_cartesian(new_cart)
        
        if not _is_valid_cart(new_cart):
            continue
        
        new_score = _vectorized_min_area_cart(new_cart)
        
        if new_score > best_score:
            best_score = new_score
            best_cart = new_cart.copy()
            step_size *= 0.97
        else:
            step_size = min(step_size * 1.02, 0.02)
    
    return best_cart, best_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    np.random.seed(42)
    
    # Phase 1: Start with symmetric configuration
    best_cart = _symmetric_initial()
    best_score = _vectorized_min_area_cart(best_cart)
    
    # Phase 2: Lagrangian relaxation optimization
    if time.time() < deadline - 30:
        lag_cart, lag_score = _lagrangian_optimize(best_cart, max_iter=100, step_size_init=0.015)
        if lag_score > best_score:
            best_score = lag_score
            best_cart = lag_cart.copy()
    
    # Phase 3: Multiple restarts with diverse initializations
    num_restarts = 10
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        # Different initialization strategies
        if r % 3 == 0:
            # Random barycentric
            x0_bary = _random_bary()
            x0_cart = _bary_to_cart(x0_bary)
        elif r % 3 == 1:
            # Near the best solution with perturbation
            x0_cart = best_cart + np.random.randn(N, 2) * 0.05
        else:
            # Structured: spread points in a grid-like pattern
            x0_cart = np.zeros((N, 2))
            # 3 corners
            x0_cart[0] = [0.0, 0.0]
            x0_cart[1] = [1.0, 0.0]
            x0_cart[2] = [0.5, S3H]
            # 3 side midpoints
            x0_cart[3] = [0.5, 0.0]
            x0_cart[4] = [0.75, S3H * 0.5]
            x0_cart[5] = [0.25, S3H * 0.5]
            # 5 interior points
            for i in range(6, N):
                angle = 2 * np.pi * (i - 6) / 5
                x0_cart[i] = [0.5 + 0.2 * np.cos(angle), S3H/3 + 0.1 * np.sin(angle)]
        
        x0_cart = _project_cartesian(x0_cart)
        
        # Soft-min optimization with multiple alphas
        for alpha in [50, 100, 200]:
            if time.time() > deadline - 10:
                break
            res = minimize(_soft_min_objective, x0_cart.flatten(), method='L-BFGS-B',
                          args=(alpha,),
                          options={'maxiter': 300, 'gtol': 1e-6})
            
            new_cart = res.x.reshape(N, 2)
            new_cart = _project_cartesian(new_cart)
            new_score = _vectorized_min_area_cart(new_cart)
            
            if new_score > best_score:
                best_score = new_score
                best_cart = new_cart.copy()
        
        # Hill climbing refinement
        if time.time() < deadline - 10:
            hc_cart, hc_score = _hill_climbing(best_cart, n_iter=3000, step_size=0.004)
            if hc_score > best_score:
                best_score = hc_score
                best_cart = hc_cart.copy()
    
    # Phase 4: Final fine-tuning with Lagrangian
    if time.time() < deadline - 5:
        lag_cart, lag_score = _lagrangian_optimize(best_cart, max_iter=50, step_size_init=0.005)
        if lag_score > best_score:
            best_score = lag_score
            best_cart = lag_cart.copy()
    
    # Phase 5: Final hill climbing
    if time.time() < deadline - 2:
        hc_cart, hc_score = _hill_climbing(best_cart, n_iter=2000, step_size=0.002)
        if hc_score > best_score:
            best_score = hc_score
            best_cart = hc_cart.copy()
    
    # Ensure final result is valid
    best_cart = _project_cartesian(best_cart)
    
    return best_cart
# EVOLVE-BLOCK-END
