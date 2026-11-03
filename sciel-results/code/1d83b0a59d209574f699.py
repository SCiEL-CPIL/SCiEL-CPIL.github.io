"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of equilateral triangle with side 1

# Safety margin from 150s limit
DEADLINE_MARGIN = 135.0

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
NUM_TRIP = len(_TRIPLETS)  # 165

# Precompute edge vectors for faster area calculation
# For each triplet (i,j,k), we need: (p_j - p_i) x (p_k - p_i)
# This equals: p_i x p_j - p_i x p_k + p_j x p_k
# So we can precompute coefficients
_T_COEFF_I = np.zeros((NUM_TRIP, 2))
_T_COEFF_J = np.zeros((NUM_TRIP, 2))
_T_COEFF_K = np.zeros((NUM_TRIP, 2))
for idx, (i, j, k) in enumerate(_TRIPLETS):
    # cross = (p_j - p_i) x (p_k - p_i) = p_j x p_k - p_j x p_i - p_i x p_k + p_i x p_i
    # = p_j x p_k - p_j x p_i - p_i x p_k
    # In 2D: (a_x*b_y - a_y*b_x)
    # cross = (p_jx*p_ky - p_jy*p_kx) - (p_jx*p_iy - p_jy*p_ix) - (p_ix*p_ky - p_iy*p_kx)
    # = p_jx*p_ky - p_jy*p_kx - p_jx*p_iy + p_jy*p_ix - p_ix*p_ky + p_iy*p_kx
    # Group by point:
    # p_i: (-p_jy)*p_ix + (p_jx - p_kx)*p_iy... let me just use the direct formula
    pass

# Triangle vertices in Cartesian
_VERTS = np.array([[0, 0], [1, 0], [0.5, S3H]])

def _bary_to_cart(bary):
    """Convert barycentric coordinates (N, 3) to Cartesian (N, 2)."""
    return bary[:, 0:1] * _VERTS[0] + bary[:, 1:2] * _VERTS[1] + bary[:, 2:3] * _VERTS[2]

def _cart_to_bary(cart):
    """Convert Cartesian (N, 2) to barycentric coordinates (N, 3)."""
    # Solve: cart = b0*V0 + b1*V1 + b2*V2, b0+b1+b2=1
    # Using the inverse of the transformation
    x = cart[:, 0]
    y = cart[:, 1]
    # b2 = y / S3
    b2 = y / S3
    # b1 = x - 0.5*b2
    b1 = x - 0.5 * b2
    # b0 = 1 - b1 - b2
    b0 = 1 - b1 - b2
    return np.column_stack([b0, b1, b2])

def _vectorized_min_area_cart(points):
    """Calculate the minimum area of all triangles formed by the points in Cartesian."""
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _vectorized_all_areas_cart(points):
    """Calculate all triangle areas for the points in Cartesian."""
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _soft_min_area_cart(points, alpha=100):
    """Calculate soft minimum of triangle areas using log-sum-exp."""
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    # Soft min: -1/alpha * log(sum(exp(-alpha * areas)))
    min_a = np.min(areas)
    return min_a - (1.0 / alpha) * np.log(np.sum(np.exp(-alpha * (areas - min_a))))

def _epigraph_objective(flat_params, alpha=50):
    """
    Epigraph formulation: maximize t such that all triangle areas >= t.
    params = [x1,y1,...,x11,y11,t]
    """
    n_params = flat_params.shape[0]
    t = flat_params[-1]
    cart = flat_params[:-1].reshape(N, 2)
    cart = _project_cartesian(cart)
    
    # Soft constraint: -log(sum(exp(alpha*(t - area_i)))) for each triangle
    areas = _vectorized_all_areas_cart(cart)
    # We want t <= min(areas), so penalty for t > areas
    diff = t - areas
    # Soft max of (t - areas): 1/alpha * log(sum(exp(alpha * (t - areas))))
    max_diff = np.max(diff)
    if max_diff > 1e10:
        return max_diff * 1e6
    soft_violation = (1.0 / alpha) * np.log(np.sum(np.exp(alpha * (diff - max_diff)))) + max_diff
    
    # Penalize violation of t <= min(areas)
    violation = max(0, soft_violation)
    
    # Also add small penalty for t being too negative (we want to maximize t)
    return -t + 1000 * violation

def _project_cartesian(points):
    """Project points to be inside the triangle."""
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    
    # Clip to triangle boundaries
    # y >= 0
    y = np.maximum(0, y)
    # y <= sqrt(3) * x
    y = np.minimum(y, S3 * x)
    # sqrt(3) * x + y <= sqrt(3)
    y = np.minimum(y, S3 - S3 * x)
    
    # Re-clip x after y adjustments
    x = np.maximum(0, x)
    x = np.minimum(1, x)
    # y >= 0
    y = np.maximum(0, y)
    # y <= sqrt(3) * x
    y = np.minimum(y, S3 * x)
    # sqrt(3) * x + y <= sqrt(3)
    y = np.minimum(y, S3 - S3 * x)
    
    return np.column_stack([x, y])

def _is_valid_cart(points):
    """Check if all points are inside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    tol = 1e-9
    return (np.all(y >= -tol) and 
            np.all(y <= S3 * x + tol) and 
            np.all(S3 * x + y <= S3 + tol))

def _random_bary():
    """Generate random barycentric coordinates."""
    r = np.random.random((N, 3))
    r /= r.sum(axis=1, keepdims=True)
    return r

def _bary_objective(flat_bary, alpha=100):
    """Objective function in barycentric coordinates."""
    bary = flat_bary.reshape(N, 3)
    # Ensure valid barycentric (non-negative, sum to 1)
    bary = np.maximum(0, bary)
    bary /= bary.sum(axis=1, keepdims=True)
    
    cart = _bary_to_cart(bary)
    cart = _project_cartesian(cart)
    
    # Convert back to barycentric for consistency
    bary_new = _cart_to_bary(cart)
    
    # Soft minimum area
    soft_min = _soft_min_area_cart(cart, alpha)
    
    # Penalty for points outside triangle (should be small after projection)
    x = cart[:, 0]
    y = cart[:, 1]
    pen1 = np.sum(np.maximum(0, -y)**2)
    pen2 = np.sum(np.maximum(0, y - S3 * x)**2)
    pen3 = np.sum(np.maximum(0, S3 * x + y - S3)**2)
    pen = pen1 + pen2 + pen3
    
    return -soft_min + 100 * pen

def _cartesian_epigraph_objective(flat_params, alpha=50):
    """
    Cartesian epigraph objective.
    flat_params = [x1,y1,...,x11,y11,t]
    """
    cart = flat_params[:-1].reshape(N, 2)
    t = flat_params[-1]
    cart = _project_cartesian(cart)
    
    areas = _vectorized_all_areas_cart(cart)
    diff = t - areas
    max_diff = np.max(diff)
    if max_diff > 1e10:
        return max_diff * 1e6
    soft_violation = (1.0 / alpha) * np.log(np.sum(np.exp(alpha * (diff - max_diff)))) + max_diff
    violation = max(0, soft_violation)
    return -t + 1000 * violation

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    np.random.seed(42)
    
    # Initial best
    best_bary = _random_bary()
    best_cart = _bary_to_cart(best_bary)
    best_cart = _project_cartesian(best_cart)
    best_score = _vectorized_min_area_cart(best_cart)
    
    # Phase 1: Annealing with soft-min in barycentric space
    alphas = [10, 50, 100, 200, 500, 1000]
    
    for alpha in alphas:
        if time.time() > deadline:
            break
        
        # Multiple restarts at each alpha level
        num_restarts = 8
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Start from current best or random
            if r == 0:
                x0 = best_bary.flatten()
            else:
                x0 = _random_bary().flatten()
            
            # Optimize with Nelder-Mead in barycentric space
            res = minimize(_bary_objective, x0, method='Nelder-Mead',
                          args=(alpha,),
                          options={'maxiter': 1500, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < _bary_objective(best_bary.flatten(), alpha):
                new_bary = res.x.reshape(N, 3)
                new_bary = np.maximum(0, new_bary)
                new_bary /= new_bary.sum(axis=1, keepdims=True)
                
                new_cart = _bary_to_cart(new_bary)
                new_cart = _project_cartesian(new_cart)
                
                new_score = _vectorized_min_area_cart(new_cart)
                if new_score > best_score:
                    best_score = new_score
                    best_bary = new_bary
                    best_cart = new_cart
    
    # Phase 2: Epigraph refinement in Cartesian space
    if time.time() < deadline - 20:
        # Initialize t with current best score
        t0 = best_score * 0.99
        x0 = np.concatenate([best_cart.flatten(), [t0]])
        
        # Try different alpha values for the epigraph formulation
        epigraph_alphas = [20, 50, 100]
        for e_alpha in epigraph_alphas:
            if time.time() > deadline:
                break
            
            res = minimize(_cartesian_epigraph_objective, x0, method='Nelder-Mead',
                          args=(e_alpha,),
                          options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-7})
            
            if res.fun < _cartesian_epigraph_objective(x0, e_alpha):
                new_params = res.x
                new_cart = new_params[:-1].reshape(N, 2)
                new_t = new_params[-1]
                new_cart = _project_cartesian(new_cart)
                
                new_score = _vectorized_min_area_cart(new_cart)
                if new_score > best_score:
                    best_score = new_score
                    best_cart = new_cart
                    x0 = np.concatenate([new_cart.flatten(), [new_score * 0.99]])
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5:
        # Use the hard minimum with very tight tolerances
        def cart_objective_tight(flat_cart):
            cart = flat_cart.reshape(N, 2)
            cart = _project_cartesian(cart)
            area = _vectorized_min_area_cart(cart)
            return -area
        
        res = minimize(cart_objective_tight, best_cart.flatten(), method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-9})
        
        if res.fun < -best_score:
            final_cart = res.x.reshape(N, 2)
            final_cart = _project_cartesian(final_cart)
            final_score = _vectorized_min_area_cart(final_cart)
            if final_score > best_score:
                best_cart = final_cart
                best_score = final_score
    
    return best_cart
# EVOLVE-BLOCK-END
