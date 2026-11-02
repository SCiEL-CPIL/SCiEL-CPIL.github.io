"""Heilbronn triangles: barycentric soft-min annealing with multi-start."""
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

def _soft_min_area_cart(points, alpha=100):
    """Smooth approximation of the minimum area."""
    areas = _all_areas_cart(points)
    min_a = np.min(areas)
    shifted = areas - min_a
    # Use log-sum-exp trick for numerical stability
    m = np.min(shifted)
    log_sum = m + np.log(np.sum(np.exp(shifted - m)))
    soft_min = min_a - (1.0 / alpha) * log_sum
    return soft_min

def _project_bary(bary):
    """Project barycentric coordinates to be inside the triangle (non-negative, sum to 1)."""
    bary = np.maximum(0, bary)
    sums = bary.sum(axis=1, keepdims=True)
    sums = np.maximum(sums, 1e-15)
    bary = bary / sums
    return bary

def _is_valid_bary(bary):
    """Check if barycentric coordinates are valid (non-negative, sum to 1)."""
    return np.all(bary >= -1e-9) and np.all(np.abs(bary.sum(axis=1) - 1) < 1e-6)

def _random_bary():
    """Generate random barycentric coordinates."""
    r = np.random.random((N, 3))
    r /= r.sum(axis=1, keepdims=True)
    return r

def _symmetric_bary():
    """Create a symmetric initial configuration in barycentric coordinates."""
    bary = np.zeros((N, 3))
    # 3 corners
    bary[0] = [1, 0, 0]
    bary[1] = [0, 1, 0]
    bary[2] = [0, 0, 1]
    # 3 side midpoints
    bary[3] = [0.5, 0.5, 0]
    bary[4] = [0.5, 0, 0.5]
    bary[5] = [0, 0.5, 0.5]
    # 5 interior points in a smaller triangle pattern
    cx, cy, cz = 1/3, 1/3, 1/3
    r = 0.15
    for i in range(6, N):
        angle = 2 * np.pi * (i - 6) / 5
        bary[i] = [cx + r * np.cos(angle) * 0.7, 
                   cy + r * np.cos(angle + 2*np.pi/3) * 0.7,
                   cz + r * np.cos(angle + 4*np.pi/3) * 0.7]
    bary = _project_bary(bary)
    return bary

def _bary_objective(flat_bary, alpha=100):
    """Objective in barycentric coordinates: minimize -soft_min_area."""
    bary = flat_bary.reshape(N, 3)
    bary = _project_bary(bary)
    cart = _bary_to_cart(bary)
    soft_min = _soft_min_area_cart(cart, alpha)
    return -soft_min

def _bary_gradient(flat_bary, alpha=100):
    """Numerical gradient of the barycentric objective."""
    bary = flat_bary.reshape(N, 3)
    bary = _project_bary(bary)
    cart = _bary_to_cart(bary)
    areas = _all_areas_cart(cart)
    min_a = np.min(areas)
    
    # Find the active constraints (minimum area triangles)
    threshold = min_a + 1e-4
    active = areas <= threshold
    
    # Compute gradient numerically for each point
    grad = np.zeros_like(flat_bary)
    eps = 1e-5
    
    for idx in range(N):
        for dim in range(3):
            bary_plus = bary.copy()
            bary_plus[idx, dim] += eps
            bary_plus = _project_bary(bary_plus)
            cart_plus = _bary_to_cart(bary_plus)
            areas_plus = _all_areas_cart(cart_plus)
            min_plus = np.min(areas_plus)
            
            bary_minus = bary.copy()
            bary_minus[idx, dim] -= eps
            bary_minus = _project_bary(bary_minus)
            cart_minus = _bary_to_cart(bary_minus)
            areas_minus = _all_areas_cart(cart_minus)
            min_minus = np.min(areas_minus)
            
            # Gradient of -min_area
            grad[idx * 3 + dim] = -(min_plus - min_minus) / (2 * eps)
    
    return grad

def _anneal_bary(bary_init, alpha_start=20, alpha_end=500, n_iter=200, time_limit=10):
    """Anneal the alpha parameter in soft-min optimization."""
    best_bary = bary_init.copy()
    best_cart = _bary_to_cart(best_bary)
    best_score = _vectorized_min_area_cart(best_cart)
    
    start_time = time.time()
    
    for i in range(n_iter):
        if time.time() - start_time > time_limit:
            break
        
        alpha = alpha_start * (alpha_end / alpha_start) ** (i / n_iter)
        
        # Use L-BFGS-B to minimize the soft-min objective
        res = minimize(_bary_objective, best_bary.flatten(), 
                      args=(alpha,),
                      method='L-BFGS-B',
                      options={'maxiter': 50, 'gtol': 1e-4})
        
        new_bary = res.x.reshape(N, 3)
        new_bary = _project_bary(new_bary)
        new_cart = _bary_to_cart(new_bary)
        new_score = _vectorized_min_area_cart(new_cart)
        
        if new_score > best_score:
            best_score = new_score
            best_bary = new_bary.copy()
            best_cart = new_cart.copy()
    
    return best_bary, best_cart, best_score

def _multi_start_optimize(n_starts=8, time_per_start=15):
    """Multi-start optimization with diverse initializations."""
    best_bary = None
    best_cart = None
    best_score = -np.inf
    
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    for s in range(n_starts):
        if time.time() > deadline - 10:
            break
        
        # Different initialization strategies
        if s == 0:
            bary_init = _symmetric_bary()
        elif s == 1:
            bary_init = _random_bary()
        elif s == 2:
            # Grid-like pattern
            bary_init = np.zeros((N, 3))
            # Distribute points in a 3x3 grid in barycentric space, take 9, plus 2 more
            grid_pts = []
            for i in range(3):
                for j in range(3):
                    b0 = 0.1 + 0.3 * i
                    b1 = 0.1 + 0.3 * j
                    b2 = 1 - b0 - b1
                    if b2 > 0:
                        grid_pts.append([b0, b1, b2])
            grid_pts = grid_pts[:N]
            for i, pt in enumerate(grid_pts):
                bary_init[i] = pt
            bary_init = _project_bary(bary_init)
        elif s == 3:
            # Perturbed symmetric
            bary_init = _symmetric_bary() + np.random.randn(N, 3) * 0.05
            bary_init = _project_bary(bary_init)
        else:
            # Random with some structure
            bary_init = _random_bary()
            # Add some symmetry by averaging with a permuted version
            perm = np.random.permutation(3)
            bary_perm = bary_init[:, perm]
            bary_init = 0.5 * (bary_init + bary_perm)
            bary_init = _project_bary(bary_init)
        
        # Anneal from this start
        bary_res, cart_res, score_res = _anneal_bary(bary_init, 
                                                     alpha_start=20, 
                                                     alpha_end=500,
                                                     n_iter=100,
                                                     time_limit=time_per_start)
        
        if score_res > best_score:
            best_score = score_res
            best_bary = bary_res.copy()
            best_cart = cart_res.copy()
    
    return best_bary, best_cart, best_score

def _refine_bary(bary, n_iter=500, step_size=0.01):
    """Hill climbing refinement in barycentric coordinates."""
    best_bary = bary.copy()
    best_cart = _bary_to_cart(best_bary)
    best_score = _vectorized_min_area_cart(best_cart)
    
    for i in range(n_iter):
        new_bary = best_bary + np.random.randn(N, 3) * step_size
        new_bary = _project_bary(new_bary)
        new_cart = _bary_to_cart(new_bary)
        new_score = _vectorized_min_area_cart(new_cart)
        
        if new_score > best_score:
            best_score = new_score
            best_bary = new_bary.copy()
            best_cart = new_cart.copy()
            step_size *= 0.98
        else:
            step_size = min(step_size * 1.01, 0.03)
    
    return best_bary, best_cart, best_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    np.random.seed(42)
    
    # Phase 1: Multi-start optimization
    best_bary, best_cart, best_score = _multi_start_optimize(n_starts=6, time_per_start=18)
    
    # Phase 2: Refinement with hill climbing
    if time.time() < deadline - 15:
        ref_bary, ref_cart, ref_score = _refine_bary(best_bary, n_iter=1000, step_size=0.005)
        if ref_score > best_score:
            best_score = ref_score
            best_bary = ref_bary.copy()
            best_cart = ref_cart.copy()
    
    # Phase 3: Another round of annealing from the best
    if time.time() < deadline - 10:
        bary_res, cart_res, score_res = _anneal_bary(best_bary,
                                                     alpha_start=100,
                                                     alpha_end=1000,
                                                     n_iter=50,
                                                     time_limit=8)
        if score_res > best_score:
            best_score = score_res
            best_bary = bary_res.copy()
            best_cart = cart_res.copy()
    
    # Phase 4: Final fine-tuning
    if time.time() < deadline - 5:
        ref_bary, ref_cart, ref_score = _refine_bary(best_bary, n_iter=500, step_size=0.002)
        if ref_score > best_score:
            best_score = ref_score
            best_bary = ref_bary.copy()
            best_cart = ref_cart.copy()
    
    # Ensure final result is valid
    best_cart = _bary_to_cart(_project_bary(best_bary))
    
    return best_cart
# EVOLVE-BLOCK-END
