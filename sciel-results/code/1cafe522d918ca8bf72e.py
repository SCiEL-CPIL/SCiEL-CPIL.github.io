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
DEADLINE_MARGIN = 145.0

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

def _soft_min_area_cart(points, temperature):
    """Soft minimum area using log-sum-exp for smooth optimization."""
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    # Soft min: -T * log(sum(exp(-area/T))) approximates min(area) as T->0
    max_area = np.max(areas)
    exp_terms = np.exp(-(areas - max_area) / temperature)
    soft_min = max_area - temperature * np.log(np.sum(exp_terms))
    return soft_min

def _project_cartesian(points):
    """Project points to be inside the triangle."""
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
    """Check if all points are inside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    tol = 1e-9
    return (np.all(y >= -tol) and 
            np.all(y <= S3 * x + tol) and 
            np.all(S3 * x + y <= S3 + tol))

def _structured_initial_bary():
    """Generate structured initial barycentric configuration with 6-fold symmetry."""
    # 3 vertices
    v0 = np.array([1, 0, 0])
    v1 = np.array([0, 1, 0])
    v2 = np.array([0, 0, 1])
    # 3 edge midpoints
    m01 = np.array([0.5, 0.5, 0])
    m12 = np.array([0, 0.5, 0.5])
    m02 = np.array([0.5, 0, 0.5])
    # Centroid
    c = np.array([1/3, 1/3, 1/3])
    # 4 more symmetric points: 1/4 and 3/4 along edges
    q1 = np.array([0.75, 0.25, 0])
    q2 = np.array([0.25, 0.75, 0])
    q3 = np.array([0, 0.75, 0.25])
    q4 = np.array([0, 0.25, 0.75])
    
    pts = np.array([v0, v1, v2, m01, m12, m02, c, q1, q2, q3, q4])
    return pts

def _random_bary():
    """Generate random barycentric coordinates."""
    r = np.random.random((N, 3))
    r /= r.sum(axis=1, keepdims=True)
    return r

def _bary_objective(flat, temperature):
    """Objective in barycentric coords: negative soft-min area."""
    bary = flat.reshape(N, 3)
    # Normalize to ensure valid barycentric
    bary = np.maximum(bary, 0)
    bary /= bary.sum(axis=1, keepdims=True)
    cart = _bary_to_cart(bary)
    return -_soft_min_area_cart(cart, temperature)

def _bary_objective_hard(flat):
    """Objective in barycentric coords: negative hard min area."""
    bary = flat.reshape(N, 3)
    bary = np.maximum(bary, 0)
    bary /= bary.sum(axis=1, keepdims=True)
    cart = _bary_to_cart(bary)
    return -_vectorized_min_area_cart(cart)

def _local_polish_bary(bary_init, max_iter=2000, tol=1e-12):
    """High-precision polish using L-BFGS-B in barycentric coordinates with soft-min."""
    bary_flat = bary_init.flatten()
    bounds = [(0.0, 1.0)] * (N * 3)
    
    # Use a very low temperature soft-min for a smooth but sharp objective
    T_polish = 1e-5
    
    def obj_polish(flat):
        return _bary_objective(flat, T_polish)
        
    try:
        res = minimize(obj_polish, bary_flat, method='L-BFGS-B',
                      bounds=bounds, options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-12, 'maxls': 50})
        final_bary = res.x.reshape(N, 3)
        final_bary = np.maximum(final_bary, 0)
        final_bary /= final_bary.sum(axis=1, keepdims=True)
        final_cart = _bary_to_cart(final_bary)
        return final_cart, _vectorized_min_area_cart(final_cart)
    except:
        # Fallback: ensure bary_init is valid barycentric
        if bary_init.shape == (N, 2):
             # If somehow passed cart, convert
             bary_init = _cart_to_bary(bary_init)
        cart = _bary_to_cart(bary_init)
        return cart, _vectorized_min_area_cart(cart)

def _soft_min_anneal_bary(bary_init, start_time, deadline, rng):
    """Multi-stage soft-min annealing in barycentric coordinates."""
    best_bary = bary_init.copy()
    best_cart = _bary_to_cart(best_bary)
    best_score = _vectorized_min_area_cart(best_cart)
    
    # Stage 1: Global search with large temperature
    T_stages = [0.05, 0.02, 0.008, 0.003, 0.001, 0.0004]
    
    for stage, T in enumerate(T_stages):
        if time.time() > deadline - 15:
            break
        
        bary_flat = best_bary.flatten()
        
        def obj(flat):
            return _bary_objective(flat, T)
        
        try:
            res = minimize(obj, bary_flat, method='L-BFGS-B',
                          bounds=[(0.0, 1.0)] * (N * 3),
                          options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 30})
            new_bary = res.x.reshape(N, 3)
            new_bary = np.maximum(new_bary, 0)
            new_bary /= new_bary.sum(axis=1, keepdims=True)
            new_cart = _bary_to_cart(new_bary)
            new_score = _vectorized_min_area_cart(new_cart)
            
            if new_score > best_score:
                best_score = new_score
                best_bary = new_bary
                best_cart = new_cart
        except:
            pass
        
        # Add random restarts within this stage
        for restart in range(3):
            if time.time() > deadline - 15:
                break
            perturbed = best_bary + rng.normal(0, 0.02, size=(N, 3))
            perturbed = np.maximum(perturbed, 0)
            perturbed /= perturbed.sum(axis=1, keepdims=True)
            
            bary_flat = perturbed.flatten()
            try:
                res = minimize(obj, bary_flat, method='L-BFGS-B',
                              bounds=[(0.0, 1.0)] * (N * 3),
                              options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 30})
                new_bary = res.x.reshape(N, 3)
                new_bary = np.maximum(new_bary, 0)
                new_bary /= new_bary.sum(axis=1, keepdims=True)
                new_cart = _bary_to_cart(new_bary)
                new_score = _vectorized_min_area_cart(new_cart)
                
                if new_score > best_score:
                    best_score = new_score
                    best_bary = new_bary
                    best_cart = new_cart
            except:
                pass
    
    return best_cart, best_score, best_bary

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    np.random.seed(42)
    rng = np.random.RandomState(42)
    
    # Start with structured initial configuration
    init_bary = _structured_initial_bary()
    best_cart = _bary_to_cart(init_bary)
    best_score = _vectorized_min_area_cart(best_cart)
    
    # Soft-min annealing in barycentric coordinates
    if time.time() < deadline - 20:
        cart, score, bary = _soft_min_anneal_bary(init_bary, start_time, deadline, rng)
        if score > best_score:
            best_score = score
            best_cart = cart
    
    # High-precision polish
    if time.time() < deadline - 5:
        polished_bary = _cart_to_bary(best_cart)
        polished, polish_score = _local_polish_bary(polished_bary, max_iter=5000, tol=1e-14)
        if polish_score > best_score:
            best_score = polish_score
            best_cart = polished
    
    # Additional polish attempts from perturbed versions
    for attempt in range(4):
        if time.time() > deadline - 2:
            break
        perturbed_bary = _cart_to_bary(best_cart)
        perturbed_bary += rng.normal(0, 0.005, size=(N, 3))
        perturbed_bary = np.maximum(perturbed_bary, 0)
        perturbed_bary /= perturbed_bary.sum(axis=1, keepdims=True)
        polished, polish_score = _local_polish_bary(perturbed_bary, max_iter=2000, tol=1e-13)
        if polish_score > best_score:
            best_score = polish_score
            best_cart = polished
    
    return best_cart
# EVOLVE-BLOCK-END
