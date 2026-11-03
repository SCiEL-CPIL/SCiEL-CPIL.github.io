"""Heilbronn 11-point problem: maximize minimum triangle area in unit equilateral triangle."""
import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

DEADLINE_MARGIN = 145.0

# Precompute all triplets
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
NUM_TRIP = len(_TRIPLETS)  # 165

# Triangle vertices
_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, S3H]])

# Fixed corners in barycentric coordinates
# v0 = (1,0,0), v1 = (0,1,0), v2 = (0,0,1)
_FIXED_BARY = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])

# Number of free points
N_FREE = N - 3

def _bary_to_cart(bary):
    """Convert barycentric (N,3) to Cartesian (N,2)."""
    return bary[:, 0:1] * _VERTS[0] + bary[:, 1:2] * _VERTS[1] + bary[:, 2:3] * _VERTS[2]

def _min_area_cart(points):
    """Minimum area of all triangles formed by points (Cartesian)."""
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _all_areas_cart(points):
    """All triangle areas (Cartesian)."""
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _soft_min_cart(points, alpha):
    """Soft minimum of triangle areas via log-sum-exp."""
    areas = _all_areas_cart(points)
    min_a = np.min(areas)
    return min_a - (1.0 / alpha) * np.log(np.sum(np.exp(-alpha * (areas - min_a))))

def _project_to_triangle(points):
    """Project points into the equilateral triangle."""
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    for _ in range(3):
        y = np.maximum(0.0, y)
        y = np.minimum(y, S3 * x)
        y = np.minimum(y, S3 - S3 * x)
        x = np.maximum(0.0, x)
        x = np.minimum(1.0, x)
        y = np.maximum(0.0, y)
        y = np.minimum(y, S3 * x)
        y = np.minimum(y, S3 - S3 * x)
    return np.column_stack([x, y])

def _is_valid(points, tol=1e-8):
    x = points[:, 0]
    y = points[:, 1]
    return (np.all(y >= -tol) and
            np.all(y <= S3 * x + tol) and
            np.all(S3 * x + y <= S3 + tol))

def _free_bary_obj(flat, alpha):
    """
    Objective for the free points in barycentric coordinates.
    flat: (N_FREE * 3,) array of barycentric coords for the 8 free points.
    """
    # Reshape to (N_FREE, 3)
    free_bary = flat.reshape(N_FREE, 3)
    
    # Normalize and clip to ensure valid barycentric coords
    free_bary = np.maximum(0.0, free_bary)
    sums = free_bary.sum(axis=1, keepdims=True)
    sums = np.where(sums < 1e-12, 1.0, sums)
    free_bary = free_bary / sums
    
    # Combine with fixed corners
    full_bary = np.vstack([_FIXED_BARY, free_bary])
    
    # Convert to Cartesian
    cart = _bary_to_cart(full_bary)
    
    # Calculate soft min area
    soft = _soft_min_cart(cart, alpha)
    
    # No penalty needed if we enforce normalization and clipping strictly in the objective
    # But just in case, add a tiny penalty if any point is slightly out (due to float errors)
    x = cart[:, 0]
    y = cart[:, 1]
    pen = (np.sum(np.maximum(0.0, -y) ** 2) +
           np.sum(np.maximum(0.0, y - S3 * x) ** 2) +
           np.sum(np.maximum(0.0, S3 * x + y - S3) ** 2))
           
    return -soft + 100.0 * pen

def _construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN

    np.random.seed(42)

    best_cart = None
    best_score = 0.0

    # Generate initial configurations for the free points in barycentric space
    # We use a few heuristic starts and random starts
    
    initial_free_barys = []
    
    # 1. Uniform random
    for _ in range(30):
        r = np.random.random((N_FREE, 3))
        r /= r.sum(axis=1, keepdims=True)
        initial_free_barys.append(r.flatten())
        
    # 2. Grid-like in barycentric
    # Create a grid of points in the triangle
    for i in range(10):
        # Simple lattice points
        xs = np.linspace(0.1, 0.9, 3)
        ys = np.linspace(0.1, 0.9, 3)
        # Take 8 points
        grid_pts = []
        for x in xs:
            for y in ys:
                # Convert to barycentric roughly? No, just randomize around a center
                pass
        
        # Better: Just perturb the centroid
        centroid = np.array([1/3, 1/3, 1/3])
        pts = []
        for _ in range(N_FREE):
            p = centroid + np.random.normal(0, 0.1, 3)
            p = np.maximum(0, p)
            p /= p.sum()
            pts.append(p)
        initial_free_barys.append(np.array(pts).flatten())

    # Annealing schedule
    alphas = [10, 50, 100, 300, 1000, 3000, 10000]

    for alpha in alphas:
        if time.time() > deadline:
            break

        for init_flat in initial_free_barys:
            if time.time() > deadline:
                break

            # Nelder-Mead optimization on the free barycentric coordinates
            res = minimize(_free_bary_obj, init_flat, method='Nelder-Mead',
                           args=(alpha,),
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            # Reconstruct full configuration
            new_free_bary = res.x.reshape(N_FREE, 3)
            new_free_bary = np.maximum(0.0, new_free_bary)
            sums = new_free_bary.sum(axis=1, keepdims=True)
            sums = np.where(sums < 1e-12, 1.0, sums)
            new_free_bary = new_free_bary / sums
            
            full_bary = np.vstack([_FIXED_BARY, new_free_bary])
            new_cart = _bary_to_cart(full_bary)
            
            # Ensure validity (projection)
            new_cart = _project_to_triangle(new_cart)
            
            score = _min_area_cart(new_cart)
            
            if score > best_score:
                best_score = score
                best_cart = new_cart

    # Refinement phase
    if best_cart is not None:
        # Convert best cart back to bary to refine? 
        # It's better to refine the bary coords directly if we kept them.
        # But we lost the exact bary coords in the loop above if we didn't store them.
        # Let's re-derive bary from cart for the final polish, or just use NM on cart?
        # Since we fixed corners, we should stick to bary for consistency.
        
        # Re-derive bary for best_cart
        # Note: _cart_to_bary was removed, let's implement a quick one or use the inverse of _bary_to_cart?
        # Actually, we can just run NM on the Cartesian coords of the free points?
        # No, the constraint is easier in Bary.
        
        # Let's implement a quick cart_to_bary for the free points
        def _cart_to_bary_free(cart_free):
            x = cart_free[:, 0]
            y = cart_free[:, 1]
            b2 = y / S3
            b1 = x - 0.5 * b2
            b0 = 1.0 - b1 - b2
            return np.column_stack([b0, b1, b2])

        # Get free cart points (indices 3 to 10)
        free_cart = best_cart[3:]
        free_bary_init = _cart_to_bary_free(free_cart)
        
        # Polish with high alpha
        for alpha in [5000, 10000, 20000]:
            if time.time() > deadline - 5:
                break
            res = minimize(_free_bary_obj, free_bary_init.flatten(), method='Nelder-Mead',
                           args=(alpha,),
                           options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9})
            new_free_bary = res.x.reshape(N_FREE, 3)
            new_free_bary = np.maximum(0.0, new_free_bary)
            sums = new_free_bary.sum(axis=1, keepdims=True)
            sums = np.where(sums < 1e-12, 1.0, sums)
            new_free_bary = new_free_bary / sums
            
            full_bary = np.vstack([_FIXED_BARY, new_free_bary])
            new_cart = _bary_to_cart(full_bary)
            new_cart = _project_to_triangle(new_cart)
            
            score = _min_area_cart(new_cart)
            if score > best_score:
                best_score = score
                best_cart = new_cart
                free_bary_init = new_free_bary # Update start for next alpha

    # Fallback
    if best_cart is None:
        # Generate a valid random config
        r = np.random.random((N, 3))
        r /= r.sum(axis=1, keepdims=True)
        best_cart = _bary_to_cart(r)
        best_cart = _project_to_triangle(best_cart)

    return best_cart

def construct():
    return _construct()
