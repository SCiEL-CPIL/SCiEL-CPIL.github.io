"""Combined program for Heilbronn-11: Barycentric optimization with soft-min annealing and symmetry-aware polish."""
import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11

# EVOLVE-BLOCK-START
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of equilateral triangle with side 1

# Safety margin from 150s limit. 
# The evaluator kills after 150s. We aim to return well before.
DEADLINE = 140.0 

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
NUM_TRIP = len(_TRIPLETS)  # 165

# Triangle vertices in Cartesian
_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, S3H]])

def _bary_to_cart(bary):
    """Convert barycentric coordinates (N, 3) to Cartesian (N, 2)."""
    return bary[:, 0:1] * _VERTS[0] + bary[:, 1:2] * _VERTS[1] + bary[:, 2:3] * _VERTS[2]

def _cart_to_bary(cart):
    """Convert Cartesian (N, 2) to barycentric coordinates (N, 3)."""
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

def _symmetric_init():
    """
    Generate a symmetric initial configuration exploiting D3 symmetry.
    Place points in orbits under the symmetry group of the equilateral triangle.
    For N=11: 1 center + 3 vertices + 3 edge midpoints + 4 interior in a ring
    Actually: 1 center, 3 near-vertices, 3 near-edge-midpoints, 4 in a smaller ring
    """
    pts = np.zeros((N, 2))
    # Point 0: center
    pts[0] = [0.5, S3H / 3]
    
    # Points 1-3: near vertices (pushed inward)
    eps = 0.15
    pts[1] = [eps, 0]
    pts[2] = [1 - eps, 0]
    pts[3] = [0.5, S3H * (1 - eps)]
    
    # Points 4-6: near edge midpoints (pushed inward)
    eps2 = 0.3
    pts[4] = [0.5 * (1 - eps2), 0]
    pts[5] = [0.5 * (1 + eps2), S3 * 0.5 * (1 - eps2) * 0.5]  # midpoint of right edge, pushed in
    pts[6] = [0.25 * (1 - eps2), S3 * 0.75 * (1 - eps2)]  # midpoint of left edge, pushed in
    
    # Points 7-10: ring of 4 points around center
    r_ring = 0.2
    for i in range(4):
        angle = 2 * np.pi * i / 4 + np.pi / 4
        pts[7 + i] = [0.5 + r_ring * np.cos(angle), S3H / 3 + r_ring * np.sin(angle) * 0.577]
    
    pts = _project_cartesian(pts)
    return pts

def _bary_objective(flat_bary, alpha=100):
    """Objective function in barycentric coordinates."""
    bary = flat_bary.reshape(N, 3)
    # Ensure valid barycentric (non-negative, sum to 1)
    bary = np.maximum(0, bary)
    bary /= bary.sum(axis=1, keepdims=True)
    
    cart = _bary_to_cart(bary)
    cart = _project_cartesian(cart)
    
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

def _active_set_polish(points, deadline, K=30, alphas=[500, 2000, 10000, 50000]):
    """
    Refine the solution by optimizing a soft-min objective over the K smallest triangles.
    Uses coordinate descent as a final polish.
    """
    current_points = points.copy()
    current_score = _vectorized_min_area_cart(current_points)
    
    for alpha in alphas:
        if time.time() > deadline:
            break
            
        # Identify the K smallest triangles
        p1 = current_points[_T_I]
        p2 = current_points[_T_J]
        p3 = current_points[_T_K]
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        # Get indices of K smallest areas
        indices = np.argsort(areas)[:K]
        
        active_i = _T_I[indices]
        active_j = _T_J[indices]
        active_k = _T_K[indices]
        
        def active_objective(flat_cart):
            cart = flat_cart.reshape(N, 2)
            cart = _project_cartesian(cart)
            
            p1_a = cart[active_i]
            p2_a = cart[active_j]
            p3_a = cart[active_k]
            v1_a = p2_a - p1_a
            v2_a = p3_a - p1_a
            cross_a = v1_a[:, 0] * v2_a[:, 1] - v1_a[:, 1] * v2_a[:, 0]
            areas_a = 0.5 * np.abs(cross_a)
            
            min_a = np.min(areas_a)
            soft_min = min_a - (1.0 / alpha) * np.log(np.sum(np.exp(-alpha * (areas_a - min_a))))
            
            x = cart[:, 0]
            y = cart[:, 1]
            pen1 = np.sum(np.maximum(0, -y)**2)
            pen2 = np.sum(np.maximum(0, y - S3 * x)**2)
            pen3 = np.sum(np.maximum(0, S3 * x + y - S3)**2)
            pen = pen1 + pen2 + pen3
            
            return -soft_min + 100 * pen
            
        res = minimize(active_objective, current_points.flatten(), method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
        
        # Evaluate the result
        new_points = res.x.reshape(N, 2)
        new_points = _project_cartesian(new_points)
        new_score = _vectorized_min_area_cart(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
    
    # Final coordinate descent polish on the active set
    if time.time() < deadline - 2:
        current_points, current_score = _coord_descent_polish(current_points, deadline, K=30)
    
    return current_points, current_score

def _coord_descent_polish(points, deadline, K=30):
    """
    Coordinate descent: for each point, optimize its position while others are fixed.
    Focus on points that are vertices of small triangles.
    """
    current_points = points.copy()
    current_score = _vectorized_min_area_cart(current_points)
    
    # Find which points are in the K smallest triangles
    p1 = current_points[_T_I]
    p2 = current_points[_T_J]
    p3 = current_points[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    indices = np.argsort(areas)[:K]
    active_points = set()
    for idx in indices:
        active_points.add(_T_I[idx])
        active_points.add(_T_J[idx])
        active_points.add(_T_K[idx])
    
    # Sort active points by how many small triangles they're in (most constrained first)
    point_count = np.zeros(N, dtype=int)
    for idx in indices:
        point_count[_T_I[idx]] += 1
        point_count[_T_J[idx]] += 1
        point_count[_T_K[idx]] += 1
    
    ordered_points = sorted(active_points, key=lambda p: -point_count[p])
    
    for _ in range(3):  # Multiple passes
        if time.time() > deadline:
            break
        
        for pi in ordered_points:
            if time.time() > deadline:
                break
            
            # Optimize point pi's position
            def single_point_obj(flat_pt):
                pt = flat_pt.reshape(2)
                new_pts = current_points.copy()
                new_pts[pi] = pt
                new_pts = _project_cartesian(new_pts)
                
                # Compute min area
                min_a = _vectorized_min_area_cart(new_pts)
                
                # Small penalty if outside
                x, y = pt
                pen = max(0, -y)**2 + max(0, y - S3 * x)**2 + max(0, S3 * x + y - S3)**2
                
                return -min_a + 10 * pen
            
            res = minimize(single_point_obj, current_points[pi], method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            
            new_pt = res.x.reshape(2)
            new_pt = _project_cartesian(new_pt.reshape(1, 2))[0]
            
            test_pts = current_points.copy()
            test_pts[pi] = new_pt
            test_pts = _project_cartesian(test_pts)
            test_score = _vectorized_min_area_cart(test_pts)
            
            if test_score > current_score:
                current_points = test_pts
                current_score = test_score
    
    return current_points, current_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    np.random.seed(42)
    
    # Start with symmetric initialization
    best_cart = _symmetric_init()
    best_score = _vectorized_min_area_cart(best_cart)
    best_bary = _cart_to_bary(best_cart)
    
    # Also try a few random starts
    for _ in range(3):
        random_bary = _random_bary()
        random_cart = _bary_to_cart(random_bary)
        random_cart = _project_cartesian(random_cart)
        random_score = _vectorized_min_area_cart(random_cart)
        if random_score > best_score:
            best_score = random_score
            best_bary = random_bary
            best_cart = random_cart
    
    # Annealing schedule - fewer restarts, faster convergence
    alphas = [50, 200, 500, 2000]
    
    for alpha in alphas:
        if time.time() > deadline - 20:  # Leave time for polish
            break
        
        # Fewer restarts, tighter tolerances
        num_restarts = 5
        for r in range(num_restarts):
            if time.time() > deadline - 20:
                break
            
            # Start from current best or random
            if r == 0:
                x0 = best_bary.flatten()
            else:
                x0 = _random_bary().flatten()
            
            # Optimize with Nelder-Mead in barycentric space
            res = minimize(_bary_objective, x0, method='Nelder-Mead',
                          args=(alpha,),
                          options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8})
            
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
    
    # Final refinement with active set smooth polish
    if time.time() < deadline - 5:
        best_cart, best_score = _active_set_polish(best_cart, deadline)
    
    return best_cart
# EVOLVE-BLOCK-END
