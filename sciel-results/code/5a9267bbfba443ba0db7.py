"""
Heilbronn triangles: maximize the smallest triangle area among 11 points.
Strategy: Topological Active-Set Pivot using Delaunay Triangulation structure.
1. Global Search: Differential Evolution to find a good topology (Delaunay graph).
2. Local Refinement: Solve linear systems to equalize the areas of the critical (Delaunay) triangles to the minimum.
3. Polish: Nelder-Mead on the full set of triangles to handle non-Delaunay critical triangles.
"""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import Delaunay
import itertools

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of the equilateral triangle with side 1

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

DEADLINE = 145  # Safety margin from 150s limit

def _bary_to_cart(bary):
    """
    Convert barycentric coordinates (u, v) to Cartesian.
    bary: shape (N, 2) where u, v >= 0 and u+v <= 1.
    P = u*V0 + v*V1 + (1-u-v)*V2
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = u * V0[0] + v * V1[0] + w * V2[0]
    y = u * V0[1] + v * V1[1] + w * V2[1]
    return np.column_stack([x, y])

def _cart_to_bary(pts):
    """
    Convert Cartesian to barycentric (u, v).
    """
    u = (S3H - pts[:, 1]) / S3H
    v = 1.0 - 2.0 * pts[:, 0]
    return np.column_stack([u, v])

def _min_triangle_area_cart(pts):
    """Compute the minimum area of any triangle formed by three points (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)

def _project_bary_to_valid(bary):
    """Ensure barycentric coordinates are within the triangle."""
    u = np.clip(bary[:, 0], 0, 1)
    v = np.clip(bary[:, 1], 0, 1)
    sum_uv = u + v
    mask = sum_uv > 1.0
    if np.any(mask):
        scale = 1.0 / sum_uv[mask]
        u[mask] *= scale
        v[mask] *= scale
    return np.column_stack([u, v])

def _solve_pivot(pts):
    """
    Topological Active-Set Pivot:
    1. Compute Delaunay triangulation of the points.
    2. Identify the critical triangles (Delaunay triangles).
    3. Solve for the displacement of points such that all critical triangle areas equal the minimum area.
    This effectively pushes the points apart to equalize the smallest areas.
    """
    # Delaunay requires points to be in general position and strictly inside convex hull or on boundary.
    # We add the 3 vertices of the container triangle to ensure a valid triangulation that covers the domain.
    # However, the task is about the 11 points. The Delaunay of just the 11 points might not include all critical triangles
    # if the critical triangle involves the boundary. But usually, the internal structure is what matters.
    # A better approach for "Active Set" in Heilbronn problems is often to consider the Delaunay of the points.
    
    # To ensure we catch boundary-critical triangles, we can include the container vertices in the Delaunay computation
    # but only optimize the 11 points.
    # Let's compute Delaunay of the 11 points.
    try:
        tri = Delaunay(pts)
    except Exception:
        return pts
    
    # Get the indices of the Delaunay triangles
    tri_indices = tri.simplices
    
    # If the number of triangles is small or something is wrong, skip
    if len(tri_indices) == 0:
        return pts
        
    # We want to equalize the areas of these triangles to the current minimum area among ALL triangles.
    # Actually, a simpler heuristic often used: equalize the areas of the Delaunay triangles to their average or min.
    # Let's compute the areas of the Delaunay triangles.
    pi = pts[tri_indices[:, 0]]
    pj = pts[tri_indices[:, 1]]
    pk = pts[tri_indices[:, 2]]
    
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    
    # The target area is the minimum of these Delaunay areas.
    target_area = np.min(areas)
    
    # We want to move points to increase the smallest areas.
    # We can set up a least-squares problem:
    # For each Delaunay triangle i, we want area_i = target_area (if area_i < target_area, this is impossible without moving others, 
    # but we want to push the small ones up).
    # Actually, we want to maximize the minimum. 
    # A common move is to solve for displacements dx such that the areas of the currently smallest triangles increase.
    
    # Let's identify the triangles that are "active" (close to the minimum).
    active_mask = areas < (target_area * 1.1 + 1e-9)
    active_tris = tri_indices[active_mask]
    
    if len(active_tris) == 0:
        return pts
        
    # Build the Jacobian for the active triangles.
    # Area_i = 0.5 * | (p_j - p_i) x (p_k - p_i) |
    # Let's assume the orientation is consistent (Delaunay usually is, or we can fix it).
    # Gradient of Area w.r.t p_i: 0.5 * (dy_j - dy_k, dx_k - dx_i) ... wait.
    # d/dp_i (0.5 * (x_j-x_i)(y_k-y_i) - (y_j-y_i)(x_k-x_i))
    # = 0.5 * ( -(y_k-y_i) - (x_j-x_i)*0 + (x_k-x_i)*0 - ... ) 
    # Let's use the standard formula:
    # dA/dp_i = 0.5 * (dy_k - dy_j, dx_j - dx_k)  -- check signs
    # dA/dp_j = 0.5 * (dy_i - dy_k, dx_k - dx_i)
    # dA/dp_k = 0.5 * (dy_j - dy_i, dx_i - dx_j)
    
    # We want to solve J * dx = b, where b is the desired change in area.
    # We want to increase the area of the active triangles.
    # Let's set the target for each active triangle to be slightly higher than the current min, say the average of active areas + epsilon.
    # Or simply, we want to push them all to the average of the Delaunay areas.
    target_avg = np.mean(areas)
    
    # Construct J (M_active x 2N) and b (M_active)
    M = len(active_tris)
    J = np.zeros((M, 2 * N))
    b = np.zeros(M)
    
    for idx, t in enumerate(active_tris):
        i, j, k = t
        pi_xy = pts[i]
        pj_xy = pts[j]
        pk_xy = pts[k]
        
        # Current area
        a_cur = 0.5 * abs((pj_xy[0]-pi_xy[0])*(pk_xy[1]-pi_xy[1]) - (pj_xy[1]-pi_xy[1])*(pk_xy[0]-pi_xy[0]))
        
        # Desired change
        delta_a = target_avg - a_cur
        
        # Gradients (assuming positive orientation, if not, the sign might be flipped, but least squares handles magnitude)
        # Let's compute the signed area gradient to be safe.
        # If the triangle is clockwise, the gradient points inwards. We want to expand.
        # Let's just use the magnitude gradient direction.
        
        # dA/dpi
        grad_pi_x = 0.5 * ( (pk_xy[1]-pi_xy[1]) - (pj_xy[1]-pi_xy[1]) ) # This is 0.5*(dy_k - dy_j) ? No.
        # Let's derive properly.
        # A = 0.5 * ( (x_j-x_i)(y_k-y_i) - (y_j-y_i)(x_k-x_i) )
        # dA/dx_i = 0.5 * ( -(y_k-y_i) - (y_j-y_i) ) = 0.5 * (y_i - y_k + y_i - y_j) = 0.5 * (2y_i - y_j - y_k) ?? No.
        # dA/dx_i = 0.5 * ( -(y_k-y_i) - (-(y_j-y_i)) ) = 0.5 * ( -y_k + y_i + y_j - y_i ) = 0.5 * (y_j - y_k)
        # dA/dy_i = 0.5 * ( -(x_k-x_i) - (-(x_j-x_i)) ) = 0.5 * ( -x_k + x_i + x_j - x_i ) = 0.5 * (x_j - x_k)
        
        # dA/dx_j = 0.5 * ( (y_k-y_i) )
        # dA/dy_j = 0.5 * ( -(x_k-x_i) )
        
        # dA/dx_k = 0.5 * ( -(y_j-y_i) )
        # dA/dy_k = 0.5 * ( (x_j-x_i) )
        
        g_pi = np.array([0.5 * (pj_xy[1] - pk_xy[1]), 0.5 * (pk_xy[0] - pj_xy[0])])
        g_pj = np.array([0.5 * (pk_xy[1] - pi_xy[1]), 0.5 * (pi_xy[0] - pk_xy[0])])
        g_pk = np.array([0.5 * (pi_xy[1] - pj_xy[1]), 0.5 * (pj_xy[0] - pi_xy[0])])
        
        # Check if the triangle is CW or CCW.
        cross_val = (pj_xy[0]-pi_xy[0])*(pk_xy[1]-pi_xy[1]) - (pj_xy[1]-pi_xy[1])*(pk_xy[0]-pi_xy[0])
        if cross_val < 0:
            g_pi = -g_pi
            g_pj = -g_pj
            g_pk = -g_pk
            
        # We want to increase the area by delta_a.
        # So we solve J * dx = delta_a
        J[idx, 2*i:2*i+2] = g_pi
        J[idx, 2*j:2*j+2] = g_pj
        J[idx, 2*k:2*k+2] = g_pk
        b[idx] = delta_a
        
    # Solve least squares
    try:
        dx, residuals, rank, s = np.linalg.lstsq(J, b, rcond=None)
    except Exception:
        return pts
        
    # Apply small step
    step = 0.1
    new_pts = pts + step * dx.reshape(N, 2)
    
    # Project to valid region (barycentric)
    new_bary = _cart_to_bary(new_pts)
    new_bary = _project_bary_to_valid(new_bary)
    new_pts = _bary_to_cart(new_bary)
    
    return new_pts

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_bary = None
    best_area = 0.0
    best_cart = None

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(bary):
        nonlocal best_bary, best_area, best_cart
        bary = _project_bary_to_valid(bary)
        cart = _bary_to_cart(bary)
        area = _min_triangle_area_cart(cart)
        if area > best_area:
            best_area = area
            best_bary = bary.copy()
            best_cart = cart.copy()

    # Initial symmetric configuration
    base_sym = np.array([
        [1.0, 0.0],   # V0
        [0.0, 1.0],   # V1
        [0.0, 0.0],   # V2
        [0.5, 0.5],   # Mid V0-V1
        [0.0, 0.5],   # Mid V1-V2
        [0.5, 0.0],   # Mid V2-V0
        [1/3, 1/3],   # Center
        [0.2, 0.2],
        [0.3, 0.3],
        [0.4, 0.2],
        [0.2, 0.4],
    ])
    _update_best(base_sym)

    # Phase 1: Differential Evolution to find a good topology
    def de_objective(flat_bc):
        bc = flat_bc.reshape(-1, 2)
        u = bc[:, 0]
        v = bc[:, 1]
        
        penalty = 0.0
        violation = np.maximum(0, u + v - 1)
        penalty = np.sum(violation ** 2) * 1000.0
        
        bary = np.column_stack([u, v])
        cart = _bary_to_cart(bary)
        area = _min_triangle_area_cart(cart)
        
        return -area + penalty

    bounds_de = [(0, 1), (0, 1)] * N
    
    if _elapsed_ok(60):
        try:
            de_result = differential_evolution(
                de_objective,
                bounds_de,
                maxiter=50,
                popsize=20,
                seed=42,
                tol=1e-4,
                polish=False,
                init='latinhypercube',
                updating='deferred'
            )
            if de_result.success or de_result.fun < 1e10:
                de_bary = de_result.x.reshape(N, 2)
                de_bary = _project_bary_to_valid(de_bary)
                _update_best(de_bary)
        except Exception:
            pass

    # Phase 2: Topological Active-Set Pivot Refinement
    # Iterate the pivot step to equalize Delaunay triangle areas
    if best_bary is not None and _elapsed_ok(40):
        current_bary = best_bary.copy()
        current_cart = _bary_to_cart(current_bary)
        
        # Run pivot for several iterations
        for _ in range(50):
            if not _elapsed_ok(5):
                break
            
            new_cart = _solve_pivot(current_cart)
            new_bary = _cart_to_bary(new_cart)
            new_bary = _project_bary_to_valid(new_bary)
            
            # Check if it improved
            new_area = _min_triangle_area_cart(new_cart)
            if new_area > best_area:
                current_bary = new_bary
                current_cart = new_cart
                _update_best(current_bary)
            else:
                # If it didn't improve, maybe the step was too large or we hit a local optimum.
                # Try a smaller step or break?
                # Let's just break if no improvement in last few steps to save time.
                # For simplicity, just break.
                break

    # Phase 3: Final Polish with Nelder-Mead on Hard Minimum
    if best_bary is not None and _elapsed_ok(15):
        nm_start = best_bary.flatten()
        
        def nm_objective(x):
            b = x.reshape(N, 2)
            b = _project_bary_to_valid(b)
            cart = _bary_to_cart(b)
            return -_min_triangle_area_cart(cart)
        
        try:
            res_nm = minimize(nm_objective, nm_start, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-11})
            if res_nm.success:
                nm_bary = res_nm.x.reshape(N, 2)
                nm_bary = _project_bary_to_valid(nm_bary)
                _update_best(nm_bary)
        except Exception:
            pass

    # Fallback
    if best_cart is None:
        u = rng.random(N)
        v = rng.random(N)
        mask = u + v > 1
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        bary = np.column_stack([u, v])
        best_cart = _bary_to_cart(bary)
        
    return best_cart

# EVOLVE-BLOCK-END
