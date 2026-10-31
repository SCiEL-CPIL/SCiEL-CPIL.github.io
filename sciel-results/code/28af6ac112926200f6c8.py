import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0  # Safety margin for 150s limit

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def cart_to_bary(cart):
    """Convert Cartesian (x, y) to barycentric (u, v)."""
    x = cart[:, 0]
    y = cart[:, 1]
    v = 2.0 * y / H
    u = 1.0 - v - x
    return np.column_stack([u, v])

def get_areas_cartesian(cart):
    """Compute all triangle areas."""
    p_i = cart[tri_idx[:, 0]]
    p_j = cart[tri_idx[:, 1]]
    p_k = cart[tri_idx[:, 2]]
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_k[:, 0] - p_i[:, 0]) * (p_j[:, 1] - p_i[:, 1])
    return 0.5 * np.abs(cross)

def calc_min_area_cartesian(points):
    """
    Calculate the minimum area of all triangles formed by N points.
    points: (N, 2) array of Cartesian coordinates.
    """
    return np.min(get_areas_cartesian(points))

def clamp_bary(bary, eps=1e-9):
    """Clamp barycentric coordinates to the strictly valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def hard_min_obj_bary(bary_flat):
    """Objective using hard min in barycentric coords."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    cart = bary_to_cart(bary)
    return -calc_min_area_cartesian(cart)

def active_triangles(cart, threshold_frac=0.05, max_count=20):
    """
    Find triangles with areas near the minimum (active constraints).
    Returns list of (i, j, k) indices.
    """
    areas = get_areas_cartesian(cart)
    min_area = np.min(areas)
    if min_area <= 0:
        return [(0, 1, 2)]
    
    threshold = min_area * (1 + threshold_frac)
    active = np.where(areas <= threshold)[0]
    
    if len(active) > max_count:
        # Take the smallest ones
        order = np.argsort(areas[active])
        active = active[order[:max_count]]
    
    return [(tri_idx[t, 0], tri_idx[t, 1], tri_idx[t, 2]) for t in active]

def coordinated_ascent(bary_init, time_limit=10.0):
    """
    Coordinated ascent: iteratively optimize individual points to increase
    the minimum area, focusing on active constraints.
    """
    start_time = time.time()
    bary = bary_init.copy()
    
    def current_min():
        cart = bary_to_cart(bary)
        return calc_min_area_cartesian(cart)
    
    best_val = current_min()
    
    max_rounds = 50
    for round_idx in range(max_rounds):
        if time.time() - start_time > time_limit - 2:
            break
        
        cart = bary_to_cart(bary)
        areas = get_areas_cartesian(cart)
        min_area = np.min(areas)
        
        # Get active triangles
        active = active_triangles(cart, threshold_frac=0.03, max_count=15)
        
        # For each point, try to move it to improve the minimum
        improved = False
        for pt_idx in range(N):
            if time.time() - start_time > time_limit - 2:
                break
            
            # Collect constraints involving this point
            constraints = []
            for (i, j, k) in active:
                involved = set([i, j, k])
                if pt_idx in involved:
                    constraints.append((i, j, k))
            
            if not constraints:
                continue
            
            # Try small perturbations in barycentric space
            best_delta = None
            best_area = calc_min_area_cartesian(cart)
            
            # Random directions
            rng = np.random.default_rng(round_idx * 1000 + pt_idx)
            for attempt in range(30):
                # Random direction in 2D barycentric
                angle = rng.uniform(0, 2 * np.pi)
                magnitude = rng.uniform(0.01, 0.05)
                du = magnitude * np.cos(angle)
                dv = magnitude * np.sin(angle)
                
                new_bary = bary.copy()
                new_bary[pt_idx, 0] += du
                new_bary[pt_idx, 1] += dv
                new_bary = clamp_bary(new_bary)
                
                new_cart = bary_to_cart(new_bary)
                new_min = calc_min_area_cartesian(new_cart)
                
                if new_min > best_area + 1e-12:
                    best_area = new_min
                    best_delta = (du, dv)
            
            # Also try gradient-like direction: move away from centroid of active triangles
            if best_delta is None:
                # Compute average position of other points in active triangles
                other_pts = []
                for (i, j, k) in constraints:
                    for idx in [i, j, k]:
                        if idx != pt_idx:
                            other_pts.append(idx)
                
                if other_pts:
                    other_cart = cart[list(set(other_pts))]
                    centroid = np.mean(other_cart, axis=0)
                    my_cart = cart[pt_idx]
                    
                    # Direction away from centroid
                    direction = my_cart - centroid
                    norm = np.linalg.norm(direction)
                    if norm > 1e-10:
                        direction = direction / norm
                        
                        # Convert cartesian direction to barycentric
                        dx_dir = direction[0]
                        dy_dir = direction[1]
                        du_est = -dy_dir / (2 * H) - dx_dir
                        dv_est = dx_dir - dy_dir / (2 * H)
                        
                        for scale in [0.02, 0.01, 0.005]:
                            new_bary = bary.copy()
                            new_bary[pt_idx, 0] += scale * du_est
                            new_bary[pt_idx, 1] += scale * dv_est
                            new_bary = clamp_bary(new_bary)
                            
                            new_cart = bary_to_cart(new_bary)
                            new_min = calc_min_area_cartesian(new_cart)
                            
                            if new_min > best_area + 1e-12:
                                best_area = new_min
                                best_delta = (scale * du_est, scale * dv_est)
            
            # Apply best delta if found
            if best_delta is not None:
                bary[pt_idx, 0] += best_delta[0]
                bary[pt_idx, 1] += best_delta[1]
                bary = clamp_bary(bary)
                improved = True
        
        if not improved:
            break
            
    return bary

def generate_lattice_init():
    """
    Generate a specific high-symmetry lattice initialization.
    """
    pts = []
    # 3 points on bottom edge
    pts.append([0.0, 0.0])
    pts.append([0.5, 0.0])
    pts.append([1.0, 0.0])
    
    # 3 points in next row up
    pts.append([0.25, 0.25])
    pts.append([0.5, 0.25])
    pts.append([0.75, 0.25])
    
    # 2 points in row above
    pts.append([0.375, 0.5])
    pts.append([0.625, 0.5])
    
    # 2 points in top row
    pts.append([0.5, 0.75])
    pts.append([0.25, 0.5])
    
    # Wait, let's just use a standard grid and pick 11
    grid_pts = []
    step = 0.25
    for i in range(5):
        for j in range(5 - i):
            u = i * step
            v = j * step
            w = 1.0 - u - v
            if w >= -1e-9:
                grid_pts.append([u, v])
                
    return np.array(grid_pts[:11])

def generate_hexagonal_init():
    """
    Generate a hexagonal packing initialization.
    """
    pts = []
    # Row 0 (bottom): 4 points
    for i in range(4):
        pts.append([0.25 + i*0.25, 0.0])
    
    # Row 1: 3 points
    for i in range(3):
        pts.append([0.375 + i*0.25, 0.25])
    
    # Row 2: 2 points
    for i in range(2):
        pts.append([0.5 + i*0.25, 0.5])
        
    # Row 3: 1 point
    pts.append([0.5, 0.75])
    
    return np.array(pts[:11])

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using local optimization.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Nelder-Mead on hard min (robust to non-smoothness)
    try:
        result1 = minimize(
            hard_min_obj_bary,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.5:
        return bary_current
    
    # Stage 2: SLSQP polish with hard min for precise convergence
    try:
        result2 = minimize(
            hard_min_obj_bary,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14}
        )
        bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    return bary_current

def apply_affine(base_pts, params):
    """
    Apply an affine transformation to base_pts.
    params: [a, b, c, d, e, f]
    x' = a*x + b*y + e
    y' = c*x + d*y + f
    """
    a, b, c, d, e, f = params
    x = base_pts[:, 0]
    y = base_pts[:, 1]
    new_x = a * x + b * y + e
    new_y = c * x + d * y + f
    return np.column_stack([new_x, new_y])

def affine_obj(params, base_pts_bary):
    """
    Objective for affine parameters.
    """
    base_cart = bary_to_cart(base_pts_bary)
    transformed = apply_affine(base_cart, params)
    
    # Check if points are inside the triangle
    bary = cart_to_bary(transformed)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Penalty if any coordinate is negative (outside)
    min_coord = np.min(np.vstack([u, v, w]).flatten())
    if min_coord < -1e-6:
        return 1e10 - min_coord 
        
    # If slightly outside, clamp and penalize
    if min_coord < 0:
        penalty = -min_coord * 1000
    else:
        penalty = 0
        
    # Calculate min area
    min_area = calc_min_area_cartesian(transformed)
    
    return penalty - min_area

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_bary = None
    best_val = -np.inf

    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        cart = bary_to_cart(bary)
        val = calc_min_area_cartesian(cart)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Standard Lattice Init + Local Refine
    lat_init = generate_lattice_init()
    refined_lat = refine_local(lat_init, time_limit=5.0)
    update_best(refined_lat)
    
    # 2. Coordinated Ascent from Lattice
    if time.time() < deadline - 60:
        ca_init = generate_lattice_init()
        ca_refined = coordinated_ascent(ca_init, time_limit=15.0)
        update_best(ca_refined)
        
        # Refine the coordinated ascent result with local optimization
        if time.time() < deadline - 40:
            ca_local = refine_local(ca_refined, time_limit=10.0)
            update_best(ca_local)

    # 3. Structural Change: Affine Deformation of the Lattice
    if time.time() < deadline - 40:
        base_pts_bary = generate_lattice_init()
        
        # Bounds for affine parameters
        affine_bounds = [
            (-0.5, 2.0),  # a
            (-0.5, 2.0),  # b
            (-0.5, 2.0),  # c
            (-0.5, 2.0),  # d
            (-0.5, 1.5),  # e
            (-0.5, 1.5)   # f
        ]
        
        try:
            # Use Differential Evolution for the 6D affine search
            result_de = differential_evolution(
                affine_obj,
                affine_bounds,
                args=(base_pts_bary,),
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                polish=False
            )
            
            best_affine_params = result_de.x
            # Apply the best affine transform to get a good starting point
            base_cart = bary_to_cart(base_pts_bary)
            transformed_cart = apply_affine(base_cart, best_affine_params)
            
            # Convert back to barycentric
            affine_bary = cart_to_bary(transformed_cart)
            affine_bary = clamp_bary(affine_bary)
            
            # Refine this affine-derived point with full local search
            refined_affine = refine_local(affine_bary, time_limit=10.0)
            update_best(refined_affine)
            
        except Exception:
            pass

    # 4. Multi-start: Random and Perturbed
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 15:
        restart_count += 1
        if best_bary is not None and restart_count % 2 == 0:
            scale = 0.05
            noise = rng.normal(0, scale, best_bary.shape)
            init = clamp_bary(best_bary + noise)
        else:
            u = rng.random(N)
            v = rng.random(N) * (1 - u)
            init = np.column_stack([u, v])
        
        remaining = deadline - 15 - time.time()
        if remaining < 3:
            break
            
        t_limit = min(5.0, remaining)
        
        # Alternate between local refine and coordinated ascent
        if restart_count % 3 == 0:
            refined = coordinated_ascent(init, time_limit=t_limit)
        else:
            refined = refine_local(init, time_limit=t_limit)
            
        update_best(refined)
        
    # Final polish
    if best_bary is not None and time.time() < deadline - 5:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj_bary,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15}
            )
            update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass

    cart = bary_to_cart(best_bary)
    return cart
