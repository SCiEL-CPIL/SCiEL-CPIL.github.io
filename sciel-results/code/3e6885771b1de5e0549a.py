"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _bary_to_cart(b):
    """Convert barycentric coords (u,v) with u,v>=0, u+v<=1 to Cartesian in unit triangle."""
    u = b[:, 0]
    v = b[:, 1]
    x = u + 0.5 * v
    y = (np.sqrt(3) / 2) * v
    return np.column_stack([x, y])

def _cart_to_bary(p):
    """Convert Cartesian to barycentric."""
    x = p[:, 0]
    y = p[:, 1]
    v = y / (np.sqrt(3) / 2)
    u = x - 0.5 * v
    return np.column_stack([u, v])

def _min_triangle_area(points_cart):
    """Compute the minimum area of any triangle formed by 3 of the points."""
    n = len(points_cart)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1, p2, p3 = points_cart[i], points_cart[j], points_cart[k]
                area = 0.5 * abs(np.cross(p2 - p1, p3 - p1))
                if area < min_area:
                    min_area = area
    return min_area

def _min_triangle_area_fast(points_cart):
    """Vectorized computation of minimum triangle area."""
    n = len(points_cart)
    # Compute all pairwise cross products
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            v = points_cart[j] - points_cart[i]
            for k in range(j + 1, n):
                w = points_cart[k] - points_cart[i]
                area = 0.5 * abs(v[0] * w[1] - v[1] * w[0])
                if area < min_area:
                    min_area = area
    return min_area

def _objective(bary_coords):
    """Objective: negative of minimum triangle area (to minimize)."""
    n = len(bary_coords)
    if n != N:
        return 0.0
    # Check constraints: u,v >= 0, u+v <= 1
    if np.any(bary_coords < -1e-9) or np.any(bary_coords[:, 0] + bary_coords[:, 1] > 1 + 1e-9):
        return 0.0
    # Clip to feasible region
    b = np.clip(bary_coords, 0, None)
    b[:, 0] = np.minimum(b[:, 0], 1 - b[:, 1])
    cart = _bary_to_cart(b)
    return -_min_triangle_area(cart)

def _sobol_init(rng):
    """Generate initial points using a quasi-random approach within the triangle."""
    # Use stratified sampling: divide triangle into sub-triangles and place points
    # Simple approach: use random but with minimum distance constraint
    points = []
    min_dist = 0.15
    attempts = 0
    while len(points) < N and attempts < 10000:
        attempts += 1
        u = rng.random()
        v = rng.random()
        if u + v <= 1:
            p = np.array([u + 0.5 * v, (np.sqrt(3) / 2) * v])
            # Check minimum distance to existing points
            ok = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    ok = False
                    break
            if ok:
                points.append(p)
    # If we couldn't place all, just fill randomly
    while len(points) < N:
        u = rng.random()
        v = rng.random()
        if u + v <= 1:
            p = np.array([u + 0.5 * v, (np.sqrt(3) / 2) * v])
            points.append(p)
    return np.array(points[:N])

def _local_nudge(points_cart, rng, max_iter=500, step_size=0.02):
    """Local search: nudge points involved in smallest triangles."""
    n = len(points_cart)
    best_area = _min_triangle_area(points_cart)
    current = points_cart.copy()
    
    for it in range(max_iter):
        # Find the smallest triangle
        min_area = np.inf
        min_idx = None
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1, p2, p3 = current[i], current[j], current[k]
                    area = 0.5 * abs(np.cross(p2 - p1, p3 - p1))
                    if area < min_area:
                        min_area = area
                        min_idx = (i, j, k)
        
        if min_area >= best_area:
            break
        
        # Try to move one of the three points to increase this triangle's area
        i, j, k = min_idx
        improved = False
        for idx in [i, j, k]:
            # Try random nudges
            for _ in range(20):
                delta = np.random.uniform(-step_size, step_size, 2)
                new_p = current[idx] + delta
                # Check if still in triangle
                bary = _cart_to_bary(np.array([new_p]))
                if bary[0, 0] >= 0 and bary[0, 1] >= 0 and bary[0, 0] + bary[0, 1] <= 1:
                    new_points = current.copy()
                    new_points[idx] = new_p
                    new_area = _min_triangle_area(new_points)
                    if new_area > best_area:
                        best_area = new_area
                        current = new_points
                        improved = True
                        break
            if improved:
                break
        if not improved:
            # Try smaller step
            step_size *= 0.8
            if step_size < 0.001:
                break
    
    return current, best_area

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s
    
    rng = np.random.RandomState(42)
    
    # Initial configuration using stratified/random sampling
    init_cart = _sobol_init(rng)
    best_cart = init_cart.copy()
    best_area = _min_triangle_area(best_cart)
    
    # Local optimization using Nelder-Mead on barycentric coordinates
    bary_init = _cart_to_bary(init_cart)
    
    # Multi-start local optimization
    n_restarts = 0
    while time.time() < deadline:
        n_restarts += 1
        if n_restarts > 50:
            break
        
        # Perturb initial points
        if n_restarts == 1:
            bary_start = bary_init.copy()
        else:
            bary_start = bary_init + rng.normal(0, 0.1, (N, 2))
            # Clip to feasible region
            bary_start = np.clip(bary_start, 0, None)
            bary_start[:, 0] = np.minimum(bary_start[:, 0], 1 - bary_start[:, 1])
        
        # Flatten for optimization
        x0 = bary_start.flatten()
        
        # Use Nelder-Mead
        try:
            result = minimize(
                _objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
        except Exception:
            continue
        
        x_opt = result.x.reshape(N, 2)
        # Ensure feasibility
        x_opt = np.clip(x_opt, 0, None)
        x_opt[:, 0] = np.minimum(x_opt[:, 0], 1 - x_opt[:, 1])
        
        cart_opt = _bary_to_cart(x_opt)
        area_opt = _min_triangle_area(cart_opt)
        
        if area_opt > best_area:
            best_area = area_opt
            best_cart = cart_opt
        
        # Also try local nudge from this result
        nudged_cart, nudged_area = _local_nudge(cart_opt, rng, max_iter=200, step_size=0.01)
        if nudged_area > best_area:
            best_area = nudged_area
            best_cart = nudged_cart
    
    # Final local nudge from best
    nudged_cart, nudged_area = _local_nudge(best_cart, rng, max_iter=300, step_size=0.005)
    if nudged_area > best_area:
        best_area = nudged_area
        best_cart = nudged_cart
    
    return best_cart
# EVOLVE-BLOCK-END
