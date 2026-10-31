import numpy as np
from scipy.optimize import minimize
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

def calc_areas_bary(bary):
    """Calculate areas of all triangles formed by N points in barycentric coords."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

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

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=50.0):
    """
    Smooth surrogate for the hard min via log-sum-exp.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    
    if np.any(areas < 1e-15):
        return 1e10
    
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    
    if max_scaled > 700:
        return -np.min(areas)
        
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    return -soft_min

def generate_hexagonal_init(scale=None, seed=42):
    """
    Generate a hexagonal lattice-like initial configuration at a given scale.
    """
    if scale is None:
        scale = 0.25
    
    pts = []
    max_k = int(1.0 / scale) + 1
    for k in range(max_k + 1):
        w_val = k * scale
        if w_val > 1.0:
            break
        num_pts_in_row = int((1 - w_val) / scale) + 1
        for j in range(num_pts_in_row):
            u_val = j * scale
            v_val = (1 - w_val) - u_val
            if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                pts.append([u_val, v_val])
    
    if len(pts) < N:
        # Fallback to random if not enough points
        rng = np.random.default_rng(seed)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        return np.column_stack([u, v])
    
    # Select N points from the lattice
    rng = np.random.default_rng(seed)
    if len(pts) == N:
        bary = np.array(pts)
    else:
        idx = rng.choice(len(pts), N, replace=False)
        bary = np.array(pts)[idx]
    
    return bary

def generate_centroid_init(seed=42):
    """
    Generate initial points based on centroid subdivision.
    Divide the triangle into smaller triangles and place points at centroids.
    """
    rng = np.random.default_rng(seed)
    
    # Start with the three vertices and centroid
    points = [
        [0.0, 0.0],      # V0
        [1.0, 0.0],      # V1
        [0.0, 1.0],      # V2 in barycentric
        [1/3, 1/3]       # Centroid
    ]
    
    # Add more points by subdividing
    while len(points) < N:
        # Pick a random existing point and add a perturbed version
        base = points[rng.integers(len(points))]
        noise = rng.normal(0, 0.15, 2)
        new_pt = base + noise
        new_pt = clamp_bary(np.array([new_pt]))[0]
        points.append(new_pt)
    
    bary = np.array(points[:N])
    return clamp_bary(bary)

def generate_edge_distributed_init(seed=42):
    """
    Distribute points along edges and interior.
    """
    rng = np.random.default_rng(seed)
    
    # Place 3 points near each vertex
    pts = []
    for _ in range(3):
        # Near V0: small u, v
        u = rng.uniform(0.01, 0.2)
        v = rng.uniform(0.01, 0.2)
        pts.append([u, v])
    
    for _ in range(3):
        # Near V1: large u
        u = rng.uniform(0.8, 0.99)
        v = rng.uniform(0.01, 0.2)
        pts.append([u, v])
    
    for _ in range(3):
        # Near V2: large v
        u = rng.uniform(0.01, 0.2)
        v = rng.uniform(0.8, 0.99)
        pts.append([u, v])
    
    # Add interior points
    for _ in range(N - len(pts)):
        u = rng.uniform(0.2, 0.6)
        v = rng.uniform(0.2, 0.6)
        if u + v <= 0.8:
            pts.append([u, v])
        else:
            u = rng.uniform(0.1, 0.4)
            v = rng.uniform(0.1, 0.4)
            pts.append([u, v])
    
    bary = np.array(pts[:N])
    return clamp_bary(bary)

def coordinate_descent_refine(bary_init, max_iter=100, time_limit=5.0):
    """
    Specialized coordinate descent: optimize each point's (u,v) while fixing others.
    This is much faster than full Nelder-Mead/SLSQP in 22D.
    """
    start_time = time.time()
    bary = bary_init.copy()
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(N):
            # Save original position
            orig_u, orig_v = bary[i, 0], bary[i, 1]
            
            # Try small perturbations in u and v directions
            best_u, best_v = orig_u, orig_v
            best_val = -hard_min_obj(bary.flatten())
            
            # Radial search: try several step sizes
            for step in [0.01, 0.005, 0.002, 0.001, -0.01, -0.005, -0.002, -0.001]:
                # Perturb u
                test_bary = bary.copy()
                test_bary[i, 0] = orig_u + step
                test_bary = clamp_bary(test_bary)
                val_u = -hard_min_obj(test_bary.flatten())
                if val_u > best_val:
                    best_val = val_u
                    best_u = test_bary[i, 0]
                    best_v = test_bary[i, 1]
                
                # Perturb v
                test_bary = bary.copy()
                test_bary[i, 1] = orig_v + step
                test_bary = clamp_bary(test_bary)
                val_v = -hard_min_obj(test_bary.flatten())
                if val_v > best_val:
                    best_val = val_v
                    best_u = test_bary[i, 0]
                    best_v = test_bary[i, 1]
                
                # Perturb both
                test_bary = bary.copy()
                test_bary[i, 0] = orig_u + step
                test_bary[i, 1] = orig_v + step
                test_bary = clamp_bary(test_bary)
                val_both = -hard_min_obj(test_bary.flatten())
                if val_both > best_val:
                    best_val = val_both
                    best_u = test_bary[i, 0]
                    best_v = test_bary[i, 1]
            
            # Update if improved
            if best_u != orig_u or best_v != orig_v:
                bary[i, 0] = best_u
                bary[i, 1] = best_v
                improved = True
        
        if not improved:
            break
    
    return clamp_bary(bary)

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using coordinate descent + SLSQP polish.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Coordinate descent (fast, many iterations)
    cd_limit = time_limit * 0.6
    bary_cd = coordinate_descent_refine(bary_current, max_iter=200, time_limit=cd_limit)
    if -hard_min_obj(bary_cd.flatten()) > -hard_min_obj(bary_current.flatten()):
        bary_current = bary_cd
    
    if time.time() - start_time > time_limit * 0.7:
        return bary_current
        
    # Stage 2: SLSQP polish
    try:
        result = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-14}
        )
        bary_polished = clamp_bary(result.x.reshape(N, 2))
        if -hard_min_obj(bary_polished.flatten()) > -hard_min_obj(bary_current.flatten()):
            bary_current = bary_polished
    except Exception:
        pass
        
    return bary_current

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # Generate diverse structured initializations
    inits = []
    
    # Hexagonal at multiple scales
    for scale in [0.2, 0.25, 0.3, 0.35, 0.4]:
        inits.append(generate_hexagonal_init(scale=scale, seed=42 + int(scale * 100)))
    
    # Centroid-based
    inits.append(generate_centroid_init(seed=100))
    
    # Edge-distributed
    inits.append(generate_edge_distributed_init(seed=200))
    
    # Random
    rng = np.random.default_rng(300)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    inits.append(np.column_stack([u, v]))
    
    # Refine each initialization
    for i, init in enumerate(inits):
        if time.time() > deadline - 10:
            break
        remaining = deadline - 10 - time.time()
        if remaining < 2:
            break
        t_limit = min(8.0, remaining / max(1, len(inits) - i))
        refined = refine_local(init, time_limit=t_limit)
        update_best(refined)
    
    # Multi-start with perturbations of best
    restart_count = 0
    while time.time() < deadline - 8:
        restart_count += 1
        if best_bary is not None and restart_count % 2 == 0:
            # Perturbation of best solution
            rng = np.random.default_rng(5000 + restart_count)
            noise = rng.normal(0, 0.03, best_bary.shape)
            init = clamp_bary(best_bary + noise)
        else:
            # Random restart
            rng = np.random.default_rng(6000 + restart_count)
            u = rng.random(N)
            v = rng.random(N) * (1 - u)
            init = np.column_stack([u, v])
        
        remaining = deadline - 8 - time.time()
        if remaining < 2:
            break
            
        t_limit = min(6.0, remaining)
        refined = refine_local(init, time_limit=t_limit)
        update_best(refined)

    # Final high-precision polish on the best candidate
    if best_bary is not None and time.time() < deadline - 3:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 1:
            # Final coordinate descent polish
            bary_final = coordinate_descent_refine(best_bary, max_iter=100, time_limit=1.5)
            update_best(bary_final)

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
