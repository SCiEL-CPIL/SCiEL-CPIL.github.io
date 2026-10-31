import numpy as np
from scipy.optimize import minimize
import time
from itertools import combinations

N = 11
TIME_LIMIT = 115.0

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
    Minimizing this pushes ALL areas up, with emphasis on the smallest.
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

def generate_symmetric_inits():
    """
    Generate several symmetric initial configurations.
    1. Hexagonal lattice subset
    2. Center + 2 rings
    3. Random with symmetry constraints
    """
    inits = []
    
    # 1. Hexagonal lattice centered
    # Center (1/3, 1/3)
    # 6 points around center
    # 4 points further out?
    # Let's try a few scales of a hexagonal grid
    
    center = np.array([1.0/3, 1.0/3])
    
    for r1 in [0.1, 0.15, 0.2, 0.25]:
        for r2 in [0.3, 0.4, 0.5]:
            pts = [center]
            # Ring 1: 6 points
            for k in range(6):
                angle = np.pi/3 * k + np.pi/6 # Offset to align with triangle axes?
                # In barycentric, rotation is tricky. 
                # Easier to work in Cartesian for init, then convert?
                # Or just use a known good pattern.
                pass
            
    # Let's just use a few random starts and a few specific structured ones.
    # Structured 1: 3 vertices, 3 midpoints, 1 center, 4 others?
    # Structured 2: Hexagonal close packing subset.
    
    # Simple random starts
    rng = np.random.default_rng(42)
    for i in range(10):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        inits.append(np.column_stack([u, v]))
        
    # Specific symmetric start:
    # 1 center
    # 6 points in a hexagon
    # 4 points in a square?
    # Let's try a "flower" pattern in Cartesian, then map to Barycentric.
    # But we are optimizing in Barycentric.
    
    # Let's add a start based on the known optimal for N=7 (hexagon+center) 
    # and add 4 points near the vertices?
    
    # Center
    p0 = [1/3, 1/3]
    # Hexagon radius 0.2
    hex_pts = []
    for k in range(6):
        theta = np.pi/3 * k
        # Approximate barycentric coords for a point at distance r from center
        # This is complex. Let's just use random + refine.
        pass
        
    return inits

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Smooth soft-min minimization with increasing beta
    for beta in [10.0, 50.0, 200.0]:
        try:
            result1 = minimize(
                soft_min_obj,
                bary_current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if result1.success or result1.fun < -hard_min_obj(bary_current.flatten()):
                bary_current = clamp_bary(result1.x.reshape(N, 2))
        except Exception:
            pass
        
        if time.time() - start_time > time_limit * 0.5:
            break
            
    # Stage 2: Hard min refinement with SLSQP (handles kinks better than NM sometimes)
    if time.time() - start_time < time_limit * 0.8:
        try:
            result2 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-16}
            )
            bary_current = clamp_bary(result2.x.reshape(N, 2))
        except Exception:
            pass
            
    # Stage 3: Nelder-Mead for final polish
    if time.time() - start_time < time_limit:
        try:
            result3 = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
            )
            bary_current = clamp_bary(result3.x.reshape(N, 2))
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

    # Generate diverse initial points
    rng = np.random.default_rng(123)
    inits = []
    
    # 1. Random starts
    for i in range(20):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        inits.append(np.column_stack([u, v]))
        
    # 2. Structured starts: Hexagonal grid
    # Try to place points in a hexagonal pattern
    for scale in [0.15, 0.2, 0.25]:
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
                    
        if len(pts) >= N:
            # Pick N points from this grid, trying to maximize spread
            # Simple heuristic: pick every k-th point or random subset
            for _ in range(5):
                if len(pts) == N:
                    bary = np.array(pts)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts)[idx]
                inits.append(bary)

    # Refine each initial point
    for init in inits:
        if time.time() > deadline - 10:
            break
            
        # Allocate time based on remaining time
        time_per_init = min(3.0, (deadline - time.time()) / max(1, len(inits)))
        refined = refine_local(init, time_limit=time_per_init)
        update_best(refined)
        
    # Final high-precision polish on the best found
    if best_bary is not None and time.time() < deadline - 5:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16}
            )
            update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
        except Exception:
            pass
        
        if best_bary is not None and time.time() < deadline - 2:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-15, 'fatol': 1e-19, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
