import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import time

N = 11
TIME_LIMIT = 115.0

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)

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

def generate_boundary_init():
    """
    Generate initial configuration with points distributed along the boundary
    of the equilateral triangle, plus one center point.
    """
    # Place 10 points along the boundary, 1 at center
    # Boundary: 3 sides, distribute points evenly
    
    # Side 1: (0,0) to (1,0) in barycentric: u=0, v=t, w=1-t for t in [0,1]
    # Side 2: (1,0) to (0.5, sqrt(3)/2): u=1-t, v=t*0.5, w=t*0.5... 
    # Let's use barycentric directly.
    
    # Vertices in barycentric:
    # V0: (1, 0)
    # V1: (0, 1)
    # V2: (0, 0)
    
    # We want to place points along the edges.
    # Edge V0-V1: (1-t, t, 0) for t in [0,1]
    # Edge V1-V2: (0, 1-t, t) for t in [0,1]
    # Edge V2-V0: (t, 0, 1-t) for t in [0,1]
    
    # Let's place 4 points on each edge (excluding vertices to avoid degeneracy, or including?)
    # Actually, for 11 points: maybe 3 on each edge + 2 interior? Or 4+4+3 on edges?
    # Let's try: 4 on edge 1, 4 on edge 2, 3 on edge 3, 1 center.
    
    pts = []
    
    # Edge V0-V1 (bottom): (1-t, t, 0)
    # t values: avoid 0 and 1 to stay strictly inside? No, boundary is allowed.
    # But let's keep them slightly inside for numerical stability, or use exact boundary.
    # The problem says "on or inside". Boundary is fine.
    # Let's place them at t = 0.1, 0.3, 0.5, 0.7
    for t in [0.1, 0.3, 0.5, 0.7]:
        pts.append([1-t, t])
        
    # Edge V1-V2 (right): (0, 1-t, t)
    # Barycentric (u,v): u=0, v=1-t, w=t
    for t in [0.1, 0.3, 0.5, 0.7]:
        pts.append([0, 1-t])
        
    # Edge V2-V0 (left): (t, 0, 1-t)
    # Barycentric (u,v): u=t, v=0, w=1-t
    for t in [0.1, 0.5, 0.9]:
        pts.append([t, 0])
        
    # Center
    pts.append([1/3, 1/3])
    
    bary = np.array(pts)
    return clamp_bary(bary)

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    """
    best_bary = None
    best_val = -np.inf
    
    for scale in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
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
            rng = np.random.default_rng(42)
            for _ in range(5):
                if len(pts) == N:
                    bary = np.array(pts)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts)[idx]
                val = -hard_min_obj(bary.flatten())
                if val > best_val:
                    best_val = val
                    best_bary = bary.copy()
                
    if best_bary is None:
        rng = np.random.default_rng(42)
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        best_bary = np.column_stack([u, v])
        
    return best_bary

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base_bary, sigma=0.05, seed=None):
    """Generate a perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, base_bary.shape)
    return clamp_bary(base_bary + noise)

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    
    # Stage 1: Smooth soft-min minimization
    try:
        result1 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(50.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if result1.success or result1.fun < -hard_min_obj(bary_current.flatten()):
            bary_current = clamp_bary(result1.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.4:
        return bary_current
        
    # Stage 2: Hard min refinement with Nelder-Mead
    try:
        result2 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.8:
        return bary_current
    
    # Stage 3: SLSQP polish
    try:
        result3 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-14}
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

    # 1. Try Boundary Init (new approach)
    boundary_init = generate_boundary_init()
    refined_boundary = refine_local(boundary_init, time_limit=8.0)
    update_best(refined_boundary)
    
    # 2. Try Hexagonal Init
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=8.0)
    update_best(refined_hex)
    
    # 3. Global Search with Differential Evolution
    if time.time() < deadline - 35:
        try:
            def de_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                areas = calc_areas_bary(bary)
                return -np.min(areas)
            
            bounds = [(0, 1)] * (N * 2)
            result_de = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=30,
                popsize=12,
                tol=1e-6,
                polish=False
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=8.0)
            update_best(refined_de)
        except Exception:
            pass
    
    # 4. Multi-start: random + perturbed restarts with local refinement
    rng = np.random.default_rng(123)
    restart_count = 0
    while time.time() < deadline - 12:
        restart_count += 1
        if best_bary is not None and restart_count % 2 == 0:
            # Perturbation of best solution
            init = generate_perturbed_init(best_bary, sigma=0.04, seed=2000 + restart_count)
        else:
            # Fresh random start
            init = generate_random_init(seed=3000 + restart_count)
        
        refined = refine_local(init, time_limit=3.0)
        update_best(refined)

    # 5. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 5:
        try:
            bounds = [(0, 1)] * (N * 2)
            result_slsqp = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15}
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
                    options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass
        
        # One more SLSQP pass
        if best_bary is not None and time.time() < deadline - 0.5:
            try:
                bounds = [(0, 1)] * (N * 2)
                result_final = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-16}
                )
                update_best(clamp_bary(result_final.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
