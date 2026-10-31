import numpy as np
from itertools import combinations
import time
from scipy.optimize import minimize

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

def clamp_bary(bary, eps=1e-10):
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

def min_area(bary):
    """Compute the minimum triangle area for a configuration."""
    areas = calc_areas_bary(bary)
    return np.min(areas)

def generate_hexagonal_init():
    """Generate a hexagonal lattice-like initial configuration."""
    best_bary = None
    best_val = -np.inf
    
    for scale in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
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
            for _ in range(10):
                if len(pts) == N:
                    bary = np.array(pts)
                else:
                    idx = rng.choice(len(pts), N, replace=False)
                    bary = np.array(pts)[idx]
                val = min_area(bary)
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

def generate_concentric_init():
    """Generate points in concentric triangular rings."""
    # Center of the equilateral triangle in barycentric coords is (1/3, 1/3)
    center = np.array([1/3, 1/3])
    pts = [center]
    
    # Add rings
    for r in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        # 6 points per ring roughly
        for i in range(6):
            angle = 2 * np.pi * i / 6
            # Approximate circle in barycentric space
            # This is a heuristic, might need clamping
            u = center[0] + r * np.cos(angle)
            v = center[1] + r * np.sin(angle)
            pts.append([u, v])
            
    # Ensure we have exactly N points
    pts = np.array(pts[:N])
    return clamp_bary(pts)

def epigraph_obj(x):
    """
    Objective function for epigraph formulation.
    x = [bary_coords (N*2), t]
    We want to maximize t subject to areas >= t.
    So we minimize -t subject to t - area_i <= 0.
    """
    bary = x[:2*N].reshape(N, 2)
    t = x[2*N]
    
    # Check validity of barycentric coords
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    if np.any(u < -1e-6) or np.any(v < -1e-6) or np.any(w < -1e-6):
        return 1e6  # Penalty for invalid barycentric coords
        
    # Calculate areas
    areas = calc_areas_bary(bary)
    
    # Constraints: t - area_i <= 0 for all i
    # In scipy minimize, constraints are of form g(x) >= 0
    # So we define g_i = area_i - t >= 0
    # But for the objective, we just return -t.
    # The constraints are handled separately.
    return -t

def epigraph_constraints(x):
    """
    Constraints for epigraph formulation.
    Returns a list of constraint dicts.
    """
    bary = x[:2*N].reshape(N, 2)
    t = x[2*N]
    
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Constraint 1: Barycentric coords within [0, 1]
    # u >= 0, v >= 0, u+v <= 1
    # We handle bounds in minimize, but let's add them as constraints too for safety if needed?
    # Actually, bounds are better for SLSQP.
    
    # Constraint 2: t <= area_i for all i
    # area_i - t >= 0
    areas = calc_areas_bary(bary)
    cons = areas - t
    
    return [{'type': 'ineq', 'fun': lambda x, c=cons: c}]

def refine_epigraph(bary_init, time_limit=5.0):
    """
    Refine a configuration using epigraph formulation with SLSQP.
    """
    start_time = time.time()
    
    # Initial t estimate
    t0 = min_area(bary_init) * 0.9
    
    # Initial point
    x0 = np.concatenate([bary_init.flatten(), [t0]])
    
    # Bounds for barycentric coords
    # u, v in [0, 1]
    # t in [0, 1] (max area is 1)
    bounds = []
    for _ in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
    bounds.append((0, 1))
    
    try:
        result = minimize(
            epigraph_obj,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=epigraph_constraints(x0),
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        
        if result.success:
            bary_opt = result.x[:2*N].reshape(N, 2)
            t_opt = result.x[2*N]
            bary_opt = clamp_bary(bary_opt)
            val = min_area(bary_opt)
            return bary_opt, val
        else:
            # If it fails, return the best point found or the init
            bary_opt = result.x[:2*N].reshape(N, 2)
            bary_opt = clamp_bary(bary_opt)
            val = min_area(bary_opt)
            return bary_opt, val
            
    except Exception as e:
        # Fallback to init if exception
        return bary_init, min_area(bary_init)

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages.
    """
    start_time = time.time()
    
    bary_current = bary_init.copy()
    current_val = min_area(bary_current)
    
    # Stage 1: Epigraph Optimization
    try:
        bary_ep, val_ep = refine_epigraph(bary_current, time_limit * 0.4)
        if val_ep > current_val:
            bary_current = bary_ep
            current_val = val_ep
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.6:
        return bary_current, current_val
        
    # Stage 2: SLSQP on min-area directly (polish)
    try:
        bounds = [(0, 1)] * (N * 2)
        result2 = minimize(
            lambda x: -min_area(x.reshape(N, 2)),
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-14}
        )
        candidate = clamp_bary(result2.x.reshape(N, 2))
        val = min_area(candidate)
        if val > current_val:
            bary_current = candidate
            current_val = val
    except Exception:
        pass
        
    # Stage 3: Nelder-Mead (polish)
    if time.time() - start_time < time_limit:
        try:
            result3 = minimize(
                lambda x: -min_area(x.reshape(N, 2)),
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
            )
            candidate = clamp_bary(result3.x.reshape(N, 2))
            val = min_area(candidate)
            if val > current_val:
                bary_current = candidate
                current_val = val
        except Exception:
            pass

    return bary_current, current_val

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # 1. Generate diverse initializations
    inits = []
    
    # Hexagonal
    hex_init = generate_hexagonal_init()
    inits.append(hex_init)
    
    # Random
    for i in range(5):
        inits.append(generate_random_init(seed=1000 + i))
    
    # Concentric
    inits.append(generate_concentric_init())
    
    # Perturbed hex
    rng = np.random.default_rng(42)
    for i in range(5):
        noise = rng.normal(0, 0.05, hex_init.shape)
        inits.append(clamp_bary(hex_init + noise))
        
    # 2. Refine each initialization
    for init in inits:
        if time.time() > deadline - 10:
            break
            
        # Estimate time budget for this refinement
        # We have ~130s total. Let's give each refinement up to 10s, but stop if time runs out.
        # Actually, let's just run them and check time.
        refined, val = refine_local(init, time_limit=10.0)
        update_best(refined)
        
    # 3. If time remains, do a final high-precision polish on the best found
    if best_bary is not None and time.time() < deadline - 5:
        try:
            # One more SLSQP pass with tighter tolerance
            bounds = [(0, 1)] * (N * 2)
            result_final = minimize(
                lambda x: -min_area(x.reshape(N, 2)),
                best_bary.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16}
            )
            update_best(clamp_bary(result_final.x.reshape(N, 2)))
        except Exception:
            pass
        
        # One more Nelder-Mead pass
        if time.time() < deadline - 2:
            try:
                result_nm = minimize(
                    lambda x: -min_area(x.reshape(N, 2)),
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-18, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
