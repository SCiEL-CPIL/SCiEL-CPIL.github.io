import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 140.0  # Safety margin for 150s limit

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2
AREA_TOTAL = np.sqrt(3) / 4.0

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

def get_min_area(bary):
    """Return the minimum triangle area for a configuration."""
    areas = calc_areas_bary(bary)
    return np.min(areas)

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
            # Try different subsets
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

def generate_structured_inits():
    """
    Generate diverse structured initial configurations inspired by known Heilbronn results.
    For n=11, use boundary-heavy arrangements with interior points.
    """
    candidates = []
    
    # Config 1: 3 corners, 4 edge quarters, 4 interior
    cfg1 = np.array([
        [0.0, 0.0],   # corner A
        [1.0, 0.0],   # corner B
        [0.0, 1.0],   # corner C
        [0.25, 0.0],  # edge AB quarter
        [0.75, 0.0],  # edge AB three-quarter
        [0.0, 0.25],  # edge AC quarter
        [0.0, 0.75],  # edge AC three-quarter
        [0.25, 0.75], # edge BC
        [0.75, 0.25], # edge BC
        [1/3, 1/3],   # centroid
        [0.3, 0.3],   # interior
    ])
    candidates.append(clamp_bary(cfg1))
    
    # Config 2: 3 corners, 3 edge midpoints, 5 interior (symmetric)
    cfg2 = np.array([
        [0.0, 0.0],   # corner
        [1.0, 0.0],   # corner
        [0.0, 1.0],   # corner
        [0.5, 0.0],   # edge midpoint AB
        [0.0, 0.5],   # edge midpoint AC
        [0.5, 0.5],   # edge midpoint BC
        [1/3, 1/3],   # centroid
        [0.2, 0.3],   # interior
        [0.5, 0.2],   # interior
        [0.3, 0.5],   # interior
        [0.4, 0.4],   # interior
    ])
    candidates.append(clamp_bary(cfg2))
    
    # Config 3: Hexagonal-inspired
    cfg3 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.33, 0.0],
        [0.67, 0.0],
        [0.0, 0.33],
        [0.0, 0.67],
        [0.17, 0.5],
        [0.5, 0.17],
        [0.33, 0.33],
        [0.5, 0.33],
    ])
    candidates.append(clamp_bary(cfg3))
    
    # Config 4: Dense boundary
    cfg4 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.2, 0.0],
        [0.8, 0.0],
        [0.0, 0.2],
        [0.0, 0.8],
        [0.1, 0.9],
        [0.9, 0.1],
        [1/3, 1/3],
        [0.4, 0.35],
    ])
    candidates.append(clamp_bary(cfg4))
    
    # Config 5: Perturbed hexagonal
    rng = np.random.default_rng(42)
    base_hex = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.0, 0.25],
        [0.0, 0.75],
        [0.25, 0.5],
        [0.5, 0.25],
        [1/3, 1/3],
        [0.4, 0.35],
    ])
    for _ in range(3):
        perturbed = base_hex + rng.normal(0, 0.03, size=base_hex.shape)
        candidates.append(clamp_bary(perturbed))
    
    return candidates

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

def coordinated_ascent(bary_init, time_limit=15.0, max_passes=20):
    """
    Coordinated per-point ascent: sequentially optimize each point's position
    while others are fixed, targeting the active constraints (smallest triangles).
    """
    start_time = time.time()
    bary_current = bary_init.copy()
    current_min_area = get_min_area(bary_current)
    
    for pass_num in range(max_passes):
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        for i in range(N):
            if time.time() - start_time > time_limit * 0.95:
                break
                
            # Fix all other points, optimize point i
            bary_fixed = bary_current.copy()
            
            def obj_i(pt_i):
                """Objective: minimize negative min area when point i is at pt_i."""
                bary_test = bary_fixed.copy()
                bary_test[i] = pt_i
                bary_test = clamp_bary(bary_test)
                min_a = get_min_area(bary_test)
                return -min_a
            
            # Initial position
            pt_init = bary_current[i].copy()
            
            # Bounds for barycentric coords of a single point
            bounds_i = [(0, 1), (0, 1)]
            
            try:
                result_i = minimize(
                    obj_i,
                    pt_init,
                    method='L-BFGS-B',
                    bounds=bounds_i,
                    options={'maxiter': 30, 'ftol': 1e-10}
                )
                
                new_pt = clamp_bary(result_i.x.reshape(1, 2))[0]
                bary_test = bary_fixed.copy()
                bary_test[i] = new_pt
                bary_test = clamp_bary(bary_test)
                new_min_area = get_min_area(bary_test)
                
                if new_min_area > current_min_area + 1e-12:
                    bary_current = bary_test
                    current_min_area = new_min_area
                    improved = True
            except Exception:
                pass
        
        if not improved:
            break
    
    return bary_current, current_min_area

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
        
    # Stage 2: Coordinated per-point ascent
    if time.time() - start_time < time_limit * 0.6:
        bary_current, _ = coordinated_ascent(bary_current, time_limit=time_limit * 0.2)
    
    if time.time() - start_time > time_limit * 0.7:
        return bary_current
    
    # Stage 3: Hard min refinement with Nelder-Mead
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
    
    # Stage 4: SLSQP polish
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

    # 1. Try Hexagonal Init
    hex_init = generate_hexagonal_init()
    refined_hex = refine_local(hex_init, time_limit=8.0)
    update_best(refined_hex)
    
    # 2. Global Search with Differential Evolution
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
    
    # 3. Multi-start: structured + random + perturbed restarts with local refinement
    rng = np.random.default_rng(123)
    structured = generate_structured_inits()
    
    # Add some random inits
    for _ in range(3):
        structured.append(generate_random_init(seed=1000 + _))
    
    restart_count = 0
    for cand in structured:
        if time.time() > deadline - 15:
            break
        
        remaining_time = deadline - 15 - time.time()
        if remaining_time < 3:
            break
            
        try:
            refined = refine_local(cand, time_limit=min(5.0, remaining_time))
            update_best(refined)
        except Exception:
            pass
        
        restart_count += 1
        
        # Also try perturbed versions of the best solution
        if best_bary is not None and time.time() < deadline - 15:
            for sigma in [0.02, 0.04]:
                if time.time() > deadline - 15:
                    break
                perturbed = generate_perturbed_init(best_bary, sigma=sigma, seed=2000 + restart_count * 10 + int(sigma * 100))
                try:
                    refined_p = refine_local(perturbed, time_limit=min(4.0, deadline - 15 - time.time()))
                    update_best(refined_p)
                except Exception:
                    pass
    
    # Final high-precision polish on the best candidate
    if best_bary is not None and time.time() < deadline - 5:
        try:
            result_nm = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
            )
            update_best(clamp_bary(result_nm.x.reshape(N, 2)))
        except Exception:
            pass
        
        # One more SLSQP polish
        if time.time() < deadline - 3:
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
        
    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
