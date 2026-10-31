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
    """Clamp barycentric coordinates to the valid triangle region."""
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1.0 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def hard_min_obj(bary_flat):
    """Objective: negative of min triangle area (to minimize)."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=50.0):
    """Smooth surrogate: negative log-sum-exp of areas."""
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

def get_min_area(bary):
    """Return the minimum triangle area for a configuration."""
    areas = calc_areas_bary(bary)
    return np.min(areas)

def generate_heilbronn_init():
    """
    Generate initial configurations inspired by known Heilbronn triangle results.
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
        [0.25, 0.5],  # edge BC
        [0.5, 0.25],  # edge BC
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
        [0.25, 0.75], # edge midpoint BC
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
    
    # Config 5-7: Perturbed hexagonal
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

def coordinated_ascent(bary_init, time_limit=5.0, max_passes=10):
    """
    Coordinated ascent: iteratively move each point to maximize the minimum area,
    focusing on active (near-minimum) triangles.
    """
    start_time = time.time()
    bary = bary_init.copy()
    current_min = get_min_area(bary)
    
    for pass_num in range(max_passes):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(N):
            if time.time() - start_time > time_limit * 0.95:
                break
            
            # Find which triangles involve point i and are active
            areas = calc_areas_bary(bary)
            min_area = np.min(areas)
            active = np.where(areas <= min_area * 1.05)[0]
            
            # Triangles involving point i
            tri_with_i = np.where((tri_idx[:, 0] == i) | (tri_idx[:, 1] == i) | 
                                  (tri_idx[:, 2] == i))[0]
            
            # Active triangles involving point i
            active_with_i = np.intersect1d(active, tri_with_i)
            
            if len(active_with_i) == 0:
                continue
            
            # Objective: maximize the minimum area among active triangles involving i
            # We want to move point i to increase these areas
            
            def point_obj(offset_flat):
                new_bary = bary.copy()
                new_bary[i, 0] += offset_flat[0]
                new_bary[i, 1] += offset_flat[1]
                new_bary = clamp_bary(new_bary)
                
                new_areas = calc_areas_bary(new_bary)
                # Only consider triangles involving point i
                tri_areas_i = new_areas[tri_with_i]
                # Focus on the minimum among active triangles
                if len(active_with_i) > 0:
                    active_areas_i = new_areas[active_with_i]
                    return -np.min(active_areas_i)
                else:
                    return -np.min(tri_areas_i)
            
            # Try to find a better position for point i
            try:
                result = minimize(
                    point_obj,
                    np.array([0.0, 0.0]),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                
                new_bary = bary.copy()
                new_bary[i, 0] += result.x[0]
                new_bary[i, 1] += result.x[1]
                new_bary = clamp_bary(new_bary)
                
                new_min = get_min_area(new_bary)
                if new_min > current_min * (1 + 1e-8):
                    bary = new_bary
                    current_min = new_min
                    improved = True
            except Exception:
                pass
        
        if not improved:
            break
    
    return bary

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
    
    if time.time() - start_time > time_limit * 0.3:
        # Use remaining time for coordinated ascent
        remaining = time_limit - (time.time() - start_time)
        if remaining > 1.0:
            bary_current = coordinated_ascent(bary_current, time_limit=remaining)
        return bary_current
        
    # Stage 2: Coordinated ascent
    try:
        bary_current = coordinated_ascent(bary_current, time_limit=time_limit * 0.4)
    except Exception:
        pass
        
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
        
    if time.time() - start_time > time_limit * 0.9:
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

    # 1. Start with multiple Heilbronn-inspired templates
    inits = generate_heilbronn_init()
    
    for init_bary in inits:
        update_best(init_bary)
    
    # 2. Refine each initial configuration
    if best_bary is not None:
        for init_bary in inits:
            if time.time() > deadline - 20:
                break
            remaining = min(8.0, deadline - start_time - 15.0)
            if remaining > 2.0:
                refined = refine_local(init_bary, time_limit=remaining)
                update_best(refined)
    
    # 3. Additional random restarts
    if best_bary is not None:
        remaining = deadline - start_time - 10.0
        if remaining > 10:
            rng = np.random.default_rng(999)
            for restart in range(4):
                if time.time() > deadline - 8:
                    break
                
                # Generate random or perturbed initial
                if restart % 2 == 0:
                    # Random
                    u = rng.random(N)
                    v = rng.random(N) * (1 - u)
                    init_bary = np.column_stack([u, v])
                else:
                    # Perturb best
                    perturbed = best_bary + rng.normal(0, 0.05, size=best_bary.shape)
                    init_bary = clamp_bary(perturbed)
                
                remaining_time = min(5.0, deadline - start_time - 8.0)
                if remaining_time > 1.0:
                    refined = refine_local(init_bary, time_limit=remaining_time)
                    update_best(refined)
    
    # 4. Final high-precision polish
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
        
        if best_bary is not None and time.time() < deadline - 3:
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
