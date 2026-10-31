import numpy as np
from scipy.optimize import minimize
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

def get_min_area(bary):
    """Get the minimum area for a barycentric configuration."""
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return np.min(areas)

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

def generate_k_lattice(k):
    """Generate a k-lattice in barycentric coordinates."""
    pts = []
    for i in range(k + 1):
        for j in range(k - i + 1):
            l = k - i - j
            if l < 0: continue
            u = i / k
            v = j / k
            # w = l/k
            pts.append([u, v])
    return np.array(pts)

def get_candidate_1():
    """
    K=3 lattice (10 points) + 1 point in a 'hole'.
    """
    k3 = generate_k_lattice(3)
    new_pt = np.array([1/9, 1/9])
    return np.vstack([k3, new_pt])

def get_candidate_2():
    """
    K=4 lattice (15 points). Remove 4 points.
    """
    k4 = generate_k_lattice(4)
    to_remove = [
        np.array([0.25, 0.25]),
        np.array([0.25, 0.5]),
        np.array([0.5, 0.25]),
        np.array([0.5, 0.0])
    ]
    mask = np.ones(len(k4), dtype=bool)
    for pt in to_remove:
        dists = np.linalg.norm(k4 - pt, axis=1)
        idx = np.argmin(dists)
        mask[idx] = False
        
    return k4[mask]

def get_candidate_3():
    """
    Hexagonal packing approximation.
    """
    k3 = generate_k_lattice(3)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.05, k3.shape)
    new_pt = np.array([0.5, 0.5]) + rng.normal(0, 0.05, 2)
    pts = np.vstack([k3 + noise, [new_pt]])
    return clamp_bary(pts)

def get_candidate_random(seed):
    """
    Random start with repulsion.
    """
    rng = np.random.default_rng(seed)
    pts = np.zeros((N, 2))
    for i in range(N):
        while True:
            u = rng.random()
            v = rng.random() * (1 - u)
            pt = np.array([u, v])
            if i == 0:
                break
            dists = np.linalg.norm(pts[:i] - pt, axis=1)
            if np.min(dists) > 0.1:
                break
        pts[i] = pt
    return pts

def coordinated_ascent(bary_init, time_limit=20.0):
    """
    Coordinated per-point ascent: sequentially optimize each point's position
    while others are fixed, targeting the active constraints (smallest triangles).
    """
    start_time = time.time()
    bary_current = bary_init.copy()
    current_min_area = get_min_area(bary_current)
    
    # Find which triangles are "active" (near the minimum)
    def find_active_triangles(bary, threshold=0.01):
        """Find triangles with area within threshold of the minimum."""
        areas = calc_areas_bary(bary)
        min_area = np.min(areas)
        active = np.where(areas < min_area + threshold)[0]
        return active
    
    # Per-point optimization using scipy minimize on just that point
    for iteration in range(50):  # Max iterations
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        for i in range(N):
            if time.time() - start_time > time_limit:
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
                    options={'maxiter': 50, 'ftol': 1e-10}
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

def refine_local(bary_init, time_limit=20.0):
    """
    Refine a configuration using coordinated ascent followed by global polish.
    """
    start_time = time.time()
    bary_current = bary_init.copy()
    
    # Stage 1: Coordinated per-point ascent
    t1_limit = time_limit * 0.6
    bary_current, min_area_1 = coordinated_ascent(bary_current, time_limit=t1_limit)
    
    if time.time() - start_time > time_limit * 0.8:
        return bary_current
        
    # Stage 2: Global Nelder-Mead polish
    try:
        result_nm = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-14, 'adaptive': True}
        )
        bary_current = clamp_bary(result_nm.x.reshape(N, 2))
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
        val = get_min_area(bary)
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    # Generate candidates
    candidates = [
        get_candidate_1(),
        get_candidate_2(),
        get_candidate_3(),
        get_candidate_random(1),
        get_candidate_random(2),
        get_candidate_random(3)
    ]

    # Allocate time
    time_per_candidate = (TIME_LIMIT - 15) / len(candidates)

    for i, cand in enumerate(candidates):
        if time.time() > deadline - 15:
            break
            
        remaining_time = deadline - 15 - time.time()
        t_limit = min(time_per_candidate, remaining_time)
        
        if t_limit < 5:
            break
            
        try:
            refined = refine_local(cand, time_limit=t_limit)
            update_best(refined)
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
        
    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart

# EVOLVE-BLOCK-END
