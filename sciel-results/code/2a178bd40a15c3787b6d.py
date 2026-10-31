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
    """Objective: negative of min triangle area (to minimize)."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=50.0):
    """
    Smooth surrogate: negative log-sum-exp of areas.
    Minimizing this encourages all areas to be large.
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

def generate_random_inits(rng, n_candidates=5):
    """Generate random initial configurations."""
    candidates = []
    for _ in range(n_candidates):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        candidates.append(np.column_stack([u, v]))
    return candidates

def electrostatic_relax(bary_init, time_limit=10.0, beta=30.0):
    """
    Physical simulation-style relaxation: treat points as charges repelling each other
    in a potential field derived from the log-sum-exp of triangle areas.
    This naturally distributes points to maximize minimal separation in area space.
    """
    start_time = time.time()
    bary = bary_init.copy()
    current_min = get_min_area(bary)
    
    # Adaptive step size
    step_size = 0.05
    patience = 0
    max_patience = 50
    
    for iteration in range(500):
        if time.time() - start_time > time_limit:
            break
        
        # Compute forces: gradient of soft_min_obj
        # Use numerical gradient
        eps = 1e-5
        forces = np.zeros_like(bary)
        
        for i in range(N):
            for j in range(2):
                bary_plus = bary.copy()
                bary_plus[i, j] += eps
                bary_plus = clamp_bary(bary_plus)
                
                bary_minus = bary.copy()
                bary_minus[i, j] -= eps
                bary_minus = clamp_bary(bary_minus)
                
                f_plus = soft_min_obj(bary_plus.flatten(), beta=beta)
                f_minus = soft_min_obj(bary_minus.flatten(), beta=beta)
                
                forces[i, j] = (f_plus - f_minus) / (2 * eps)
        
        # Normalize forces
        force_mag = np.linalg.norm(forces)
        if force_mag > 1e-10:
            forces /= force_mag
        
        # Update positions
        bary_new = bary - step_size * forces
        bary_new = clamp_bary(bary_new)
        
        new_min = get_min_area(bary_new)
        
        if new_min > current_min + 1e-12:
            bary = bary_new
            current_min = new_min
            patience = 0
            # Slowly decrease step size
            step_size *= 0.98
        else:
            patience += 1
            step_size *= 0.9
            
            if patience > max_patience:
                break
    
    return bary, current_min

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

def refine_local(bary_init, time_limit=20.0):
    """
    Refine a configuration using electrostatic relaxation, coordinated ascent,
    and global Nelder-Mead polish.
    """
    start_time = time.time()
    bary_current = bary_init.copy()
    
    # Stage 1: Electrostatic relaxation
    t1_limit = time_limit * 0.3
    if time.time() - start_time < time_limit * 0.7:
        bary_current, _ = electrostatic_relax(bary_current, time_limit=t1_limit)
    
    # Stage 2: Coordinated per-point ascent
    t2_limit = time_limit * 0.4
    if time.time() - start_time < time_limit * 0.6:
        bary_current, _ = coordinated_ascent(bary_current, time_limit=t2_limit)
    
    # Stage 3: Global Nelder-Mead polish
    if time.time() - start_time < time_limit * 0.8:
        try:
            result_nm = minimize(
                hard_min_obj,
                bary_current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-14, 'adaptive': True}
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

    # Generate diverse candidates
    rng = np.random.default_rng(123)
    structured = generate_structured_inits()
    randoms = generate_random_inits(rng, n_candidates=4)
    candidates = structured + randoms

    # Allocate time per candidate
    time_per_candidate = (TIME_LIMIT - 20) / len(candidates)

    for i, cand in enumerate(candidates):
        if time.time() > deadline - 20:
            break
            
        remaining_time = deadline - 20 - time.time()
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
