import numpy as np
from scipy.optimize import differential_evolution, minimize
from itertools import combinations
import time

N = 11
TIME_LIMIT = 142.0  # Safety margin for 150s limit

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

def cart_to_bary(x, y):
    """Convert Cartesian to barycentric (u, v) where w = 1-u-v."""
    w = y / H
    v = x - 0.5 * w
    u = 1.0 - w - v
    return [u, v]

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

def hard_min_obj(bary_flat):
    """Objective using hard min."""
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def soft_min_obj(bary_flat, beta=100.0):
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

def generate_hexagonal_init():
    """
    Generate a hexagonal-like lattice initialization.
    For 11 points, use a hexagonal packing arrangement.
    """
    candidates = []
    
    pts = []
    for i in range(4):
        pts.append([i / 3.0, 0.0])
    for i in range(3):
        pts.append([(i + 0.5) / 3.0, H / 3.0])
    for i in range(2):
        pts.append([(i + 1.0) / 3.0, 2 * H / 3.0])
    pts.append([0.5, H])
    
    pts = np.array(pts)
    candidates.append(pts)
    
    rng = np.random.default_rng(123)
    for _ in range(3):
        perturbed = pts + rng.normal(0, 0.02, size=pts.shape)
        perturbed = clamp_bary(perturbed)
        candidates.append(perturbed)
    
    return candidates

def generate_k3_lattice_plus():
    """
    Generate K=3 lattice (10 pts) + 11th point.
    """
    candidates = []
    K = 3
    lattice_pts = []
    for i in range(K + 1):
        for j in range(K + 1 - i):
            u = i / K
            v = j / K
            lattice_pts.append([u, v])
    
    lattice_pts = np.array(lattice_pts)
    
    add_points = [
        [0.5, 0.25],
        [0.25, 0.5],
        [0.25, 0.25],
        [0.4, 0.4],
        [0.3, 0.3]
    ]
    
    for ap in add_points:
        pts = list(lattice_pts)
        pts.append(ap)
        candidates.append(np.array(pts))
        
    return candidates

def generate_k4_lattice_minus():
    """
    Generate K=4 lattice (15 pts) and remove 4 points.
    """
    K = 4
    lattice_pts = []
    for i in range(K + 1):
        for j in range(K + 1 - i):
            u = i / K
            v = j / K
            lattice_pts.append([u, v])
            
    lattice_pts = np.array(lattice_pts)
    
    interior_idx = []
    for i, pt in enumerate(lattice_pts):
        u, v = pt
        w = 1 - u - v
        if u > 0.01 and v > 0.01 and w > 0.01:
            interior_idx.append(i)
            
    remove_idx = set(interior_idx)
    remove_idx.add(0)
    
    if len(remove_idx) < 4:
        remove_idx.add(1)
        
    keep_idx = [i for i in range(len(lattice_pts)) if i not in remove_idx]
    kept_pts = lattice_pts[keep_idx]
    
    return [kept_pts]

def generate_random_init(rng, n_candidates=5):
    """Generate random initial configurations."""
    candidates = []
    for _ in range(n_candidates):
        u = rng.random(N)
        v = rng.random(N) * (1 - u)
        candidates.append(np.column_stack([u, v]))
    return candidates

def generate_boundary_interior_init():
    """
    Generate configurations with some points on the boundary and some interior.
    Optimal Heilbronn configurations often have points on edges.
    """
    candidates = []
    
    # Strategy 1: 3 corners + 4 edge-midpoints + 4 interior
    corners = np.array([[0,0], [1,0], [0.5, H]])
    edge_pts = [
        [0.25, 0], [0.75, 0],
        [0.125, H*0.25], [0.375, H*0.5],
        [0.625, H*0.5], [0.875, H*0.25],
    ]
    
    corner_bary = np.array([cart_to_bary(x, y) for x, y in corners])
    edge_bary = np.array([cart_to_bary(x, y) for x, y in edge_pts])
    
    interior_bary = np.array([
        [1/3, 1/3],
        [0.2, 0.3], [0.5, 0.2], [0.3, 0.4]
    ])
    
    pts = np.vstack([corner_bary, edge_bary[:4], interior_bary[:4]])
    candidates.append(pts)
    
    # Strategy 2: 3 corners + 3 edge points + 5 interior
    edge_pts2 = [
        [0.3, 0], [0.7, 0],
        [0.2, H*0.4],
    ]
    edge_bary2 = np.array([cart_to_bary(x, y) for x, y in edge_pts2])
    interior_bary2 = np.array([
        [1/3, 1/3],
        [0.15, 0.25], [0.6, 0.15], [0.35, 0.45], [0.5, 0.3]
    ])
    pts2 = np.vstack([corner_bary, edge_bary2, interior_bary2])
    candidates.append(pts2)
    
    # Strategy 3: 6 boundary + 5 interior
    boundary_bary = []
    for t in [0.1, 0.4, 0.6, 0.9]:
        boundary_bary.append([0, t])
    for u_val in [0.3, 0.7]:
        boundary_bary.append([u_val, 0])
    
    interior_bary3 = np.array([
        [1/3, 1/3],
        [0.2, 0.35], [0.55, 0.2], [0.35, 0.4], [0.5, 0.35]
    ])
    
    boundary_arr = np.array(boundary_bary)
    pts3 = np.vstack([boundary_arr, interior_bary3])
    candidates.append(pts3)
    
    # Strategy 4: Symmetric edge quarters + interior ring
    edge_quarter_bary = []
    edge_quarter_bary.append([0, 0.25])
    edge_quarter_bary.append([0, 0.75])
    edge_quarter_bary.append([0.25, 0])
    edge_quarter_bary.append([0.75, 0])
    edge_quarter_bary.append([0.25, 0.75])
    edge_quarter_bary.append([0.75, 0.25])
    
    interior_bary4 = []
    interior_bary4.append([1/3, 1/3])
    for angle in range(4):
        theta = 2*np.pi*angle/4 + np.pi/4
        r = 0.15
        du = r * np.cos(theta)
        dv = r * np.sin(theta)
        interior_bary4.append([1/3 + du, 1/3 + dv])
    
    edge_q_arr = np.array(edge_quarter_bary)
    interior_arr4 = np.array(interior_bary4)
    pts4 = np.vstack([edge_q_arr, interior_arr4])
    candidates.append(pts4)
    
    return candidates

def generate_perturbed_best(rng, base_bary, n_perturb=3, scale=0.05):
    """Generate perturbed versions of a good configuration."""
    candidates = []
    for _ in range(n_perturb):
        perturbed = base_bary + rng.normal(0, scale, size=base_bary.shape)
        candidates.append(clamp_bary(perturbed))
    return candidates

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using multiple optimization stages with active-set acceleration.
    """
    start_time = time.time()
    bounds = [(0, 1)] * (N * 2)
    
    bary_current = bary_init.copy()
    current_val = -hard_min_obj(bary_current.flatten())
    
    # Stage 1: Smooth soft-min minimization with L-BFGS-B
    try:
        result1 = minimize(
            soft_min_obj,
            bary_current.flatten(),
            args=(100.0,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        new_bary = clamp_bary(result1.x.reshape(N, 2))
        new_val = -hard_min_obj(new_bary.flatten())
        if new_val > current_val:
            bary_current = new_bary
            current_val = new_val
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.3:
        return bary_current
        
    # Stage 2: Nelder-Mead on all points
    try:
        result2 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        new_bary = clamp_bary(result2.x.reshape(N, 2))
        new_val = -hard_min_obj(new_bary.flatten())
        if new_val > current_val:
            bary_current = new_bary
            current_val = new_val
    except Exception:
        pass
        
    if time.time() - start_time > time_limit * 0.7:
        return bary_current
    
    # Stage 3: SLSQP polish
    try:
        result3 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-14}
        )
        new_bary = clamp_bary(result3.x.reshape(N, 2))
        new_val = -hard_min_obj(new_bary.flatten())
        if new_val > current_val:
            bary_current = new_bary
            current_val = new_val
    except Exception:
        pass
        
    # Stage 4: Coordinated ascent - move each point to maximize min area
    if time.time() - start_time < time_limit * 0.9:
        try:
            for _ in range(2):
                for i in range(N):
                    if time.time() - start_time > time_limit * 0.95:
                        break
                    def single_point_obj(xi):
                        bary_temp = bary_current.copy()
                        bary_temp[i] = xi
                        bary_temp = clamp_bary(bary_temp)
                        areas = calc_areas_bary(bary_temp)
                        return -np.min(areas)
                    
                    res_i = minimize(
                        single_point_obj,
                        bary_current[i],
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                    )
                    new_bary = bary_current.copy()
                    new_bary[i] = clamp_bary(res_i.x.reshape(1, 2))[0]
                    new_val = -hard_min_obj(new_bary.flatten())
                    if new_val > current_val:
                        bary_current = new_bary
                        current_val = new_val
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

    candidates = []
    
    # 1. Hexagonal lattice based candidates
    candidates.extend(generate_hexagonal_init())
    
    # 2. K=3 Lattice based candidates
    candidates.extend(generate_k3_lattice_plus())
    
    # 3. K=4 Lattice based candidates
    candidates.extend(generate_k4_lattice_minus())
    
    # 4. Boundary-interior mixed candidates
    candidates.extend(generate_boundary_interior_init())
    
    # 5. Random candidates
    rng = np.random.default_rng(42)
    candidates.extend(generate_random_init(rng, n_candidates=4))

    # Process candidates with time allocation
    n_cands = len(candidates)
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 25:
            break
            
        remaining = deadline - 25 - time.time()
        if remaining < 3:
            break
            
        if i < 4:
            t_limit = min(7.0, remaining)
        elif i < 9:
            t_limit = min(5.0, remaining)
        elif i < 10:
            t_limit = min(5.0, remaining)
        elif i < 14:
            t_limit = min(6.0, remaining)
        else:
            t_limit = min(3.0, remaining)
            
        try:
            refined = refine_local(cand, time_limit=t_limit)
            update_best(refined)
        except Exception:
            continue

    # Generate perturbed versions of best if time permits
    if best_bary is not None and time.time() < deadline - 30:
        rng2 = np.random.default_rng(999)
        perturb_cands = generate_perturbed_best(rng2, best_bary, n_perturb=3, scale=0.03)
        for pc in perturb_cands:
            if time.time() > deadline - 25:
                break
            remaining = deadline - 25 - time.time()
            if remaining < 3:
                break
            try:
                refined_p = refine_local(pc, time_limit=min(5.0, remaining))
                update_best(refined_p)
            except Exception:
                continue

    # Global Search with Differential Evolution if time permits
    if time.time() < deadline - 40:
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
                maxiter=25,
                popsize=12,
                tol=1e-6,
                polish=False
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            remaining = deadline - 25 - time.time()
            if remaining > 5:
                refined_de = refine_local(de_bary, time_limit=min(8.0, remaining))
                update_best(refined_de)
        except Exception:
            pass

    # Second DE run with different seed if time permits
    if time.time() < deadline - 30:
        try:
            def de_obj(x):
                bary = x.reshape(N, 2)
                bary = clamp_bary(bary)
                areas = calc_areas_bary(bary)
                return -np.min(areas)
            
            bounds = [(0, 1)] * (N * 2)
            
            result_de2 = differential_evolution(
                de_obj,
                bounds,
                seed=123,
                maxiter=15,
                popsize=10,
                tol=1e-6,
                polish=False
            )
            
            de_bary2 = clamp_bary(result_de2.x.reshape(N, 2))
            remaining = deadline - 20 - time.time()
            if remaining > 4:
                refined_de2 = refine_local(de_bary2, time_limit=min(6.0, remaining))
                update_best(refined_de2)
        except Exception:
            pass

    # Final polish on best solution
    if best_bary is not None and time.time() < deadline - 5:
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
        
        if best_bary is not None and time.time() < deadline - 2:
            try:
                result_nm = minimize(
                    hard_min_obj,
                    best_bary.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
                )
                update_best(clamp_bary(result_nm.x.reshape(N, 2)))
            except Exception:
                pass

    if best_bary is None:
        best_bary = generate_hexagonal_init()[0]

    cart = bary_to_cart(best_bary)
    
    return cart
