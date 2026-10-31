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
    
    # Hexagonal lattice points within the triangle
    # Use a grid with hexagonal offset
    pts = []
    
    # Row 0: 4 points along the bottom
    for i in range(4):
        pts.append([i / 3.0, 0.0])
    
    # Row 1: 3 points offset
    for i in range(3):
        pts.append([(i + 0.5) / 3.0, H / 3.0])
    
    # Row 2: 2 points
    for i in range(2):
        pts.append([(i + 1.0) / 3.0, 2 * H / 3.0])
    
    # Row 3: 1 point at top
    pts.append([0.5, H])
    
    pts = np.array(pts)
    candidates.append(pts)
    
    # Add some variations with slight perturbations
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
    
    # Add 11th point at various positions
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
    We remove the 3 interior points and 1 corner point to maximize spread.
    """
    K = 4
    lattice_pts = []
    for i in range(K + 1):
        for j in range(K + 1 - i):
            u = i / K
            v = j / K
            lattice_pts.append([u, v])
            
    lattice_pts = np.array(lattice_pts)
    
    # Find interior points
    interior_idx = []
    for i, pt in enumerate(lattice_pts):
        u, v = pt
        w = 1 - u - v
        if u > 0.01 and v > 0.01 and w > 0.01:
            interior_idx.append(i)
            
    # Remove interior points
    remove_idx = set(interior_idx)
    
    # Remove 1 corner point, e.g., index 0 (0,0)
    remove_idx.add(0)
    
    # We need to remove exactly 4. If we have 3 interior, we remove 3+1.
    # If we have more interior, remove 4 interior.
    # For K=4, interior count is 3.
    if len(remove_idx) < 4:
        # Add another corner
        remove_idx.add(1) # (1,0)
        
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
    # Edge midpoints (skip corners)
    edge_pts = [
        [0.25, 0], [0.75, 0],       # bottom edge
        [0.125, H*0.25], [0.375, H*0.5],  # left edge (parametric)
        [0.625, H*0.5], [0.875, H*0.25],  # right edge (parametric)
    ]
    # Convert to barycentric
    def cart_to_bary_correct(x, y):
        w = y / H
        v = x - 0.5 * w
        u = 1 - w - v
        return [u, v]
    
    corner_bary = np.array([cart_to_bary_correct(x, y) for x, y in corners])
    edge_bary = np.array([cart_to_bary_correct(x, y) for x, y in edge_pts])
    
    # Interior points - use centroid-based placement
    interior_bary = np.array([
        [1/3, 1/3],  # centroid
        [0.2, 0.3], [0.5, 0.2], [0.3, 0.4]
    ])
    
    pts = np.vstack([corner_bary, edge_bary[:4], interior_bary[:4]])
    # We need exactly 11: 3 corners + 4 edges + 4 interior = 11
    candidates.append(pts)
    
    # Strategy 2: 3 corners + 3 edge points + 5 interior (more spread)
    edge_pts2 = [
        [0.3, 0], [0.7, 0],
        [0.2, H*0.4],
    ]
    edge_bary2 = np.array([cart_to_bary_correct(x, y) for x, y in edge_pts2])
    interior_bary2 = np.array([
        [1/3, 1/3],
        [0.15, 0.25], [0.6, 0.15], [0.35, 0.45], [0.5, 0.3]
    ])
    pts2 = np.vstack([corner_bary, edge_bary2, interior_bary2])
    candidates.append(pts2)
    
    # Strategy 3: No corners, all on boundary + interior mix
    # 6 boundary points evenly spaced + 5 interior
    boundary_bary = []
    # Bottom edge: t from 0 to 1, u=0, v=t
    for t in [0.1, 0.4, 0.6, 0.9]:
        boundary_bary.append([0, t])
    # Left edge: parametrize from (0,0) to (0.5, H)
    # In bary: u goes 0->1, v=0
    for u_val in [0.3, 0.7]:
        boundary_bary.append([u_val, 0])
        
    interior_bary3 = np.array([
        [1/3, 1/3],
        [0.2, 0.35], [0.55, 0.2], [0.35, 0.4], [0.5, 0.35]
    ])
    
    boundary_arr = np.array(boundary_bary)
    pts3 = np.vstack([boundary_arr, interior_bary3])
    candidates.append(pts3)
    
    # Strategy 4: Symmetric hexagonal with boundary emphasis
    # Place points at 1/4 and 3/4 positions on each edge
    edge_quarter_bary = []
    # Bottom: (0.25,0), (0.75,0) -> bary [0, 0.25], [0, 0.75]
    edge_quarter_bary.append([0, 0.25])
    edge_quarter_bary.append([0, 0.75])
    # Left edge from (0,0) to top: parametrize u from 0 to 1, v=0
    # Point at 1/4: u=0.25, v=0
    edge_quarter_bary.append([0.25, 0])
    edge_quarter_bary.append([0.75, 0])
    # Right edge from (1,0) to top: parametrize
    # From (1,0): u=0,v=1; to (0.5,H): u=1,v=0
    # At 1/4: u=0.25, v=0.75
    edge_quarter_bary.append([0.25, 0.75])
    edge_quarter_bary.append([0.75, 0.25])
    
    # 5 interior points in pentagon pattern
    interior_bary4 = []
    # Center
    interior_bary4.append([1/3, 1/3])
    # Ring around center
    for angle in range(4):
        theta = 2*np.pi*angle/4 + np.pi/4
        r = 0.15
        # In barycentric, "circular" is tricky; use small offsets
        du = r * np.cos(theta)
        dv = r * np.sin(theta)
        interior_bary4.append([1/3 + du, 1/3 + dv])
        
    edge_q_arr = np.array(edge_quarter_bary)
    interior_arr4 = np.array(interior_bary4)
    pts4 = np.vstack([edge_q_arr, interior_arr4])
    candidates.append(pts4)
    
    return candidates

def generate_symmetric_init():
    """
    Generate configurations with 3-fold rotational symmetry.
    11 points: 1 at centroid, 3 orbits of 3 points, plus 1 extra.
    Total 1 + 9 = 10. We need 11.
    So: 1 at centroid, 3 orbits of 3, plus 1 extra on symmetry axis.
    """
    candidates = []
    
    centroid = np.array([1.0/3, 1.0/3])
    
    def rotate_120(bary_pt):
        u, v = bary_pt
        w = 1 - u - v
        return np.array([w, u])
        
    def rotate_240(bary_pt):
        u, v = bary_pt
        w = 1 - u - v
        return np.array([v, w])
        
    # Try varying radii for the 3 orbits
    radii_options = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    
    for r1 in radii_options:
        for r2 in radii_options:
            if r1 >= r2: continue
            for r3 in radii_options:
                if r2 >= r3: continue
                
                pts = [centroid.copy()]
                
                # Orbit 1
                ref1 = np.array([1/3 + r1, 1/3 - r1/2])
                ref1 = clamp_bary(ref1.reshape(1,2))[0]
                pts.extend([ref1, rotate_120(ref1), rotate_240(ref1)])
                
                # Orbit 2
                ref2 = np.array([1/3 + r2, 1/3 - r2/2])
                ref2 = clamp_bary(ref2.reshape(1,2))[0]
                pts.extend([ref2, rotate_120(ref2), rotate_240(ref2)])
                
                # Orbit 3
                ref3 = np.array([1/3 + r3, 1/3 - r3/2])
                ref3 = clamp_bary(ref3.reshape(1,2))[0]
                pts.extend([ref3, rotate_120(ref3), rotate_240(ref3)])
                
                # We have 10 points. Add 1 more on the vertical axis of symmetry (u=1/3)
                # Try different positions for the 11th point
                for extra_y in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    extra = np.array([1/3, 1/3 + extra_y])
                    extra = clamp_bary(extra.reshape(1,2))[0]
                    all_pts = np.array(pts + [extra])
                    candidates.append(all_pts)
                    
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
        
    # Stage 2: Active-set Nelder-Mead - only optimize points that constrain the min
    try:
        # Find which triangles give the minimum area
        areas = calc_areas_bary(bary_current)
        min_area = np.min(areas)
        # Points involved in near-minimum triangles
        active_mask = areas < min_area * 1.05
        active_triangles = np.where(active_mask)[0]
        active_points = set()
        for ti in active_triangles:
            active_points.add(tri_idx[ti, 0])
            active_points.add(tri_idx[ti, 1])
            active_points.add(tri_idx[ti, 2])
        
        # Full Nelder-Mead on all points
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
            for _ in range(2):  # Two passes
                for i in range(N):
                    if time.time() - start_time > time_limit * 0.95:
                        break
                    # Optimize point i while fixing others
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
    
    # 1. Hexagonal lattice based candidates (primary)
    candidates.extend(generate_hexagonal_init())
    
    # 2. K=3 Lattice based candidates (secondary)
    candidates.extend(generate_k3_lattice_plus())
    
    # 3. K=4 Lattice based candidates
    candidates.extend(generate_k4_lattice_minus())
    
    # 4. Boundary-interior mixed candidates (new strategy)
    candidates.extend(generate_boundary_interior_init())
    
    # 5. Symmetric 3-fold candidates (from inspiration)
    candidates.extend(generate_symmetric_init())
    
    # 6. Random candidates
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
            
        # Allocate time based on candidate type
        if i < 4:  # Hexagonal candidates
            t_limit = min(7.0, remaining)
        elif i < 9:  # K=3 candidates
            t_limit = min(5.0, remaining)
        elif i < 10: # K=4 candidate
            t_limit = min(5.0, remaining)
        elif i < 14: # Boundary-interior candidates
            t_limit = min(6.0, remaining)
        elif i < 14 + len(generate_symmetric_init()): # Symmetric candidates
            t_limit = min(4.0, remaining)
        else:  # Random candidates
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
            
            # Run differential evolution with moderate settings
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
        # Fallback
        best_bary = generate_hexagonal_init()[0]

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
