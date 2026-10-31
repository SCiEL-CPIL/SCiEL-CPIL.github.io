import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import time

N = 11
TIME_LIMIT = 145.0

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2
CENTROID = np.array([0.5, np.sqrt(3) / 6])

def bary_to_cart(bary):
    """Convert barycentric coords (u, v) to Cartesian (x, y)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def cart_to_bary(cart):
    """Convert Cartesian (x, y) to barycentric coords (u, v) where w = 1-u-v."""
    x = cart[:, 0]
    y = cart[:, 1]
    # Barycentric: p = u*V0 + v*V1 + w*V2, u+v+w=1
    # V0=(0,0), V1=(1,0), V2=(0.5, H)
    # x = v + 0.5*w, y = H*w
    # w = y/H, v = x - 0.5*w, u = 1 - v - w
    w = y / H
    v = x - 0.5 * w
    u = 1.0 - v - w
    return np.column_stack([u, v])

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

def get_min_area_and_indices(bary):
    """Get minimum area and which triangle achieves it."""
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    min_idx = np.argmin(areas)
    return areas[min_idx], tri_idx[min_idx]

def active_triangles(bary, tol=1e-6):
    """Get indices of triangles with area close to the minimum."""
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    min_area = np.min(areas)
    return np.where(areas <= min_area + tol)[0]

def soft_min_obj(bary_flat, beta=50.0):
    """Smooth surrogate for the hard min via log-sum-exp."""
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

def project_to_triangle(cart):
    """Project a point onto the triangle (clamp to boundary if outside)."""
    bary = cart_to_bary(cart)
    bary = clamp_bary(bary)
    return bary_to_cart(bary)

def generate_boundary_init(seed=42):
    """
    Generate points distributed on/near the boundary of the triangle.
    Uses the geometric insight that optimal max-min configurations 
    often have points on the boundary.
    """
    rng = np.random.default_rng(seed)
    bary = np.zeros((N, 2))
    
    # Distribute points along the three edges
    # Edge 0-1: w=0, u+v=1 -> bary = (u, 1-u)
    # Edge 1-2: u=0, v+w=1 -> bary = (0, v)  
    # Edge 2-0: v=0, u+w=1 -> bary = (u, 0)
    
    # 3 points near vertices (slightly inside)
    bary[0] = [0.97, 0.015]  # near V0
    bary[1] = [0.015, 0.97]  # near V1
    bary[2] = [0.015, 0.015] # near V2
    
    # 4 points on edges (2 per edge pair, avoiding vertices)
    bary[3] = [0.5, 0.49]    # edge 0-1
    bary[4] = [0.01, 0.49]   # edge 1-2
    bary[5] = [0.49, 0.01]   # edge 2-0
    bary[6] = [0.25, 0.74]   # edge 0-1 closer to V1
    
    # 3 interior points forming a smaller rotated triangle
    angle_offset = rng.uniform(0, 2*np.pi)
    for i in range(3):
        angle = angle_offset + 2*np.pi*i/3
        r = 0.18
        cx = 1/3 + r * np.cos(angle)
        cy = 1/3 + r * np.sin(angle) * 0.577  # aspect ratio for equilateral
        bary[7+i] = [cx, cy]
    
    # 1 more interior point
    bary[10] = [1/3 + rng.uniform(-0.05, 0.05), 1/3 + rng.uniform(-0.05, 0.05)]
    
    return clamp_bary(bary)

def generate_radial_init(seed=42):
    """
    Generate points using radial/angular parameterization from centroid.
    Points are placed at various angles and distances from centroid.
    """
    rng = np.random.default_rng(seed)
    bary = np.zeros((N, 2))
    
    # 1 point at centroid
    bary[0] = [1/3, 1/3]
    
    # 10 points in 5 pairs at different angles, varying radii
    for i in range(5):
        angle = 2*np.pi*i/5 + rng.uniform(-0.3, 0.3)
        r1 = rng.uniform(0.15, 0.45)
        r2 = rng.uniform(0.2, 0.5)
        
        # Convert from "polar" in barycentric space
        # In barycentric, centroid is (1/3, 1/3)
        # Moving in direction (cos(angle), sin(angle)) in Cartesian,
        # then converting back to barycentric
        cx = 1/3 + r1 * np.cos(angle)
        cy = 1/3 + r1 * np.sin(angle) * 0.577
        bary[2*i+1] = [cx, cy]
        
        cx2 = 1/3 + r2 * np.cos(angle + np.pi/5)
        cy2 = 1/3 + r2 * np.sin(angle + np.pi/5) * 0.577
        bary[2*i+2] = [cx2, cy2]
    
    return clamp_bary(bary)

def generate_symmetric_init():
    """
    Generate a 3-fold symmetric configuration.
    """
    bary = np.zeros((N, 2))
    
    # 3 points near vertices
    bary[0] = [0.96, 0.02]
    bary[1] = [0.02, 0.96]
    bary[2] = [0.02, 0.02]
    
    # 6 points in 2 rings of 3, symmetric
    for ring, (r, offset) in enumerate([(0.25, 0.0), (0.42, np.pi/3)]):
        for i in range(3):
            angle = offset + 2*np.pi*i/3
            cx = 1/3 + r * np.cos(angle)
            cy = 1/3 + r * np.sin(angle) * 0.577
            bary[3 + ring*3 + i] = [cx, cy]
    
    # 1 center
    bary[9] = [1/3, 1/3]
    
    # 1 extra
    bary[10] = [0.38, 0.38]
    
    return clamp_bary(bary)

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

def coordinated_ascent(bary_init, max_iter=200, time_limit=5.0):
    """
    Coordinated ascent: for each point, try to move it to increase 
    the minimum area, focusing on active (minimum-area) triangles.
    """
    start_time = time.time()
    bary = bary_init.copy()
    current_val = -hard_min_obj(bary.flatten())
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        
        for i in range(N):
            # Get active triangles involving point i
            min_area, min_tri = get_min_area_and_indices(bary)
            
            # Try small moves in each direction
            best_move = None
            best_area = min_area
            
            # Try radial moves (toward/away from centroid) and angular moves
            directions = []
            # Radial in barycentric: move toward centroid or away
            centroid_bary = np.array([1/3, 1/3])
            radial_dir = centroid_bary - bary[i]
            radial_dir /= (np.linalg.norm(radial_dir) + 1e-10)
            directions.append(radial_dir)
            directions.append(-radial_dir)
            
            # Angular: perpendicular to radial
            perp_dir = np.array([-radial_dir[1], radial_dir[0]])
            directions.append(perp_dir)
            directions.append(-perp_dir)
            
            # Coordinate moves
            directions.append(np.array([1.0, 0.0]))
            directions.append(np.array([-1.0, 0.0]))
            directions.append(np.array([0.0, 1.0]))
            directions.append(np.array([0.0, -1.0]))
            
            # Try different step sizes
            for step in [0.01, 0.03, 0.05, 0.08, 0.12]:
                for d in directions:
                    new_bary = bary.copy()
                    new_bary[i] = bary[i] + step * d
                    new_bary = clamp_bary(new_bary)
                    
                    new_min_area, _ = get_min_area_and_indices(new_bary)
                    
                    if new_min_area > best_area + 1e-10:
                        best_area = new_min_area
                        best_move = new_bary.copy()
            
            if best_move is not None:
                bary = best_move
                current_val = best_area
                improved = True
        
        if not improved:
            break
    
    return bary

def refine_local(bary_init, time_limit=5.0):
    """
    Refine a configuration using coordinated ascent + local optimization.
    """
    start_time = time.time()
    
    bary_current = bary_init.copy()
    
    # Stage 1: Coordinated ascent
    try:
        ca_time = min(3.0, time_limit * 0.5)
        bary_ca = coordinated_ascent(bary_current, max_iter=150, time_limit=ca_time)
        val_ca = -hard_min_obj(bary_ca.flatten())
        val_cur = -hard_min_obj(bary_current.flatten())
        if val_ca > val_cur:
            bary_current = bary_ca
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.7:
        return bary_current
    
    # Stage 2: Nelder-Mead on hard min
    try:
        result2 = minimize(
            hard_min_obj,
            bary_current.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-16, 'adaptive': True}
        )
        bary_current = clamp_bary(result2.x.reshape(N, 2))
    except Exception:
        pass
    
    if time.time() - start_time > time_limit * 0.9:
        return bary_current
    
    # Stage 3: One more coordinated ascent pass
    try:
        remaining = max(0.5, time_limit - (time.time() - start_time))
        bary_ca2 = coordinated_ascent(bary_current, max_iter=100, time_limit=remaining)
        val_ca2 = -hard_min_obj(bary_ca2.flatten())
        val_cur2 = -hard_min_obj(bary_current.flatten())
        if val_ca2 > val_cur2:
            bary_current = bary_ca2
    except Exception:
        pass
    
    return bary_current

def basin_hopping_step(bary_current, temp=0.05, seed=None):
    """
    Perform a basin hopping step: random perturbation + local refinement.
    Accept if better or with Metropolis probability.
    """
    rng = np.random.default_rng(seed)
    
    # Perturb
    noise = rng.normal(0, temp, bary_current.shape)
    bary_perturbed = clamp_bary(bary_current + noise)
    
    # Local refine
    bary_refined = refine_local(bary_perturbed, time_limit=3.0)
    
    return bary_refined

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

    # 1. Try multiple structured initializations
    inits = []
    
    # Boundary-focused init
    inits.append(generate_boundary_init(seed=42))
    
    # Radial init
    inits.append(generate_radial_init(seed=42))
    
    # Symmetric init
    inits.append(generate_symmetric_init())
    
    # Random inits
    for seed in [1, 7, 13]:
        inits.append(generate_random_init(seed))
    
    for init in inits:
        refined = refine_local(init, time_limit=4.0)
        update_best(refined)
    
    # 2. Basin hopping with adaptive temperature
    if best_bary is not None and time.time() < deadline - 30:
        temp = 0.08
        bh_count = 0
        while time.time() < deadline - 25:
            bh_count += 1
            remaining = deadline - 25 - time.time()
            if remaining < 5:
                break
            
            # Adaptive temperature: decrease if stuck, increase if improving
            if bh_count > 0 and bh_count % 5 == 0:
                temp *= 0.9  # cool down
            
            bary_new = basin_hopping_step(
                best_bary, 
                temp=temp, 
                seed=1000 + bh_count
            )
            
            if update_best(bary_new):
                temp = max(0.02, temp * 1.1)  # heat up on improvement
            
            # Occasionally try from a random start
            if bh_count % 7 == 0 and time.time() < deadline - 30:
                random_init = generate_random_init(seed=2000 + bh_count)
                refined_random = refine_local(random_init, time_limit=3.0)
                update_best(refined_random)
    
    # 3. Differential evolution for global search
    if time.time() < deadline - 25:
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
                maxiter=40,
                popsize=20,
                tol=1e-6,
                polish=False
            )
            
            de_bary = clamp_bary(result_de.x.reshape(N, 2))
            refined_de = refine_local(de_bary, time_limit=5.0)
            update_best(refined_de)
        except Exception:
            pass
    
    # 4. Final high-precision polish
    if best_bary is not None and time.time() < deadline - 10:
        remaining = deadline - 10 - time.time()
        if remaining > 5:
            # Multiple coordinated ascent passes with decreasing step sizes
            bary_polished = best_bary.copy()
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                bary_polished = coordinated_ascent(
                    bary_polished, 
                    max_iter=100, 
                    time_limit=min(3.0, deadline - 5 - time.time())
                )
                update_best(bary_polished)
            
            # SLSQP polish
            if time.time() < deadline - 3:
                try:
                    bounds = [(0, 1)] * (N * 2)
                    result_slsqp = minimize(
                        hard_min_obj,
                        best_bary.flatten(),
                        method='SLSQP',
                        bounds=bounds,
                        options={'maxiter': 1000, 'ftol': 1e-14}
                    )
                    update_best(clamp_bary(result_slsqp.x.reshape(N, 2)))
                except Exception:
                    pass

    # Convert to cartesian
    cart = bary_to_cart(best_bary)
    
    return cart
