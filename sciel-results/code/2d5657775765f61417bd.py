"""Heilbronn triangles: maximize minimum triangle area for 11 points in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations
from scipy.spatial import Delaunay

N = 11
H = np.sqrt(3) / 2

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

# Pre-compute triangle indices
tri_indices = list(combinations(range(N), 3))
tri_idx = np.array(tri_indices)  # (165, 3)

def calc_areas_bary(bary):
    """
    Calculate areas of all triangles formed by N points given in barycentric coords (u, v).
    Returns array of shape (165,) containing the area ratios.
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    
    # Extract coordinates for all triangles
    ui = u[tri_idx[:, 0]]
    vi = v[tri_idx[:, 0]]
    wi = w[tri_idx[:, 0]]
    
    uj = u[tri_idx[:, 1]]
    vj = v[tri_idx[:, 1]]
    wj = w[tri_idx[:, 1]]
    
    uk = u[tri_idx[:, 2]]
    vk = v[tri_idx[:, 2]]
    wk = w[tri_idx[:, 2]]
    
    # Determinant calculation vectorized
    det = ui * (vj * wk - vk * wj) - vi * (uj * wk - uk * wj) + wi * (uj * vk - uk * vj)
    
    return np.abs(det)

def clamp_bary(bary, eps=1e-6):
    """
    Clamp barycentric coordinates to the strictly valid triangle region.
    u >= eps, v >= eps, u+v <= 1 - eps
    """
    u = bary[:, 0].copy()
    v = bary[:, 1].copy()
    
    # Clip to strict interior
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    # Ensure u + v <= 1 - eps
    sums = u + v
    invalid = sums > (1 - eps)
    if np.any(invalid):
        scale = (1 - eps) / np.where(sums > 0, sums, 1.0)
        u[invalid] *= scale[invalid]
        v[invalid] *= scale[invalid]
        
    # Re-check bounds after scaling
    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    
    return np.column_stack([u, v])

def soft_min_obj(bary_flat, beta=100.0):
    """
    Objective function for optimization.
    Minimizes the negative soft-min of the areas.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    
    areas = calc_areas_bary(bary)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, subtract max
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        return -np.min(areas)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min = -log_sum / beta
    
    return -soft_min

def _delaunay_smooth_obj(bary_flat, beta=200.0, lam=0.1):
    """
    Hybrid objective: combines soft-min of all triangle areas with a Delaunay-based
    smooth surrogate that encourages uniform small-triangle areas.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    cart = bary_to_cart(bary)
    
    # Standard soft-min of all 165 triangles
    areas = calc_areas_bary(bary)
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:
        soft_min_val = -np.min(areas)
    else:
        log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min_val = -log_sum / beta
    
    # Delaunay-based smooth term: penalize variance of Delaunay triangle areas
    try:
        # Add triangle vertices to ensure Delaunay works on the convex hull
        all_pts = np.vstack([cart, V0, V1, V2])
        tri = Delaunay(all_pts)
        delaunay_areas = []
        for simplex in tri.simplices:
            p0 = all_pts[simplex[0]]
            p1 = all_pts[simplex[1]]
            p2 = all_pts[simplex[2]]
            area = 0.5 * abs(np.cross(p1 - p0, p2 - p0))
            delaunay_areas.append(area)
        
        if len(delaunay_areas) > 0:
            da = np.array(delaunay_areas)
            # Penalize variance: encourages uniform areas
            var_penalty = np.var(da)
        else:
            var_penalty = 0.0
    except Exception:
        var_penalty = 0.0
    
    # Combined objective
    return -soft_min_val + lam * var_penalty

def hard_min_obj(bary_flat):
    """
    Objective function using hard min.
    """
    bary = bary_flat.reshape(N, 2)
    bary = clamp_bary(bary)
    areas = calc_areas_bary(bary)
    return -np.min(areas)

def bary_to_cart(bary):
    """
    Convert barycentric coords (u, v) to Cartesian (x, y).
    bary: (N, 2) array of (u, v). w = 1 - u - v.
    Point = u*A + v*B + w*C
    A=(0,0), B=(1,0), C=(0.5, H)
    x = v + 0.5*w
    y = w*H
    """
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    x = v + 0.5 * w
    y = w * H
    return np.column_stack([x, y])

def generate_hexagonal_init():
    """
    Generate a hexagonal lattice-like initial configuration.
    """
    best_bary = None
    best_val = -np.inf
    
    for s in np.linspace(0.1, 0.5, 20):
        points = []
        for i in range(5):
            for j in range(5):
                u = i * s
                v = j * s * np.sqrt(3) / 2
                if u + v <= 1 and len(points) < N:
                    points.append([u, v])
        
        if len(points) == N:
            bary = np.array(points)
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

def generate_centroidal_init():
    """
    Generate a centroidal Voronoi-like initialization: place points at
    the centroids of a recursive subdivision of the triangle.
    """
    rng = np.random.default_rng(7)
    
    # Start with 3 points near vertices, then iteratively add at worst-spot
    points = []
    # Near each vertex but pulled inward
    for _ in range(3):
        u = 0.25 + rng.normal(0, 0.05)
        v = 0.25 + rng.normal(0, 0.05)
        points.append([u, v])
    
    # Add remaining points at positions that maximize minimum distance to existing
    while len(points) < N:
        best_pos = None
        best_min_dist = -1
        for _ in range(200):
            u = rng.random() * 0.9 + 0.05
            v = rng.random() * (1 - u) * 0.9 + 0.05
            if u + v > 0.95:
                continue
            pos = np.array([u, v])
            min_dist = min(np.linalg.norm(pos - np.array(p)) for p in points)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_pos = pos
        if best_pos is not None:
            points.append(best_pos.tolist())
    
    bary = np.array(points)
    return clamp_bary(bary)

def generate_grid_init():
    """
    Generate a 3x3 grid plus 2 extra points as initial configuration.
    The 3x3 grid points are placed at ideal barycentric positions with small perturbations.
    The 2 extra points are placed near the centroid.
    """
    rng = np.random.default_rng(42)
    
    # 3x3 grid in barycentric coordinates
    # Barycentric coords for a 3x3 grid: u, v in {0, 0.5, 1} but adjusted to fit inside
    # Use positions: (u, v) where u, v are in {1/4, 1/2, 3/4} scaled to fit
    grid_positions = []
    for i in range(3):
        for j in range(3):
            u = (i + 0.5) / 3.0
            v = (j + 0.5) / 3.0 * (1 - u)  # Adjust v to stay inside
            grid_positions.append([u, v])
    
    # Add 2 extra points near centroid
    extra1 = [1/3 + rng.normal(0, 0.05), 1/3 + rng.normal(0, 0.05)]
    extra2 = [1/3 + rng.normal(0, 0.05), 1/3 + rng.normal(0, 0.05)]
    
    bary = np.array(grid_positions + [extra1, extra2])
    bary = clamp_bary(bary)
    
    return bary

def generate_random_init(seed=None):
    """Generate random valid barycentric coordinates."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1 - u)
    return np.column_stack([u, v])

def generate_perturbed_init(base, noise_scale=0.05, seed=None):
    """Generate perturbed version of a base configuration."""
    rng = np.random.default_rng(seed)
    bary = base.copy()
    bary += rng.normal(0, noise_scale, bary.shape)
    return clamp_bary(bary)

def _local_polish_bary(points_cart, deadline):
    """
    Local refinement in barycentric coordinates using Nelder-Mead and L-BFGS-B.
    Takes Cartesian points, converts to barycentric, optimizes, returns Cartesian.
    """
    # Convert to barycentric
    # x = v + 0.5*w, y = w*H
    # w = y/H, v = x - 0.5*w, u = 1 - v - w
    w = points_cart[:, 1] / H
    v = points_cart[:, 0] - 0.5 * w
    u = 1.0 - v - w
    bary = np.column_stack([u, v])
    bary = clamp_bary(bary)
    
    current_min_area = -hard_min_obj(bary.flatten())
    
    # Nelder-Mead polish with tighter tolerances
    if time.time() < deadline - 6:
        try:
            nm_result = minimize(
                hard_min_obj,
                bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            new_bary = clamp_bary(nm_result.x.reshape(N, 2))
            new_min_area = -hard_min_obj(new_bary.flatten())
            if new_min_area > current_min_area:
                bary = new_bary
                current_min_area = new_min_area
        except Exception:
            pass
    
    # L-BFGS-B with soft min for finer tuning
    if time.time() < deadline - 4:
        try:
            lbfgs_result = minimize(
                lambda x: soft_min_obj(x, beta=500.0),
                bary.flatten(),
                method='L-BFGS-B',
                bounds=[(0, 1)] * (N * 2),
                options={'maxiter': 500, 'ftol': 1e-18, 'gtol': 1e-14}
            )
            new_bary = clamp_bary(lbfgs_result.x.reshape(N, 2))
            new_min_area = -hard_min_obj(new_bary.flatten())
            if new_min_area > current_min_area:
                bary = new_bary
                current_min_area = new_min_area
        except Exception:
            pass
    
    # Convert back to Cartesian
    cart = bary_to_cart(bary)
    return cart, current_min_area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_bary = None
    best_val = -np.inf
    
    def update_best(bary):
        """Update best if improved."""
        nonlocal best_bary, best_val
        bary = clamp_bary(bary)
        val = -hard_min_obj(bary.flatten())
        if val > best_val:
            best_val = val
            best_bary = bary.copy()
            return True
        return False

    def try_optimize(bary_init, method='L-BFGS-B', beta1=50.0, beta2=200.0, maxiter1=200, maxiter2=300, use_delaunay=False):
        """
        Three-stage optimization:
        1. Smooth landscape (low beta) to find a good basin.
        2. Medium landscape to refine.
        3. Sharp landscape (high beta) for final precision.
        Optionally uses Delaunay-based smooth surrogate in stage 1.
        """
        if time.time() > deadline - 15:
            return
        
        try:
            bounds = [(0, 1)] * (N * 2)
            
            # Stage 1: Exploration
            if use_delaunay:
                def obj1(x):
                    return _delaunay_smooth_obj(x, beta=beta1, lam=0.1)
            else:
                def obj1(x):
                    return soft_min_obj(x, beta=beta1)
            
            result1 = minimize(
                obj1,
                bary_init.flatten(),
                method=method,
                bounds=bounds,
                options={'maxiter': maxiter1, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            update_best(result1.x.reshape(N, 2))
            
            # Check time before stage 2
            if time.time() > deadline - 8:
                return

            # Stage 2: Medium refinement
            beta_mid = (beta1 + beta2) / 2.0
            def obj2(x):
                return soft_min_obj(x, beta=beta_mid)
            
            result2 = minimize(
                obj2,
                result1.x,
                method=method,
                bounds=bounds,
                options={'maxiter': maxiter2 // 2, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            update_best(result2.x.reshape(N, 2))
            
            if time.time() > deadline - 4:
                return

            # Stage 3: Sharp exploitation
            def obj3(x):
                return soft_min_obj(x, beta=beta2)
            
            result3 = minimize(
                obj3,
                result2.x,
                method=method,
                bounds=bounds,
                options={'maxiter': maxiter2, 'ftol': 1e-16, 'gtol': 1e-12}
            )
            update_best(result3.x.reshape(N, 2))
                
        except Exception:
            pass

    # 1. Start with hexagonal lattice initialization
    hex_init = generate_hexagonal_init()
    try_optimize(hex_init, method='L-BFGS-B', beta1=50.0, beta2=250.0, maxiter1=300, maxiter2=400)
    
    # 2. Try grid-based initialization with Delaunay smooth surrogate
    grid_init = generate_grid_init()
    try_optimize(grid_init, method='L-BFGS-B', beta1=50.0, beta2=250.0, maxiter1=200, maxiter2=300, use_delaunay=True)
    
    # 3. Try centroidal initialization
    centroid_init = generate_centroidal_init()
    try_optimize(centroid_init, method='L-BFGS-B', beta1=50.0, beta2=200.0, maxiter1=200, maxiter2=300)
    
    # 4. Try multiple random initializations
    num_random_starts = 8
    for i in range(num_random_starts):
        if time.time() > deadline - 40:
            break
        rng_seed = 100 + i
        rand_init = generate_random_init(seed=rng_seed)
        try_optimize(rand_init, method='L-BFGS-B', beta1=50.0, beta2=200.0, maxiter1=150, maxiter2=250)
        
    # 5. Perturb the best found and re-optimize
    if best_bary is not None:
        for i in range(6):
            if time.time() > deadline - 25:
                break
            perturbed = generate_perturbed_init(best_bary, noise_scale=0.02, seed=200 + i)
            try_optimize(perturbed, method='L-BFGS-B', beta1=100.0, beta2=350.0, maxiter1=200, maxiter2=300)
    
    # 6. Final polish with Nelder-Mead on hard min
    if best_bary is not None and time.time() < deadline - 18:
        try:
            nm_result = minimize(
                hard_min_obj,
                best_bary.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            update_best(nm_result.x.reshape(N, 2))
        except Exception:
            pass
    
    # 7. Additional L-BFGS-B refinement with very high beta for precision
    if best_bary is not None and time.time() < deadline - 10:
        try:
            def obj_high_beta(x):
                return soft_min_obj(x, beta=600.0)
            
            bounds = [(0, 1)] * (N * 2)
            lbfgs_result = minimize(
                obj_high_beta,
                best_bary.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 800, 'ftol': 1e-18, 'gtol': 1e-14}
            )
            update_best(lbfgs_result.x.reshape(N, 2))
        except Exception:
            pass
    
    # 8. Convert best barycentric to Cartesian
    if best_bary is not None:
        best_cart = bary_to_cart(best_bary)
        # Final local polish in Cartesian space
        if time.time() < deadline - 5:
            best_cart, _ = _local_polish_bary(best_cart, deadline)
        return best_cart
    else:
        # Fallback: return hexagonal init converted to Cartesian
        return bary_to_cart(hex_init)
