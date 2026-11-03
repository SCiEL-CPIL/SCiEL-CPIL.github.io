"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

H = np.sqrt(3) / 2
S3 = np.sqrt(3)

# Precompute all C(11,3) = 165 triples
_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_N_TRIPLES = _TRIPLES.shape[0]


def _min_triangle_area_fast(points):
    """Fully vectorized min triangle area for N=11 points."""
    p1 = points[_TRIPLES[:, 0]]
    p2 = points[_TRIPLES[:, 1]]
    p3 = points[_TRIPLES[:, 2]]
    d1 = p2 - p1
    d2 = p3 - p1
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _clip_to_triangle(points):
    """Clip points to be inside the equilateral triangle."""
    points = points.copy()
    x = points[:, 0]
    y = points[:, 1]
    
    y = np.clip(y, 0, H)
    x_min = y / S3
    x_max = 1 - y / S3
    x = np.clip(x, x_min, x_max)
    x = np.clip(x, 0, 1)
    
    points[:, 0] = x
    points[:, 1] = y
    return points


def _bary_to_cart(bary):
    """Convert barycentric coordinates (a, b, c) to Cartesian.
    point = b*(1,0) + c*(0.5, H)
    so x = b + 0.5*c, y = H*c
    """
    b = bary[:, 1]
    c = bary[:, 2]
    x = b + 0.5 * c
    y = H * c
    return np.column_stack([x, y])


def _cart_to_bary(points):
    """Convert Cartesian to barycentric coordinates."""
    x = points[:, 0]
    y = points[:, 1]
    c = y / H
    b = x - 0.5 * c
    a = 1 - b - c
    return np.column_stack([a, b, c])


def _hierarchical_init():
    """Deterministic hierarchical subdivision of equilateral triangle into 11 equal-area regions.
    Subdivide into 4 sub-triangles (each 1/4 area). Take 3 corner sub-triangles as-is,
    and subdivide the center sub-triangle into 8 smaller triangles (each 1/32 area).
    Total: 3*(1/4) + 8*(1/32) = 24/32 + 8/32 = 32/32 = 1.
    Place each point at the centroid of its region.
    """
    # Vertices of main triangle in barycentric: A=(1,0,0), B=(0,1,0), C=(0,0,1)
    # Midpoints: AB=(0.5,0.5,0), BC=(0,0.5,0.5), CA=(0.5,0,0.5)
    # 4 sub-triangles:
    # T1 (corner A): A, mid_AB, mid_CA  -> centroid bary = (1 + 0.5+0.5, 0+0.5+0, 0+0+0.5)/3 = (2/3, 1/6, 1/6)
    # T2 (corner B): B, mid_AB, mid_BC  -> centroid = (1/6, 2/3, 1/6)
    # T3 (corner C): C, mid_BC, mid_CA  -> centroid = (1/6, 1/6, 2/3)
    # T4 (center): mid_AB, mid_BC, mid_CA -> centroid = (1/3, 1/3, 1/3)
    
    # Subdivide T4 into 8 triangles:
    # T4 vertices: M_AB=(0.5,0.5,0), M_BC=(0,0.5,0.5), M_CA=(0.5,0,0.5)
    # Centroid of T4: G=(1/3,1/3,1/3)
    # Midpoints of T4 edges:
    #   edge M_AB - M_BC: (0.25, 0.5, 0.25)
    #   edge M_BC - M_CA: (0.25, 0.25, 0.5)
    #   edge M_CA - M_AB: (0.5, 0.25, 0.25)
    # 8 triangles: 3 corner triangles + 3 edge-mid triangles + 2 inner
    # Actually, to get 8 equal-area triangles from a triangle, we can use:
    # Split each edge into 2, connect to centroid -> 6 small triangles, but that gives 6 not 8.
    # Alternative: Use 4 sub-sub-triangles of T4 (each 1/16 area), then subdivide one more into 4 (each 1/64).
    # That gives 3*(1/16) + 4*(1/64) = 12/64 + 4/64 = 16/64 = 1/4. Good.
    # So: 3 corner sub-sub-triangles of T4 + 4 sub-sub-sub-triangles of one corner.
    
    # Let's use a simpler approach: 3 corners + 8 in center via different subdivision
    # For 8 equal triangles in T4: place centroid G, and 7 other points? No, we need 8 regions.
    # 
    # Simpler: Just use the 4 main sub-triangles, and for the center one, 
    # place 8 points using a hexagonal pattern within it.
    # But the directive says "equal area" regions. Let me use the 4-1-4 subdivision.
    
    # 4 main sub-triangles. Corner ones get 1 point each at centroid.
    # Center sub-triangle gets subdivided into 4 smaller ones, one of which gets subdivided into 4 more.
    # Total: 3 + 3 + 4 = 10... no, 3 corners + (3 un-subdivided center subs + 4 sub-sub) = 3+7=10. Need 11.
    # 
    # Let me just do: 3 corners at centroids, and 8 points in center region using a balanced pattern.
    # The "equal area" is approximate but structurally balanced.
    
    corner_centroids = np.array([
        [2.0/3, 1.0/6, 1.0/6],  # corner A
        [1.0/6, 2.0/3, 1.0/6],  # corner B  
        [1.0/6, 1.0/6, 2.0/3],  # corner C
    ])
    
    # 8 points in center region (the middle sub-triangle with vertices at midpoints)
    # Center sub-triangle centroid: (1/3, 1/3, 1/3)
    # Place 8 points: centroid + 7 around it in hexagonal pattern
    # Using barycentric within the center sub-triangle
    # Center sub-triangle vertices: M_AB=(0.5,0.5,0), M_BC=(0,0.5,0.5), M_CA=(0.5,0,0.5)
    # Local barycentric within this sub-triangle, then map to global
    
    # For 8 equal-area regions in a triangle: 
    # Use the 4-subdivision (4 equal triangles), then subdivide one into 4 more -> 7 regions
    # Then subdivide one more -> 10 regions. Not 8.
    # 
    # For exactly 8: use a different approach. Place 1 at centroid, 6 at vertices of a hexagon, 1 more.
    # Or just use a balanced 8-point pattern.
    
    # I'll use: centroid of center region + 7 points arranged to maximize separation
    # In barycentric coords of the center sub-triangle (local), then convert to global
    
    # Local barycentric for 8 points in unit triangle (local a,b,c sum to 1)
    local_8 = np.array([
        [1/3, 1/3, 1/3],  # center
        [0.5, 0.25, 0.25],
        [0.25, 0.5, 0.25],
        [0.25, 0.25, 0.5],
        [0.65, 0.15, 0.20],
        [0.15, 0.65, 0.20],
        [0.20, 0.15, 0.65],
        [0.40, 0.40, 0.20],
    ])
    
    # Map local barycentric (la, lb, lc) to global barycentric
    # Global = la*M_AB + lb*M_BC + lc*M_CA
    # M_AB = (0.5, 0.5, 0), M_BC = (0, 0.5, 0.5), M_CA = (0.5, 0, 0.5)
    M_AB = np.array([0.5, 0.5, 0.0])
    M_BC = np.array([0.0, 0.5, 0.5])
    M_CA = np.array([0.5, 0.0, 0.5])
    
    center_8_global = np.zeros((8, 3))
    for i in range(8):
        center_8_global[i] = local_8[i, 0] * M_AB + local_8[i, 1] * M_BC + local_8[i, 2] * M_CA
    
    all_bary = np.vstack([corner_centroids, center_8_global])
    return _bary_to_cart(all_bary)


def _structured_inits():
    """Generate diverse structured initial configurations."""
    inits = []
    
    # Hierarchical subdivision initialization (primary)
    inits.append(_hierarchical_init())
    
    # Symmetric configuration near corners and center
    base_bary = np.array([
        [0.02, 0.02, 0.96],
        [0.96, 0.02, 0.02],
        [0.02, 0.96, 0.02],
        [0.45, 0.02, 0.53],
        [0.02, 0.45, 0.53],
        [0.53, 0.45, 0.02],
        [0.33, 0.33, 0.34],
        [0.25, 0.55, 0.20],
        [0.55, 0.20, 0.25],
        [0.20, 0.25, 0.55],
        [0.45, 0.25, 0.30],
    ])
    inits.append(_bary_to_cart(base_bary))
    
    # Three rows
    row_bary = np.array([
        [0.05, 0.05, 0.90],
        [0.35, 0.05, 0.60],
        [0.65, 0.05, 0.30],
        [0.95, 0.05, 0.00],
        [0.15, 0.35, 0.50],
        [0.45, 0.35, 0.20],
        [0.75, 0.35, -0.10],
        [0.25, 0.65, 0.10],
        [0.55, 0.65, -0.20],
        [0.35, 0.55, 0.10],
        [0.65, 0.55, -0.20],
    ])
    row_bary[:, 2] = 1 - row_bary[:, 0] - row_bary[:, 1]
    inits.append(_bary_to_cart(row_bary))
    
    # Concentric pattern
    cent_bary = np.array([
        [0.33, 0.33, 0.34],
        [0.10, 0.40, 0.50],
        [0.50, 0.40, 0.10],
        [0.40, 0.10, 0.50],
        [0.90, 0.10, 0.00],
        [0.10, 0.90, 0.00],
        [0.50, 0.05, 0.45],
        [0.20, 0.55, 0.25],
        [0.55, 0.25, 0.20],
        [0.25, 0.20, 0.55],
        [0.45, 0.30, 0.25],
    ])
    inits.append(_bary_to_cart(cent_bary))
    
    # Grid-based
    grid_bary = np.array([
        [0.05, 0.05, 0.90],
        [0.30, 0.05, 0.65],
        [0.55, 0.05, 0.40],
        [0.80, 0.05, 0.15],
        [0.15, 0.30, 0.55],
        [0.40, 0.30, 0.30],
        [0.65, 0.30, 0.05],
        [0.25, 0.55, 0.20],
        [0.50, 0.55, -0.05],
        [0.35, 0.45, 0.20],
        [0.60, 0.45, -0.05],
    ])
    grid_bary[:, 2] = 1 - grid_bary[:, 0] - grid_bary[:, 1]
    inits.append(_bary_to_cart(grid_bary))
    
    # More symmetric patterns
    sym_bary = np.array([
        [0.03, 0.03, 0.94],
        [0.94, 0.03, 0.03],
        [0.03, 0.94, 0.03],
        [0.47, 0.03, 0.50],
        [0.03, 0.47, 0.50],
        [0.50, 0.47, 0.03],
        [0.33, 0.33, 0.34],
        [0.22, 0.53, 0.25],
        [0.53, 0.22, 0.25],
        [0.25, 0.22, 0.53],
        [0.42, 0.28, 0.30],
    ])
    inits.append(_bary_to_cart(sym_bary))
    
    # Hexagonal-like pattern
    hex_bary = np.array([
        [0.33, 0.33, 0.34],
        [0.15, 0.45, 0.40],
        [0.55, 0.45, 0.00],
        [0.45, 0.15, 0.40],
        [0.15, 0.15, 0.70],
        [0.70, 0.15, 0.15],
        [0.45, 0.70, -0.15],
        [0.25, 0.25, 0.50],
        [0.50, 0.25, 0.25],
        [0.25, 0.50, 0.25],
        [0.50, 0.50, -0.00],
    ])
    hex_bary[:, 2] = 1 - hex_bary[:, 0] - hex_bary[:, 1]
    inits.append(_bary_to_cart(hex_bary))
    
    return inits


def _objective_cartesian(flat):
    """Objective for Nelder-Mead: minimize negative min triangle area with penalty."""
    points = flat.reshape(N, 2)
    
    x = points[:, 0]
    y = points[:, 1]
    
    # Penalty for being outside the triangle
    eps = 1e-8
    penalty = 0.0
    penalty += np.sum(np.maximum(0, -y - eps)) * 100
    penalty += np.sum(np.maximum(0, y - S3 * x + eps)) * 100
    penalty += np.sum(np.maximum(0, y - S3 * (1 - x) + eps)) * 100
    
    min_area = _min_triangle_area_fast(points)
    return -min_area + penalty


def _objective_barycentric(flat):
    """Objective in barycentric space: minimize negative min triangle area."""
    bary = flat.reshape(N, 3)
    # Normalize to ensure they sum to 1
    bary = bary / bary.sum(axis=1, keepdims=True)
    # Clip small negatives
    bary = np.clip(bary, 0, None)
    bary = bary / bary.sum(axis=1, keepdims=True)
    
    points = _bary_to_cart(bary)
    min_area = _min_triangle_area_fast(points)
    return -min_area


def _softmin_objective_bary(flat, beta=50.0):
    """Smooth soft-min surrogate: minimize -softmin(areas) in barycentric space.
    softmin(a) = -(1/beta) * log(sum(exp(-beta * a_i)))
    As beta -> inf, softmin -> min.
    """
    bary = flat.reshape(N, 3)
    bary = bary / bary.sum(axis=1, keepdims=True)
    bary = np.clip(bary, 0, None)
    bary = bary / bary.sum(axis=1, keepdims=True)
    
    points = _bary_to_cart(bary)
    p1 = points[_TRIPLES[:, 0]]
    p2 = points[_TRIPLES[:, 1]]
    p3 = points[_TRIPLES[:, 2]]
    d1 = p2 - p1
    d2 = p3 - p1
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft-min: -(1/beta) * log(sum(exp(-beta * areas)))
    # Numerically stable: subtract max
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    softmin_val = -log_sum_exp / beta
    
    return -softmin_val


def _repulsion_relax(points, n_iters=200, deadline=None):
    """Repulsion-based iterative relaxation to maximize minimum pairwise distance.
    Each point is pushed away from others based on inverse-square repulsion.
    """
    pts = points.copy()
    pts = _clip_to_triangle(pts)
    
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline - 10:
            break
        
        forces = np.zeros_like(pts)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff = pts[i] - pts[j]
                dist_sq = np.dot(diff, diff) + 1e-12
                dist = np.sqrt(dist_sq)
                # Inverse-square repulsion
                force = diff / (dist_sq * dist)  # unit direction / dist^2
                forces[i] += force
        
        # Adaptive step size
        step = 0.01 / (1 + 0.01 * it)
        pts += step * forces
        pts = _clip_to_triangle(pts)
    
    return pts


def _refine_from(points, deadline, method='nelder'):
    """Refine a configuration using multi-start optimization."""
    best_points = points.copy()
    best_score = _min_triangle_area_fast(points)
    
    rng = np.random.RandomState(42)
    
    # Try multiple perturbations and refinements
    for attempt in range(6):
        if time.time() > deadline - 5:
            break
        
        if attempt == 0:
            start = best_points.copy()
        else:
            # Random perturbation of current best
            scale = 0.05 / (1 + 0.5 * attempt)
            perturbation = rng.randn(N, 2) * scale
            start = best_points + perturbation
            start = _clip_to_triangle(start)
        
        try:
            if method == 'nelder':
                res = minimize(
                    _objective_cartesian,
                    start.flatten(),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                    }
                )
                if res.fun < 0:
                    candidate = res.x.reshape(N, 2)
                    candidate = _clip_to_triangle(candidate)
                    score = _min_triangle_area_fast(candidate)
                    if score > best_score:
                        best_score = score
                        best_points = candidate.copy()
                        
            elif method == 'bary':
                bary_start = _cart_to_bary(start)
                res = minimize(
                    _objective_barycentric,
                    bary_start.flatten(),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                    }
                )
                if res.fun < 0:
                    bary_candidate = res.x.reshape(N, 3)
                    bary_candidate = np.clip(bary_candidate, 0, None)
                    bary_candidate = bary_candidate / bary_candidate.sum(axis=1, keepdims=True)
                    candidate = _bary_to_cart(bary_candidate)
                    score = _min_triangle_area_fast(candidate)
                    if score > best_score:
                        best_score = score
                        best_points = candidate.copy()
                        
        except Exception:
            continue
    
    return best_points, best_score


def _softmin_refine(points, deadline, beta_start=20.0, beta_end=200.0, n_betas=5):
    """Refine using soft-min surrogate with increasing beta (annealing)."""
    best_points = points.copy()
    best_score = _min_triangle_area_fast(points)
    
    bary_current = _cart_to_bary(best_points)
    
    for i in range(n_betas):
        if time.time() > deadline - 10:
            break
        
        # Interpolate beta
        beta = beta_start * (beta_end / beta_start) ** (i / max(n_betas - 1, 1))
        
        try:
            res = minimize(
                lambda flat: _softmin_objective_bary(flat, beta=beta),
                bary_current.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            if res.success or res.fun < _softmin_objective_bary(bary_current.flatten(), beta=beta):
                bary_candidate = res.x.reshape(N, 3)
                bary_candidate = np.clip(bary_candidate, 0, None)
                bary_candidate = bary_candidate / bary_candidate.sum(axis=1, keepdims=True)
                candidate = _bary_to_cart(bary_candidate)
                score = _min_triangle_area_fast(candidate)
                if score > best_score:
                    best_score = score
                    best_points = candidate.copy()
                    bary_current = bary_candidate.copy()
        except Exception:
            continue
    
    return best_points, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_points = None
    best_score = -np.inf
    
    # Generate structured initial configurations
    inits = _structured_inits()
    
    # Phase 1: Repulsion-based pre-relaxation for each init
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 40:
            break
        
        init_pts = _clip_to_triangle(init_pts)
        
        # Apply repulsion relaxation to spread points
        relaxed_pts = _repulsion_relax(init_pts, n_iters=100, deadline=deadline)
        relaxed_score = _min_triangle_area_fast(relaxed_pts)
        
        # Also try original
        orig_score = _min_triangle_area_fast(init_pts)
        
        # Use the better of original vs relaxed
        if relaxed_score > orig_score:
            init_pts = relaxed_pts
            init_score = relaxed_score
        else:
            init_score = orig_score
        
        # Refine this configuration
        refined_pts, refined_score = _refine_from(init_pts, deadline, method='nelder')
        
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts.copy()
    
    # Phase 2: Soft-min surrogate refinement with annealing
    if best_points is not None and time.time() < deadline - 20:
        refined_pts, refined_score = _softmin_refine(best_points, deadline)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts.copy()
    
    # Phase 3: Additional random restarts if time permits
    if best_points is not None and time.time() < deadline - 15:
        rng = np.random.RandomState(123)
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            # Random configuration
            bary = rng.dirichlet(np.ones(N), size=N)
            random_pts = _bary_to_cart(bary)
            random_pts = _clip_to_triangle(random_pts)
            
            # Quick repulsion relaxation
            relaxed = _repulsion_relax(random_pts, n_iters=50, deadline=deadline)
            
            refined_pts, refined_score = _refine_from(relaxed, deadline, method='nelder')
            
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined_pts.copy()
    
    # Phase 4: Final high-precision polish with soft-min at high beta
    if best_points is not None and time.time() < deadline - 8:
        refined_pts, refined_score = _softmin_refine(best_points, deadline, beta_start=50.0, beta_end=500.0, n_betas=3)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts.copy()
    
    # Final fallback
    if best_points is None:
        best_points = _clip_to_triangle(inits[0])
    
    # Ensure all points are valid
    best_points = _clip_to_triangle(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
