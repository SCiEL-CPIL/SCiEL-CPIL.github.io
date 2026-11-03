"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

H = np.sqrt(3) / 2
S3 = np.sqrt(3)

# Precompute all C(11,3) = 165 triples
_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_N_TRIPLES = _TRIPLES.shape[0]

# Vertices of the equilateral triangle
_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, H]])


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
    """Clip points to be inside the equilateral triangle (0,0), (1,0), (0.5, H)."""
    points = points.copy()
    x = points[:, 0]
    y = points[:, 1]
    
    # Clip y to [0, H]
    y = np.clip(y, 0, H)
    
    # For a given y, x must be in [y/S3, 1 - y/S3]
    x_min = y / S3
    x_max = 1.0 - y / S3
    x = np.clip(x, x_min, x_max)
    x = np.clip(x, 0, 1)
    
    points[:, 0] = x
    points[:, 1] = y
    return points


def _bary_to_cart(bary):
    """Convert barycentric coordinates to Cartesian.
    bary: (n, 3) array with columns (alpha, beta, gamma) where alpha+beta+gamma=1
    Vertex 0 = (0,0), Vertex 1 = (1,0), Vertex 2 = (0.5, H)
    """
    # x = 0*alpha + 1*beta + 0.5*gamma = beta + 0.5*gamma
    # y = 0*alpha + 0*beta + H*gamma = H*gamma
    x = bary[:, 1] + 0.5 * bary[:, 2]
    y = H * bary[:, 2]
    return np.column_stack([x, y])


def _is_inside_triangle(points, eps=1e-6):
    """Check if all points are inside the equilateral triangle."""
    x = points[:, 0]
    y = points[:, 1]
    
    if np.any(y < -eps) or np.any(y > H + eps) or np.any(x < -eps) or np.any(x > 1 + eps):
        return False
    
    # Check left edge: y <= S3 * x  =>  S3*x - y >= 0
    if np.any(S3 * x - y < -eps):
        return False
    
    # Check right edge: y <= S3 * (1-x)  =>  S3*(1-x) - y >= 0
    if np.any(S3 * (1 - x) - y < -eps):
        return False
    
    return True


def _objective(flat):
    """
    Objective function: minimize -min_area + penalty.
    flat: 2*N array
    """
    points = flat.reshape(N, 2)
    points = _clip_to_triangle(points)
    min_area = _min_triangle_area_fast(points)
    return -min_area


def _objective_bary(flat):
    """
    Objective in barycentric coordinates.
    flat: 3*N array of barycentric coords (each point has 3 coords summing to 1)
    We optimize over 2*N free parameters (alpha, beta) and set gamma = 1 - alpha - beta.
    """
    # flat is 2*N: for each point, (alpha, beta)
    ab = flat.reshape(N, 2)
    alpha = ab[:, 0]
    beta = ab[:, 1]
    gamma = 1.0 - alpha - beta
    
    # Clip to ensure valid barycentric coords
    # We want alpha >= 0, beta >= 0, gamma >= 0
    # If any is negative, we need to project back to the simplex
    # Simple projection: if gamma < 0, scale alpha and beta to sum to 1
    # More robust: use the fact that the triangle is the convex hull of 3 vertices
    # For now, clip and normalize
    bary = np.column_stack([alpha, beta, gamma])
    
    # Project to simplex: if any coord is negative, set to 0 and renormalize
    # This is a simple approach; more robust would be proper simplex projection
    for i in range(N):
        b = bary[i]
        if b[0] < 0 or b[1] < 0 or b[2] < 0:
            # Simple projection: clip negatives to 0, renormalize
            b = np.clip(b, 0, 1)
            s = b.sum()
            if s > 0:
                b = b / s
            else:
                b = np.array([1/3, 1/3, 1/3])
            bary[i] = b
    
    points = _bary_to_cart(bary)
    min_area = _min_triangle_area_fast(points)
    return -min_area


def _fs_init(n_points=11, n_iter=50, seed=42):
    """
    Farthest Point Sampling based initialization.
    Generates a well-spread set of points inside the triangle.
    """
    rng = np.random.RandomState(seed)
    
    # Start with a random point or a corner
    # Let's start with a point near the center
    points = [np.array([0.5, H / 3])]
    
    for _ in range(n_points - 1):
        # Generate many candidate points
        n_candidates = 1000
        # Uniform sampling in triangle:
        # x = r1 * (1 - sqrt(r2)) ... no, standard method:
        # r1, r2 uniform in [0,1]
        # if r1+r2 > 1, r1=1-r1, r2=1-r2
        # x = r1 + 0.5*r2
        # y = H * r2
        
        r1 = rng.rand(n_candidates)
        r2 = rng.rand(n_candidates)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        
        cx = r1 + 0.5 * r2
        cy = H * r2
        candidates = np.column_stack([cx, cy])
        
        # Compute min distance from each candidate to existing points
        # points is a list of (2,) arrays
        # Convert to array for distance computation
        pts_arr = np.array(points)
        
        # dist_sq[i, j] = ||candidates[i] - pts_arr[j]||^2
        # Use broadcasting
        # candidates: (M, 2), pts_arr: (K, 2)
        # diff: (M, K, 2) -> too big? M=1000, K<=11. 1000*11*2 = 22000 floats. Fine.
        diff = candidates[:, np.newaxis, :] - pts_arr[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        
        # For each candidate, find min dist to any existing point
        min_dist_sq = np.min(dist_sq, axis=1)
        
        # Pick the candidate with the largest min_dist_sq
        best_idx = np.argmax(min_dist_sq)
        points.append(candidates[best_idx])
    
    return np.array(points)


def _structured_inits():
    """Generate structured initial configurations based on symmetry."""
    inits = []
    
    # 1. Symmetric hexagonal-like arrangement
    # Center + 6 around in a ring + 4 more
    center = np.array([0.5, H/3])
    r1 = 0.15
    r2 = 0.28
    
    pts = [center]
    # Inner ring (6 points)
    for i in range(6):
        angle = i * np.pi / 3
        x = 0.5 + r1 * np.cos(angle)
        y = H/3 + r1 * np.sin(angle)
        pts.append(np.array([x, y]))
    # Outer ring (4 points, adjusted for triangle shape)
    for angle in [np.pi/6, np.pi/2, 5*np.pi/6, 7*np.pi/6]:
        x = 0.5 + r2 * np.cos(angle)
        y = H/3 + r2 * np.sin(angle)
        pts.append(np.array([x, y]))
    
    pts = np.array(pts)
    pts = _clip_to_triangle(pts)
    inits.append(pts.flatten())
    
    # 2. Grid-like with corners emphasized
    # Place points near corners and edges
    corner_pts = np.array([
        [0.05, 0.02],
        [0.95, 0.02],
        [0.5, H - 0.02],
        [0.15, 0.15],
        [0.85, 0.15],
        [0.35, H - 0.15],
        [0.65, H - 0.15],
        [0.5, H/3],
        [0.3, H/5],
        [0.7, H/5],
        [0.5, 2*H/5],
    ])
    corner_pts = _clip_to_triangle(corner_pts)
    inits.append(corner_pts.flatten())
    
    # 3. Barycentric structured init
    # Use rational barycentric coordinates
    bary_configs = [
        # 3 corners + 3 midpoints + center + 3 more
        [[0.05, 0.05, 0.90],
         [0.95, 0.05, 0.00],
         [0.05, 0.95, 0.00],
         [0.50, 0.50, 0.00],
         [0.50, 0.00, 0.50],
         [0.00, 0.50, 0.50],
         [1/3, 1/3, 1/3],
         [0.25, 0.25, 0.50],
         [0.25, 0.50, 0.25],
         [0.50, 0.25, 0.25],
         [0.40, 0.30, 0.30]],
    ]
    for bary in bary_configs:
        bary = np.array(bary)
        bary[:, 2] = 1 - bary[:, 0] - bary[:, 1]
        pts = _bary_to_cart(bary)
        pts = _clip_to_triangle(pts)
        inits.append(pts.flatten())
    
    # 4. FS-based with different seeds
    for seed in [42, 123, 456, 789, 1000]:
        pts = _fs_init(n_points=N, seed=seed)
        inits.append(pts.flatten())
    
    return inits


def _generate_inits():
    """Generate diverse initial configurations."""
    return _structured_inits()


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    best_points = None
    best_score = -np.inf
    
    inits = _generate_inits()
    
    # Phase 1: Global search with Differential Evolution
    # We use a subset of time for DE
    de_time_budget = 50  # seconds
    
    if time.time() < deadline - de_time_budget:
        try:
            # Bounds for DE: x in [0,1], y in [0,H]
            # But we clip inside, so bounds can be loose
            bounds = [(0, 1), (0, H)] * N
            
            def de_objective(flat):
                points = flat.reshape(N, 2)
                points = _clip_to_triangle(points)
                return -_min_triangle_area_fast(points)
            
            # DE settings
            de_result = differential_evolution(
                de_objective,
                bounds,
                maxiter=40,
                popsize=12,
                tol=0.005,
                seed=42,
                polish=False,
                init='latinhypercube',
                updating='immediate'
            )
            
            if de_result.fun < 0:
                score = -de_result.fun
                if score > best_score:
                    best_score = score
                    best_points = _clip_to_triangle(de_result.x.reshape(N, 2))
                    
        except Exception:
            pass
    
    # Phase 2: Local refinement with multiple methods
    # Use the best point from DE and all initializations
    
    candidates = []
    if best_points is not None:
        candidates.append(best_points.flatten())
    candidates.extend(inits)
    
    # Also add some random perturbations of the best point found so far
    if best_points is not None:
        rng = np.random.RandomState(42)
        for _ in range(8):
            perturbed = best_points + rng.randn(N, 2) * 0.08
            perturbed = _clip_to_triangle(perturbed)
            candidates.append(perturbed.flatten())
    
    # Try SLSQP first (handles non-smooth min better), then Nelder-Mead
    for i, init_flat in enumerate(candidates):
        if time.time() > deadline - 5:
            break
            
        init_points = _clip_to_triangle(init_flat.reshape(N, 2))
        
        # Method 1: SLSQP
        try:
            bounds = [(0, 1), (0, H)] * N
            res_slsqp = minimize(
                _objective,
                init_flat,
                method='SLSQP',
                bounds=bounds,
                options={
                    'maxiter': 300,
                    'ftol': 1e-12
                }
            )
            
            if res_slsqp.fun < 0:
                score = -res_slsqp.fun
                if score > best_score:
                    best_score = score
                    best_points = _clip_to_triangle(res_slsqp.x.reshape(N, 2))
        except Exception:
            pass
        
        # Method 2: Nelder-Mead
        if time.time() > deadline - 5:
            break
        try:
            res_nm = minimize(
                _objective,
                init_flat,
                method='Nelder-Mead',
                options={
                    'maxiter': 1500,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            if res_nm.fun < 0:
                score = -res_nm.fun
                if score > best_score:
                    best_score = score
                    best_points = _clip_to_triangle(res_nm.x.reshape(N, 2))
        except Exception:
            pass
    
    # Phase 3: Final polishing with L-BFGS-B on the best point
    if best_points is not None and time.time() < deadline - 3:
        try:
            bounds = [(0, 1), (0, H)] * N
            res_lbfgs = minimize(
                _objective,
                best_points.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 500,
                    'ftol': 1e-15
                }
            )
            
            if res_lbfgs.fun < 0:
                score = -res_lbfgs.fun
                if score > best_score:
                    best_score = score
                    best_points = _clip_to_triangle(res_lbfgs.x.reshape(N, 2))
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        # Use the best initial
        for init_flat in inits:
            pts = _clip_to_triangle(init_flat.reshape(N, 2))
            score = _min_triangle_area_fast(pts)
            if score > best_score:
                best_score = score
                best_points = pts
                
    # Absolute fallback
    if best_points is None or not _is_inside_triangle(best_points, eps=1e-3):
        best_points = _fs_init(n_points=N, seed=0)
        best_points = _clip_to_triangle(best_points)
        
    return best_points
# EVOLVE-BLOCK-END
