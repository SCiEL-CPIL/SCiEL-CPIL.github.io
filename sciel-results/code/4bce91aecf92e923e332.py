"""Circle packing in unit square - 26 circles maximizing sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import Delaunay, Voronoi

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Check if circles are valid (inside square, disjoint interiors)."""
    n = len(radii)
    for i in range(n):
        if pos[i, 0] - radii[i] < -tol or pos[i, 0] + radii[i] > 1.0 + tol:
            return False
        if pos[i, 1] - radii[i] < -tol or pos[i, 1] + radii[i] > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True


def _voronoi_radii(pts, tol=1e-12):
    """
    Compute the maximum radius for each point such that circles are disjoint
    and inside the unit square, using Voronoi distances.
    This ensures geometric non-overlap by construction.
    """
    n = len(pts)
    if n < 3:
        # Fallback for small n
        radii = np.zeros(n)
        for i in range(n):
            radii[i] = min(pts[i,0], 1-pts[i,0], pts[i,1], 1-pts[i,1])
        return radii

    # Add boundary points to ensure Voronoi cells are bounded by the square
    # We add points slightly outside the square to define the boundary
    # Actually, scipy Voronoi on points inside [0,1] will have unbounded cells.
    # We need to compute distance to the nearest edge of the square for each point.
    
    # Compute distance to square boundaries
    dist_to_boundary = np.zeros(n)
    for i in range(n):
        dist_to_boundary[i] = min(pts[i, 0], 1.0 - pts[i, 0], pts[i, 1], 1.0 - pts[i, 1])

    # Compute Voronoi diagram
    try:
        vor = Voronoi(pts)
    except Exception:
        # Degenerate case, return boundary distances
        return dist_to_boundary

    # For each point, find the minimum distance to any Voronoi vertex
    # or to the boundary of the square.
    radii = np.zeros(n)
    
    # Voronoi.vertices are the vertices of the Voronoi diagram.
    # vor.regions[i] is the list of vertex indices for the region of point i.
    # A region is bounded if it doesn't contain -1.
    
    for i in range(n):
        min_dist = dist_to_boundary[i]
        region = vor.regions[i]
        if region is not None and -1 not in region:
            # Bounded region: compute distance to all vertices
            verts = vor.vertices[region]
            if len(verts) > 0:
                dx = verts[:, 0] - pts[i, 0]
                dy = verts[:, 1] - pts[i, 1]
                dists = np.sqrt(dx**2 + dy**2)
                min_dist = min(min_dist, np.min(dists))
        else:
            # Unbounded region: distance is determined by the boundary
            # We already have dist_to_boundary, but we should also check
            # if there are any finite vertices in the region that are closer
            # than the boundary distance.
            if region is not None:
                finite_verts = [v for v in region if v != -1]
                if finite_verts:
                    verts = vor.vertices[finite_verts]
                    if len(verts) > 0:
                        dx = verts[:, 0] - pts[i, 0]
                        dy = verts[:, 1] - pts[i, 1]
                        dists = np.sqrt(dx**2 + dy**2)
                        min_dist = min(min_dist, np.min(dists))
        
        radii[i] = min_dist

    return radii


def _voronoi_objective(pts_flat, deadline):
    """
    Objective function: minimize negative sum of Voronoi radii.
    pts_flat is a 1D array of length 2*N.
    """
    if time.time() > deadline:
        return 1e10
    
    pts = pts_flat.reshape(N, 2)
    
    # Ensure points are inside the square with a small margin
    # to avoid numerical issues with Voronoi
    pts = np.clip(pts, 1e-6, 1.0 - 1e-6)
    
    radii = _voronoi_radii(pts)
    
    # Ensure radii are non-negative
    radii = np.maximum(0, radii)
    
    return -np.sum(radii)


def _voronoi_gradient(pts_flat, deadline):
    """
    Numerical gradient of the Voronoi objective.
    """
    if time.time() > deadline:
        return np.zeros(2*N)
    
    pts = pts_flat.reshape(N, 2)
    pts = np.clip(pts, 1e-6, 1.0 - 1e-6)
    
    radii = _voronoi_radii(pts)
    f0 = -np.sum(radii)
    
    grad = np.zeros(2*N)
    eps = 1e-6
    
    for i in range(2*N):
        pts_pert = pts.copy()
        pts_pert[i//2, i%2] += eps
        pts_pert = np.clip(pts_pert, 1e-6, 1.0 - 1e-6)
        radii_pert = _voronoi_radii(pts_pert)
        f_pert = -np.sum(np.maximum(0, radii_pert))
        grad[i] = (f_pert - f0) / eps
    
    return grad


def _make_grid(n, seed=42):
    """Create a grid of n circles inside the unit square."""
    side = int(np.ceil(np.sqrt(n)))
    pos = np.zeros((n, 2))
    for i in range(side):
        for j in range(side):
            idx = i * side + j
            if idx < n:
                pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
    return pos


def _make_random(n, seed=42):
    """Random initial positions."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, size=(n, 2))
    return pos


def _make_hexagonal(n, seed=42):
    """Create a hexagonal packing pattern of points."""
    pos = np.zeros((n, 2))
    r = 0.1
    rows = 5
    count = 0
    for row in range(rows):
        y = 0.1 + row * r * np.sqrt(3)
        if row % 2 == 0:
            xs = np.arange(5) * 2 * r + r
        else:
            xs = np.arange(5) * 2 * r + 2 * r
        for col, x in enumerate(xs):
            if count < n:
                pos[count] = [x, y]
                count += 1
    return pos


def _make_center_ring(n, seed=42):
    """One center, rings around."""
    pos = np.zeros((n, 2))
    pos[0] = [0.5, 0.5]
    
    # Inner ring: 6 circles
    dist_inner = 0.25
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pos[i+1] = [0.5 + dist_inner * np.cos(angle), 0.5 + dist_inner * np.sin(angle)]
    
    # Outer ring: remaining circles
    dist_outer = 0.45
    remaining = n - 7
    for i in range(remaining):
        angle = 2 * np.pi * i / remaining
        pos[i+7] = [0.5 + dist_outer * np.cos(angle), 0.5 + dist_outer * np.sin(angle)]
    
    return pos


def _post_process(pos, radii, deadline):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    
    for iteration in range(200):
        if time.time() > deadline:
            break
        valid = True
        
        # Check boundaries
        for i in range(N):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r:
                radii[i] = max_r * 0.99
                valid = False
        
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    # Shrink both proportionally
                    scale = dist / (min_dist + 1e-12) * 0.99
                    radii[i] *= scale
                    radii[j] *= scale
                    valid = False
        
        if valid:
            break
    
    # Final clamp
    for i in range(N):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        radii[i] = max(1e-6, min(radii[i], max_r))
    
    return pos, radii


def _polish(pos, radii, deadline):
    """High-precision polish pass using L-BFGS-B on the full parameterization."""
    pos = pos.copy()
    radii = radii.copy()
    
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = []
    for i in range(N):
        bounds.append((radii[i], 1.0 - radii[i]))  # x
        bounds.append((radii[i], 1.0 - radii[i]))  # y
    for i in range(N):
        bounds.append((1e-4, 0.3))  # radius
    
    # Define a smooth objective for polishing
    def smooth_obj(x):
        if time.time() > deadline:
            return 1e10
        p = x[:2*N].reshape(N, 2)
        r = x[2*N:]
        
        # Boundary penalty
        viol_l = np.maximum(0, r - p[:, 0])
        viol_r = np.maximum(0, p[:, 0] - (1.0 - r))
        viol_b = np.maximum(0, r - p[:, 1])
        viol_t = np.maximum(0, p[:, 1] - (1.0 - r))
        boundary_penalty = np.sum(viol_l**2) + np.sum(viol_r**2) + np.sum(viol_b**2) + np.sum(viol_t**2)
        
        # Overlap penalty
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        min_dists = r[:, np.newaxis] + r[np.newaxis, :]
        np.fill_diagonal(dists, np.inf)
        violations = np.maximum(0, min_dists - dists)
        np.fill_diagonal(violations, 0)
        overlap_penalty = 0.5 * np.sum(violations**2)
        
        return -np.sum(r) + 100.0 * (boundary_penalty + overlap_penalty)
    
    try:
        result = minimize(smooth_obj, x0, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
        if result.success:
            pos_opt = result.x[:2*N].reshape(N, 2)
            radii_opt = result.x[2*N:]
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt, deadline)
            if _is_valid(pos_pp, radii_pp):
                return pos_pp, radii_pp
    except Exception:
        pass
    
    return pos, radii


def _voronoi_optimize(pos_init, deadline):
    """
    Optimize the Voronoi seed points using L-BFGS-B with numerical gradients.
    """
    pts_flat = pos_init.flatten()
    
    # Bounds: points must be inside [0, 1]
    bounds = [(1e-6, 1.0 - 1e-6)] * (2 * N)
    
    try:
        result = minimize(_voronoi_objective, pts_flat, method='L-BFGS-B',
                          bounds=bounds, jac=_voronoi_gradient,
                          options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 5000})
        
        if result.success:
            pos_opt = result.x.reshape(N, 2)
            radii_opt = _voronoi_radii(pos_opt)
            return pos_opt, radii_opt
    except Exception:
        pass
    
    return pos_init, _voronoi_radii(pos_init)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    def update_best(pos, radii):
        nonlocal best_pos, best_radii, best_score
        pos_pp, radii_pp = _post_process(pos, radii, deadline)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()
    
    # Generate initial configurations of seed points
    configs = []
    configs.append(_make_grid(N))
    configs.append(_make_hexagonal(N))
    configs.append(_make_center_ring(N))
    
    rng = np.random.RandomState(42)
    for seed in range(5):
        configs.append(_make_random(N, seed=seed + 100))
    
    # Phase 1: Voronoi optimization from each config
    for pos_init in configs:
        if time.time() > deadline - 30:
            break
        
        pos_opt, radii_opt = _voronoi_optimize(pos_init, deadline)
        update_best(pos_opt, radii_opt)
    
    # Phase 2: Nelder-Mead refinement on best found (using Voronoi objective)
    if best_pos is not None and time.time() < deadline - 20:
        try:
            result_nm = minimize(_voronoi_objective, best_pos.flatten(), method='Nelder-Mead',
                                 bounds=[(1e-6, 1.0 - 1e-6)] * (2*N),
                                 options={'maxiter': 3000, 'maxfev': 15000, 'xatol': 1e-8, 'fatol': 1e-8})
            pos_opt = result_nm.x.reshape(N, 2)
            radii_opt = _voronoi_radii(pos_opt)
            update_best(pos_opt, radii_opt)
        except Exception:
            pass
    
    # Phase 3: Differential evolution for global exploration
    if time.time() < deadline - 40:
        try:
            bounds_de = [(0.05, 0.95)] * (2 * N)
            
            de_result = differential_evolution(_voronoi_objective, bounds_de, maxiter=80,
                                               popsize=15, seed=42, tol=1e-8,
                                               polish=False, mutation=(0.3, 1.0),
                                               recombination=0.7)
            
            if de_result.x is not None:
                pos_opt = de_result.x.reshape(N, 2)
                radii_opt = _voronoi_radii(pos_opt)
                update_best(pos_opt, radii_opt)
        except Exception:
            pass
    
    # Phase 4: Polish the best result
    if best_pos is not None and time.time() < deadline - 5:
        best_pos, best_radii = _polish(best_pos, best_radii, deadline)
        
        # Extra polish pass if time permits
        if time.time() < deadline - 2:
            best_pos, best_radii = _polish(best_pos, best_radii, deadline)
    
    # Fallback
    if best_pos is None:
        pos = _make_grid(N)
        radii = _voronoi_radii(pos)
        best_pos, best_radii = _post_process(pos, radii, deadline)
    
    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
