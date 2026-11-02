"""Improved packing of 50 equal spheres in a unit cube."""
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import time

N = 50
CUBE_HALF = 0.5
DEADLINE_MARGIN = 15.0  # safety margin before 150s kill


def _compute_radius(pts):
    """Compute the common radius for a given set of centers."""
    if pts.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check boundaries
    max_coord = np.max(np.abs(pts), axis=1)
    if np.any(max_coord > CUBE_HALF + 1e-12):
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    if min_dist <= 1e-12:
        return 0.0
    r_pair = min_dist / 2.0
    r_wall = CUBE_HALF - np.max(max_coord)
    return min(r_pair, r_wall)


def _pair_dist_matrix(pts):
    """Compute pairwise distance matrix."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return dists


def _soft_min(vals, beta=50.0):
    """Soft minimum approximation (numerically stable)."""
    vals = np.asarray(vals)
    m = np.min(vals)
    return m - 1.0 / beta * np.log(np.sum(np.exp(-beta * (vals - m))))


def _objective_and_grad(pts, beta=50.0):
    """Vectorized objective and gradient for soft-min radius maximization."""
    pts = np.asarray(pts).reshape(N, 3)
    
    # Pairwise distances (vectorized)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Wall margins
    wall_margins = CUBE_HALF - np.abs(pts)  # shape (N, 3)
    wall_margins_flat = wall_margins.flatten()
    
    # Upper triangle indices for pairs
    tri_i, tri_j = np.triu_indices(N, k=1)
    pair_radii = dists[tri_i, tri_j] / 2.0
    
    # Combined radii
    all_radii = np.concatenate([pair_radii, wall_margins_flat])
    
    # Soft min (stable)
    m = np.min(all_radii)
    exp_term = np.exp(-beta * (all_radii - m))
    sum_exp = np.sum(exp_term)
    r_soft = m - (1.0 / beta) * np.log(sum_exp)
    obj = -r_soft
    
    # Weights for gradient
    weights = exp_term / sum_exp
    
    # Vectorized gradient computation
    grad = np.zeros_like(pts)
    
    # Pair contributions: for pair (i,j), weight w, contribution w*(p_i-p_j)/(2*d_ij) to grad[i]
    # Use np.add.at for accumulation
    pair_dists = dists[tri_i, tri_j]
    pair_weights = weights[:len(tri_i)]
    
    # Direction vectors
    dir_vecs = (pts[tri_i] - pts[tri_j]) / (2.0 * pair_dists[:, np.newaxis])
    
    # Accumulate into grad
    np.add.at(grad, tri_i, pair_weights[:, np.newaxis] * dir_vecs)
    np.add.at(grad, tri_j, -pair_weights[:, np.newaxis] * dir_vecs)
    
    # Wall contributions: for wall term (i, a), weight w, contribution -w*sign(p_i[a]) to grad[i,a]
    wall_weights = weights[len(tri_i):]
    signs = np.sign(pts)
    # Reshape wall weights to (N, 3)
    wall_w = wall_weights.reshape(N, 3)
    grad -= wall_w * signs
    
    # Objective gradient is -grad (since we minimize -r_soft)
    obj_grad = -grad
    
    return obj, obj_grad.ravel()


def _lloyd_step(pts, dists):
    """One Lloyd iteration: move each point to centroid of its Voronoi cell."""
    try:
        delaunay = Delaunay(pts)
        # Compute Voronoi vertices
        voronoi = delaunay.coplanar
        # Use scipy's Voronoi for better support
        from scipy.spatial import Voronoi
        vor = Voronoi(pts)
        
        new_pts = pts.copy()
        for i in range(N):
            # Get vertices of Voronoi cell
            if vor.regions[i] is None or len(vor.regions[i]) == 0:
                continue
            verts_idx = vor.regions[i]
            if -1 in verts_idx:
                continue
            verts = vor.vertices[verts_idx]
            
            # Compute centroid
            centroid = np.mean(verts, axis=0)
            
            # Clip to cube
            centroid = np.clip(centroid, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
            
            # Check if centroid is valid (inside cube and not too close to others)
            new_pts[i] = centroid
        
        # Clip all points
        new_pts = np.clip(new_pts, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
        return new_pts
    except Exception:
        return pts


def _lloyd_optimize(pts, max_iters=50, time_budget=10.0):
    """Run Lloyd iterations with time budget."""
    start = time.time()
    for _ in range(max_iters):
        if time.time() - start > time_budget:
            break
        dists = _pair_dist_matrix(pts)
        pts = _lloyd_step(pts, dists)
    return pts


def _repulsion_anneal(pts, start_temp=0.1, end_temp=0.001, n_steps=100, time_budget=15.0):
    """Repulsion annealing to escape local minima (vectorized)."""
    start = time.time()
    rng = np.random.default_rng(123)
    for step in range(n_steps):
        if time.time() - start > time_budget:
            break
        t = step / max(n_steps - 1, 1)
        temp = start_temp * (end_temp / start_temp) ** t
        
        # Compute repulsive forces (vectorized)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Force magnitude: temp / d^2, direction: diff / d
        # Combined: diff * temp / d^3
        with np.errstate(divide='ignore', invalid='ignore'):
            force_factor = temp / (dists ** 3)
        np.fill_diagonal(force_factor, 0)
        forces = np.sum(diff * force_factor[:, :, np.newaxis], axis=1)
        
        # Add wall repulsion (vectorized)
        margins = CUBE_HALF - np.abs(pts)  # (N, 3)
        wall_mask = margins < 0.1
        wall_force = np.where(wall_mask, temp * (1.0 - margins / 0.1), 0.0) * np.sign(pts)
        forces += wall_force
        
        # Metropolis acceptance
        new_pts = pts + forces * 0.01
        new_pts = np.clip(new_pts, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
        
        r_old = _compute_radius(pts)
        r_new = _compute_radius(new_pts)
        
        if r_new > r_old or rng.random() < np.exp((r_new - r_old) / max(temp, 1e-10)):
            pts = new_pts
    
    return pts


def _smooth_polish(pts, beta_start=50.0, beta_end=500.0, n_iters=30, time_budget=30.0):
    """Smooth optimization with annealed beta."""
    start = time.time()
    bounds = [(-CUBE_HALF, CUBE_HALF)] * (N * 3)
    
    for it in range(n_iters):
        if time.time() - start > time_budget:
            break
        t = it / max(n_iters - 1, 1)
        beta = beta_start * (beta_end / beta_start) ** t
        
        def obj_func(p):
            f, g = _objective_and_grad(p, beta)
            return f, g
        
        try:
            res = minimize(obj_func, pts.ravel(), jac=True, bounds=bounds,
                          method='L-BFGS-B', options={'maxiter': 20, 'gtol': 1e-7})
            pts_new = res.x.reshape(N, 3)
            pts_new = np.clip(pts_new, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
            
            r_new = _compute_radius(pts_new)
            r_old = _compute_radius(pts)
            if r_new >= r_old:
                pts = pts_new
        except Exception:
            pass
    
    return pts


def _active_set_polish(pts, time_budget=10.0, tol=1e-9):
    """Polish using only the binding constraints (active set)."""
    start = time.time()
    
    for _ in range(20):
        if time.time() - start > time_budget:
            break
        
        # Find the minimum radius and which constraints are binding
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        r_pair = min_dist / 2.0
        
        wall_margins = CUBE_HALF - np.max(np.abs(pts), axis=1)
        r_wall = np.min(wall_margins)
        
        r = min(r_pair, r_wall)
        
        # Find binding pairs (within 2% of min)
        threshold = r * 2.0 + 1e-6
        tri_i, tri_j = np.triu_indices(N, k=1)
        pair_dists = dists[tri_i, tri_j]
        binding_pairs = np.where(pair_dists < threshold)[0]
        
        # Find binding walls
        binding_walls = np.where(wall_margins < threshold)[0]
        
        if len(binding_pairs) == 0 and len(binding_walls) == 0:
            break
        
        # Build a small QP-like problem: move binding points to increase min radius
        # Use simple coordinate descent on binding constraints
        improved = False
        for _ in range(5):
            if time.time() - start > time_budget:
                break
            
            # For each binding pair, try to increase their separation
            for idx in binding_pairs:
                i, j = tri_i[idx], tri_j[idx]
                d = dists[i, j]
                if d < 1e-12:
                    continue
                # Move i and j apart slightly
                direction = (pts[i] - pts[j]) / d
                step = 0.001
                new_i = pts[i] + direction * step
                new_j = pts[j] - direction * step
                
                # Check if this improves the radius
                new_pts = pts.copy()
                new_pts[i] = new_i
                new_pts[j] = new_j
                new_pts = np.clip(new_pts, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
                
                r_new = _compute_radius(new_pts)
                if r_new > r + 1e-12:
                    pts = new_pts
                    improved = True
                    # Recompute distances
                    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                    dists = np.sqrt(np.sum(diff**2, axis=2))
                    np.fill_diagonal(dists, np.inf)
                    min_dist = np.min(dists)
                    r_pair = min_dist / 2.0
                    wall_margins = CUBE_HALF - np.max(np.abs(pts), axis=1)
                    r_wall = np.min(wall_margins)
                    r = min(r_pair, r_wall)
                    break
            
            # For each binding wall, try to move point away from wall
            for i in binding_walls:
                # Find which coordinate is closest to wall
                coords = np.abs(pts[i])
                axis = np.argmax(coords)
                sign = np.sign(pts[i, axis])
                
                step = 0.001
                new_pts = pts.copy()
                new_pts[i, axis] -= sign * step
                new_pts = np.clip(new_pts, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
                
                r_new = _compute_radius(new_pts)
                if r_new > r + 1e-12:
                    pts = new_pts
                    improved = True
                    # Recompute
                    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                    dists = np.sqrt(np.sum(diff**2, axis=2))
                    np.fill_diagonal(dists, np.inf)
                    min_dist = np.min(dists)
                    r_pair = min_dist / 2.0
                    wall_margins = CUBE_HALF - np.max(np.abs(pts), axis=1)
                    r_wall = np.min(wall_margins)
                    r = min(r_pair, r_wall)
                    break
            
            if not improved:
                break
        
        if not improved:
            break
    
    return pts


def _seed_grid():
    """Grid-based initial placement."""
    rng = np.random.default_rng(42)
    m = 4  # 4x4x4 = 64 > 50
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pts = grid[:N].copy()
    pts += rng.normal(0, 0.01, size=pts.shape)
    return np.clip(pts, -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)


def _seed_fcc():
    """FCC-like initial placement."""
    rng = np.random.default_rng(99)
    # FCC lattice points in a cube
    a = 1.0 / 4.0  # lattice constant
    pts = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                x, y, z = i * a, j * a, k * a
                pts.append([x, y, z])
                pts.append([x + a/2, y + a/2, z])
                pts.append([x + a/2, y, z + a/2])
                pts.append([x, y + a/2, z + a/2])
    pts = np.array(pts)
    # Keep first N points
    pts = pts[:N].copy()
    pts += rng.normal(0, 0.005, size=pts.shape)
    return np.clip(pts, -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)


def _seed_bcc():
    """BCC-like initial placement."""
    rng = np.random.default_rng(123)
    # BCC lattice: corners + body center
    a = 1.0 / 3.0  # lattice constant
    pts = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                x, y, z = i * a - 0.5, j * a - 0.5, k * a - 0.5
                # Corner
                pts.append([x, y, z])
                # Body center
                if i < 3 and j < 3 and k < 3:
                    pts.append([x + a/2, y + a/2, z + a/2])
    pts = np.array(pts)
    # Keep first N points
    pts = pts[:N].copy()
    pts += rng.normal(0, 0.005, size=pts.shape)
    return np.clip(pts, -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)


def _seed_shell_core():
    """Shell-core initial placement: some points near walls, some in core."""
    rng = np.random.default_rng(456)
    pts = []
    # Core: 27 points in a 3x3x3 grid
    core_a = 0.6 / 3.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                x = (i + 0.5) * core_a - 0.3
                y = (j + 0.5) * core_a - 0.3
                z = (k + 0.5) * core_a - 0.3
                pts.append([x, y, z])
    # Shell: 23 points distributed near the faces
    shell_pts = []
    # 6 faces, ~4 per face
    face_offsets = [0.35, 0.40, 0.45, 0.48]
    for face in range(6):
        for off in face_offsets[:4]:
            if face == 0:  # +x
                shell_pts.append([off, rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)])
            elif face == 1:  # -x
                shell_pts.append([-off, rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)])
            elif face == 2:  # +y
                shell_pts.append([rng.uniform(-0.3, 0.3), off, rng.uniform(-0.3, 0.3)])
            elif face == 3:  # -y
                shell_pts.append([rng.uniform(-0.3, 0.3), -off, rng.uniform(-0.3, 0.3)])
            elif face == 4:  # +z
                shell_pts.append([rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), off])
            elif face == 5:  # -z
                shell_pts.append([rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), -off])
    pts.extend(shell_pts[:N - len(pts)])
    pts = np.array(pts)
    pts += rng.normal(0, 0.003, size=pts.shape)
    return np.clip(pts, -CUBE_HALF + 1e-6, CUBE_HALF - 1e-6)


def _seed_random():
    """Random initial placement."""
    rng = np.random.default_rng(77)
    pts = rng.uniform(-CUBE_HALF + 0.05, CUBE_HALF - 0.05, size=(N, 3))
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN  # 135s effective
    
    # Try multiple seeds and keep the best
    seeds = [_seed_grid(), _seed_fcc(), _seed_bcc(), _seed_shell_core(), _seed_random()]
    
    best_pts = None
    best_r = 0.0
    
    for seed_idx, seed_pts in enumerate(seeds):
        if time.time() > deadline - 20:
            break
        
        pts = seed_pts.copy()
        
        # Phase 1: Lloyd equalization
        if time.time() < deadline - 25:
            pts = _lloyd_optimize(pts, max_iters=30, time_budget=8.0)
        
        # Phase 2: Repulsion annealing
        if time.time() < deadline - 20:
            pts = _repulsion_anneal(pts, start_temp=0.05, end_temp=0.001, 
                                    n_steps=80, time_budget=12.0)
        
        # Phase 3: Smooth polish
        if time.time() < deadline - 10:
            pts = _smooth_polish(pts, beta_start=30.0, beta_end=400.0,
                                n_iters=25, time_budget=20.0)
        
        # Phase 4: Active set polish
        if time.time() < deadline - 5:
            pts = _active_set_polish(pts, time_budget=8.0)
        
        # Phase 5: Final high-precision smooth polish
        if time.time() < deadline - 3:
            pts = _smooth_polish(pts, beta_start=400.0, beta_end=1000.0,
                                n_iters=10, time_budget=5.0)
        
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
        
        # If we have good time left, do another round of polish on the best
        if seed_idx == 0 and time.time() < deadline - 30:
            # Extra polish pass
            if time.time() < deadline - 15:
                pts = _smooth_polish(best_pts.copy(), beta_start=100.0, beta_end=500.0,
                                    n_iters=20, time_budget=15.0)
                r = _compute_radius(pts)
                if r > best_r:
                    best_r = r
                    best_pts = pts.copy()
            if time.time() < deadline - 5:
                pts = _active_set_polish(best_pts.copy(), time_budget=10.0)
                r = _compute_radius(pts)
                if r > best_r:
                    best_r = r
                    best_pts = pts.copy()
    
    if best_pts is None:
        best_pts = _seed_grid()
    
    return best_pts
