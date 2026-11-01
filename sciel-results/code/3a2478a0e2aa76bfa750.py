"""Improved packing of 50 equal spheres in a unit cube."""
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import time

N = 50
CUBE_HALF = 0.5
DEADLINE_MARGIN = 10.0  # safety margin before 150s kill


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


def _active_set_polish(pts, time_budget=15.0, tol=1e-10, max_iters=200):
    """
    Exact active-set polish: directly maximize the true min-radius by
    projecting gradient onto active constraints (pairs and walls at min distance).
    This avoids the soft-min approximation error.
    """
    start = time.time()
    pts = pts.copy()
    r = _compute_radius(pts)
    
    for it in range(max_iters):
        if time.time() - start > time_budget:
            break
        
        # Compute all pair distances and wall margins
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        wall_margins = CUBE_HALF - np.abs(pts)  # (N, 3)
        
        # Find active constraints (those at or near the minimum)
        min_pair_dist = np.min(dists)
        min_wall_margin = np.min(wall_margins)
        r = min(min_pair_dist / 2.0, min_wall_margin)
        
        # Active pairs: distance within tol of min
        active_mask_pairs = dists <= (min_pair_dist + tol * max(1.0, min_pair_dist))
        # Active walls: margin within tol of min
        active_mask_walls = wall_margins <= (min_wall_margin + tol * max(1.0, min_wall_margin))
        
        # Build gradient of r with respect to pts
        # r = min over active constraints
        # For active pair (i,j): dr/dp_i = (p_i - p_j) / (2 * d_ij), dr/dp_j = -(p_i - p_j) / (2 * d_ij)
        # For active wall (i,a): dr/dp_i[a] = -sign(p_i[a])
        
        grad = np.zeros_like(pts)
        n_active = 0
        
        # Pair contributions
        tri_i, tri_j = np.triu_indices(N, k=1)
        active_pairs = active_mask_pairs[tri_i, tri_j]
        if np.any(active_pairs):
            ai = tri_i[active_pairs]
            aj = tri_j[active_pairs]
            d = dists[ai, aj]
            valid = d > 1e-12
            if np.any(valid):
                ai_v = ai[valid]
                aj_v = aj[valid]
                d_v = d[valid]
                dir_vecs = (pts[ai_v] - pts[aj_v]) / (2.0 * d_v[:, np.newaxis])
                np.add.at(grad, ai_v, dir_vecs)
                np.add.at(grad, aj_v, -dir_vecs)
                n_active += len(ai_v)
        
        # Wall contributions
        if np.any(active_mask_walls):
            wi, wa = np.where(active_mask_walls)
            signs = np.sign(pts[wi, wa])
            # Scatter: for each (i, a), grad[i, a] -= sign
            for idx in range(len(wi)):
                grad[wi[idx], wa[idx]] -= signs[idx]
            n_active += len(wi)
        
        if n_active == 0:
            break
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        grad = grad / grad_norm
        
        # Line search: find best step size
        best_step = 0.0
        best_r = r
        for alpha in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
            new_pts = pts + alpha * grad
            new_pts = np.clip(new_pts, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
            new_r = _compute_radius(new_pts)
            if new_r > best_r:
                best_r = new_r
                best_step = alpha
        
        if best_step == 0.0:
            # Try smaller steps
            for alpha in [0.0001, 0.0005, 0.001]:
                new_pts = pts + alpha * grad
                new_pts = np.clip(new_pts, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
                new_r = _compute_radius(new_pts)
                if new_r > best_r:
                    best_r = new_r
                    best_step = alpha
        
        if best_step == 0.0:
            break
        
        pts = pts + best_step * grad
        pts = np.clip(pts, -CUBE_HALF + 1e-12, CUBE_HALF - 1e-12)
        r = best_r
        
        # Check convergence
        if best_step < 1e-10:
            break
    
    return pts


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
    """Repulsion annealing to escape local minima."""
    start = time.time()
    rng = np.random.default_rng(123)
    for step in range(n_steps):
        if time.time() - start > time_budget:
            break
        t = step / max(n_steps - 1, 1)
        temp = start_temp * (end_temp / start_temp) ** t
        
        # Compute repulsive forces
        dists = _pair_dist_matrix(pts)
        forces = np.zeros_like(pts)
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = dists[i, j]
                if d < 1e-12:
                    continue
                # Repulsive force: k/d^2 in direction away from j
                force_mag = temp / (d ** 2)
                forces[i] += (pts[i] - pts[j]) / d * force_mag
        
        # Add wall repulsion
        for i in range(N):
            for a in range(3):
                margin = CUBE_HALF - np.abs(pts[i, a])
                if margin < 0.1:
                    forces[i, a] += temp * (1.0 - margin / 0.1) * np.sign(pts[i, a])
        
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


def _seed_random():
    """Random initial placement."""
    rng = np.random.default_rng(77)
    pts = rng.uniform(-CUBE_HALF + 0.05, CUBE_HALF - 0.05, size=(N, 3))
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN  # 140s effective
    
    # Try multiple seeds and keep the best
    seeds = [_seed_grid(), _seed_fcc(), _seed_random()]
    
    best_pts = None
    best_r = 0.0
    
    for si, seed_pts in enumerate(seeds):
        # Allocate time based on seed index: give more time to later seeds
        if si == 0:
            phase_budgets = [8.0, 12.0, 15.0, 8.0, 10.0]
        elif si == 1:
            phase_budgets = [10.0, 15.0, 20.0, 10.0, 12.0]
        else:
            phase_budgets = [10.0, 15.0, 20.0, 10.0, 12.0]
        
        if time.time() > deadline - 30:
            break
        
        pts = seed_pts.copy()
        
        # Phase 1: Lloyd equalization
        if time.time() < deadline - 50:
            pts = _lloyd_optimize(pts, max_iters=30, time_budget=phase_budgets[0])
        
        # Phase 2: Repulsion annealing
        if time.time() < deadline - 40:
            pts = _repulsion_anneal(pts, start_temp=0.05, end_temp=0.001, 
                                    n_steps=80, time_budget=phase_budgets[1])
        
        # Phase 3: Smooth polish
        if time.time() < deadline - 25:
            pts = _smooth_polish(pts, beta_start=30.0, beta_end=400.0,
                                n_iters=25, time_budget=phase_budgets[2])
        
        # Phase 4: Final high-precision smooth polish
        if time.time() < deadline - 12:
            pts = _smooth_polish(pts, beta_start=400.0, beta_end=1000.0,
                                n_iters=10, time_budget=phase_budgets[3])
        
        # Phase 5: Exact active-set polish
        if time.time() < deadline - 5:
            pts = _active_set_polish(pts, time_budget=phase_budgets[4], tol=1e-10, max_iters=300)
        
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    if best_pts is None:
        best_pts = _seed_grid()
    
    return best_pts
