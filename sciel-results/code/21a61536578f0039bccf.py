"""Improved packing of 50 equal spheres in a unit cube."""
import numpy as np
from scipy.spatial import Delaunay, Voronoi
from scipy.spatial.distance import cdist
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
    """Compute pairwise distance matrix using cdist for speed."""
    return cdist(pts, pts, metric='euclidean')


def _soft_min(vals, beta=50.0):
    """Soft minimum approximation (numerically stable)."""
    vals = np.asarray(vals)
    m = np.min(vals)
    return m - 1.0 / beta * np.log(np.sum(np.exp(-beta * (vals - m))))


def _objective_and_grad(pts, beta=50.0):
    """Vectorized objective and gradient for soft-min radius maximization."""
    pts = np.asarray(pts).reshape(N, 3)
    
    # Pairwise distances using cdist
    dists = cdist(pts, pts, metric='euclidean')
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
    
    # Pair contributions
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


def _lloyd_step(pts):
    """One Lloyd iteration: move each point to centroid of its Voronoi cell."""
    try:
        vor = Voronoi(pts)
        
        new_pts = pts.copy()
        for i in range(N):
            # Get vertices of Voronoi cell
            region_idx = vor.point_region[i]
            verts_idx = vor.regions[region_idx]
            if verts_idx is None or len(verts_idx) == 0 or -1 in verts_idx:
                # Unbounded region: nudge toward center
                new_pts[i] = 0.95 * pts[i] + 0.05 * np.array([0.0, 0.0, 0.0])
                continue
            verts = vor.vertices[verts_idx]
            
            # Compute centroid
            centroid = np.mean(verts, axis=0)
            
            # Clip to cube
            centroid = np.clip(centroid, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
            new_pts[i] = centroid
        
        # Clip all points
        new_pts = np.clip(new_pts, -CUBE_HALF + 1e-9, CUBE_HALF - 1e-9)
        
        # Push points near walls back in
        dist_to_wall = CUBE_HALF - np.max(np.abs(new_pts), axis=1)
        close_to_wall = dist_to_wall < 0.02
        if np.any(close_to_wall):
            center = np.zeros(3)
            new_pts[close_to_wall] += 0.03 * (center - new_pts[close_to_wall])
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
        pts = _lloyd_step(pts)
    return pts


def _metropolis_mc(pts, n_steps=3000, time_budget=15.0, start_temp=0.01, temp_decay=0.999):
    """Metropolis Monte Carlo to escape local minima with single-point moves."""
    start = time.time()
    rng = np.random.default_rng(42)
    p = pts.copy()
    current_r = _compute_radius(p)
    best_r = current_r
    best_p = p.copy()
    
    step_size = 0.01
    T = start_temp
    
    for step in range(n_steps):
        if time.time() - start > time_budget:
            break
        
        # Pick a random point and axis
        i = rng.integers(N)
        axis = rng.integers(3)
        delta = rng.normal(0, step_size)
        
        p_trial = p.copy()
        p_trial[i, axis] += delta
        p_trial = np.clip(p_trial, -CUBE_HALF, CUBE_HALF)
        
        r_trial = _compute_radius(p_trial)
        
        # Accept if better, or with probability exp((r_trial - r_current)/T)
        if r_trial >= current_r or (T > 1e-10 and rng.random() < np.exp((r_trial - current_r) / T)):
            p = p_trial
            current_r = r_trial
            
            if current_r > best_r:
                best_r = current_r
                best_p = p.copy()
        
        # Anneal temperature and step size
        T *= temp_decay
        step_size *= 0.9999
    
    return best_p


def _repulsion_anneal(pts, start_temp=0.05, end_temp=0.001, n_steps=80, time_budget=12.0):
    """Repulsion annealing with vectorized forces."""
    start = time.time()
    rng = np.random.default_rng(123)
    
    for step in range(n_steps):
        if time.time() - start > time_budget:
            break
        t = step / max(n_steps - 1, 1)
        temp = start_temp * (end_temp / start_temp) ** t
        
        # Compute repulsive forces (vectorized)
        dists = _pair_dist_matrix(pts)
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion: only for close pairs
        mask = dists < 0.3  # threshold
        np.fill_diagonal(mask, False)
        
        if np.any(mask):
            diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            safe_dists = np.where(dists > 1e-12, dists, 1.0)
            unit_vecs = diffs / safe_dists[:, :, np.newaxis]
            # Force magnitude: temp / d^2
            force_mag = temp / (safe_dists ** 2)
            force_mag = np.where(mask, force_mag, 0.0)
            # Sum forces
            forces += np.sum(unit_vecs * force_mag[:, :, np.newaxis], axis=1)
        
        # Add wall repulsion (vectorized)
        for a in range(3):
            margin = CUBE_HALF - np.abs(pts[:, a])
            close_wall = margin < 0.1
            if np.any(close_wall):
                wall_force = temp * (1.0 - margin[close_wall] / 0.1)
                signs = np.sign(pts[close_wall, a])
                forces[close_wall, a] += wall_force * signs
        
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
    deadline = start_time + 150.0 - DEADLINE_MARGIN  # 135s effective
    
    # Try multiple seeds and keep the best
    seeds = [_seed_grid(), _seed_fcc(), _seed_random()]
    
    best_pts = None
    best_r = 0.0
    
    for seed_pts in seeds:
        if time.time() > deadline - 25:
            break
        
        pts = seed_pts.copy()
        
        # Phase 1: Lloyd equalization
        if time.time() < deadline - 30:
            pts = _lloyd_optimize(pts, max_iters=30, time_budget=8.0)
        
        # Phase 2: Repulsion annealing
        if time.time() < deadline - 25:
            pts = _repulsion_anneal(pts, start_temp=0.05, end_temp=0.001, 
                                    n_steps=80, time_budget=10.0)
        
        # Phase 3: Metropolis Monte Carlo to escape local minima
        if time.time() < deadline - 20:
            pts = _metropolis_mc(pts, n_steps=2000, time_budget=12.0)
        
        # Phase 4: Smooth polish
        if time.time() < deadline - 10:
            pts = _smooth_polish(pts, beta_start=30.0, beta_end=400.0,
                                n_iters=25, time_budget=15.0)
        
        # Phase 5: Final high-precision polish
        if time.time() < deadline - 5:
            pts = _smooth_polish(pts, beta_start=400.0, beta_end=1000.0,
                                n_iters=10, time_budget=8.0)
        
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    if best_pts is None:
        best_pts = _seed_grid()
    
    return best_pts
