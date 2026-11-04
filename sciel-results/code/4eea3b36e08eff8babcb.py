import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

N = 300

def _compute_radius(pts):
    """Compute the packing radius for given points using KD-tree for speed."""
    norms = np.linalg.norm(pts, axis=1)
    min_bound = 1.0 - norms.max()
    if min_bound <= 0:
        return 0.0
    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    min_pair = dists[:, 1].min() / 2.0
    return min(min_pair, min_bound)

def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice initialization."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0
    
    for attempt in range(20):
        base_spacing = 2.0 / np.sqrt(n_points) * 0.95
        spacing = base_spacing * (0.90 + 0.012 * attempt)
        
        k = int(1.0 / spacing) + 5
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        
        if len(rows) < n_points:
            continue
            
        pts = np.array(rows)
        
        angle = rng.uniform(0, 2 * np.pi)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        pts = pts @ rot_matrix.T
        
        if attempt > 0:
            noise_scale = spacing * 0.005
            pts = pts + rng.normal(0, noise_scale, pts.shape)
            
        norms = np.linalg.norm(pts, axis=1)
        mask = norms < 1.0
        pts_inside = pts[mask]
        
        if len(pts_inside) < n_points:
            continue
            
        order = np.argsort(norms[mask], kind="stable")
        selected = pts_inside[order[:n_points]]
        
        d = cdist(selected, selected)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
        min_bound = 1.0 - np.linalg.norm(selected, axis=1).max()
        r = min(min_pair, min_bound)
        
        if r > best_r:
            best_r = r
            best_pts = selected.copy()
            
    if best_pts is None:
        spacing = 2.0 / np.sqrt(n_points) * 0.95
        k = int(1.0 / spacing) + 5
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    rows.append((x, y))
        pts = np.array(rows)
        order = np.argsort(np.linalg.norm(pts, axis=1), kind="stable")
        best_pts = pts[order[:n_points]]
        
    return best_pts

def _objective_and_grad(x, T):
    """
    Objective: minimize -softmin(constraints) where constraints are
    half-distances between pairs and distances to boundary.
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, k=1)
    pair_half = d[iu] / 2.0
    
    all_vals = np.concatenate([pair_half, boundary])
    
    # Soft min via log-sum-exp
    neg_vals_T = -all_vals / T
    max_nv = np.max(neg_vals_T)
    if max_nv > 700:
        obj = all_vals.min()
        min_idx = np.argmin(all_vals)
        weights = np.zeros_like(all_vals)
        weights[min_idx] = 1.0
    else:
        exp_shifted = np.exp(neg_vals_T - max_nv)
        weights = exp_shifted / np.sum(exp_shifted)
        obj = -T * (np.log(np.sum(exp_shifted)) + max_nv)
    
    # Gradient
    n_pairs = len(pair_half)
    safe_dists = np.where(d > 1e-10, d, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    W = np.zeros((n, n))
    W[iu] = weights[:n_pairs]
    W[iu[1], iu[0]] = weights[:n_pairs]
    
    grad_pairs = 0.5 * np.einsum('ij,ijc->ic', W, unit_vecs)
    
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    grad_boundary = -pts / safe_norms[:, np.newaxis] * weights[n_pairs:][:, np.newaxis]
    
    grad_pts = grad_pairs + grad_boundary
    grad_obj = -grad_pts.ravel()
    
    return obj, grad_obj

def _force_relax(pts, n_steps=800, deadline=None):
    """Force-directed relaxation with overlap repulsion."""
    pos = pts.copy().astype(np.float64)
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    n = len(pos)
    
    r_est = best_r
    lr = 0.003
    cooldown = 0.999
    
    tree = cKDTree(pos)
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
            
        forces = np.zeros_like(pos)
        
        search_radius = 2.2 * r_est
        pairs = tree.query_pairs(search_radius)
        
        for (i, j) in pairs:
            diff = pos[i] - pos[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                continue
            target = 2.0 * r_est
            if dist < target:
                overlap = target - dist
                force_mag = 5.0 * (overlap ** 3) / (target ** 2)
                unit_vec = diff / dist
                forces[i] += force_mag * unit_vec
                forces[j] -= force_mag * unit_vec
        
        # Boundary forces
        norms = np.linalg.norm(pos, axis=1)
        wall_margin = 0.05
        for i in range(n):
            if norms[i] > 1.0 - wall_margin:
                excess_b = norms[i] - (1.0 - wall_margin)
                if excess_b > 0:
                    f_mag = 10.0 * (excess_b ** 2)
                    if norms[i] > 1e-10:
                        direction = -pos[i] / norms[i]
                    else:
                        direction = np.array([1.0, 0.0])
                    forces[i] += f_mag * direction
        
        pos = pos + lr * forces
        
        # Clamp to unit circle
        norms = np.linalg.norm(pos, axis=1)
        mask_out = norms > 1.0
        if np.any(mask_out):
            pos[mask_out] = pos[mask_out] / norms[mask_out, np.newaxis] * 0.9999
            
        if step % 25 == 0:
            tree = cKDTree(pos)
        
        if step % 40 == 0:
            r = _compute_radius(pos)
            if r > best_r:
                best_r = r
                best_pos = pos.copy()
                r_est = r
                lr = min(0.015, lr * 1.01)
                
        lr *= cooldown
        
    return best_pos, best_r

def _soft_min_optimize(pts, T, maxiter=100, deadline=None):
    """Run L-BFGS-B optimization with soft-min objective."""
    x0 = pts.ravel()
    
    def func(x):
        return _objective_and_grad(x, T)[0]
    
    def jac(x):
        return _objective_and_grad(x, T)[1]
    
    try:
        result = minimize(
            func, x0, jac=jac,
            method='L-BFGS-B',
            options={
                'maxiter': maxiter,
                'ftol': 1e-15,
                'gtol': 1e-12,
                'maxls': 50,
                'maxfun': 300,
            }
        )
        new_pts = result.x.reshape(-1, 2)
        norms = np.linalg.norm(new_pts, axis=1)
        if np.any(norms >= 1.0) or not np.all(np.isfinite(new_pts)):
            return pts, _compute_radius(pts)
        new_r = _compute_radius(new_pts)
        return new_pts, new_r
    except Exception:
        return pts, _compute_radius(pts)

def _simulated_annealing(pts, deadline):
    """Simulated annealing with random perturbations."""
    pos = pts.copy().astype(np.float64)
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    n = len(pos)
    r_est = best_r
    
    T = 0.02
    T_min = 0.0001
    cooling = 0.998
    
    rng = np.random.default_rng(12345)
    
    while T > T_min and time.time() < deadline:
        # Pick a random point and perturb it
        i = rng.integers(n)
        delta = rng.normal(0, r_est * 0.3, 2)
        new_pos = pos.copy()
        new_pos[i] += delta
        
        # Keep inside circle
        norm_i = np.linalg.norm(new_pos[i])
        if norm_i > 1.0 - r_est:
            new_pos[i] = new_pos[i] / norm_i * (1.0 - r_est)
        
        new_r = _compute_radius(new_pos)
        delta_r = new_r - _compute_radius(pos)
        
        if delta_r > 0 or rng.random() < np.exp(delta_r / T):
            pos = new_pos
            if new_r > best_r:
                best_r = new_r
                best_pos = pos.copy()
        
        T *= cooling
    
    return best_pos, best_r

def _multi_start_optimize(pts, deadline):
    """Try multiple optimization temperatures from the current point."""
    best_pos = pts.copy()
    best_r = _compute_radius(pts)
    
    temperatures = [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002]
    
    for T in temperatures:
        if time.time() > deadline:
            break
        
        new_pts, new_r = _soft_min_optimize(best_pos, T, maxiter=80, deadline=deadline)
        
        if new_r > best_r:
            best_r = new_r
            best_pos = new_pts.copy()
        elif time.time() > deadline - 5:
            break
            
    return best_pos, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_pts = None
    best_r = 0.0
    
    # Phase 1: Multiple initializations with force relaxation
    inits = [
        _hex_init(N, seed=42),
        _hex_init(N, seed=123),
        _hex_init(N, seed=999),
        _hex_init(N, seed=777),
        _hex_init(N, seed=555),
    ]
    
    for idx, pts in enumerate(inits):
        if time.time() > deadline - 60.0:
            break
            
        relaxed_pts, relaxed_r = _force_relax(pts, n_steps=600, deadline=deadline - 50.0)
        
        if relaxed_r > best_r:
            best_r = relaxed_r
            best_pts = relaxed_pts.copy()
            
    if best_pts is None:
        best_pts = _hex_init(N, seed=0)
        best_r = _compute_radius(best_pts)
        
    # Phase 2: Simulated annealing for broader search
    if time.time() < deadline - 35.0:
        sa_pts, sa_r = _simulated_annealing(best_pts, deadline=deadline - 30.0)
        if sa_r > best_r:
            best_r = sa_r
            best_pts = sa_pts.copy()
    
    # Phase 3: Force relaxation again after SA
    if time.time() < deadline - 25.0:
        relaxed_pts, relaxed_r = _force_relax(best_pts, n_steps=500, deadline=deadline - 20.0)
        if relaxed_r > best_r:
            best_r = relaxed_r
            best_pts = relaxed_pts.copy()
    
    # Phase 4: Multi-temperature soft-min optimization
    if time.time() < deadline - 10.0:
        opt_pts, opt_r = _multi_start_optimize(best_pts, deadline=deadline - 5.0)
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts.copy()
    
    # Phase 5: Final polish
    if time.time() < deadline - 3.0:
        # Very fine soft-min
        final_pts, final_r = _soft_min_optimize(best_pts, T=0.00001, maxiter=150, deadline=deadline - 1.0)
        if final_r > best_r:
            best_r = final_r
            best_pts = final_pts.copy()
    
    # Final validity check
    if best_pts is None:
        best_pts = _hex_init(N, seed=0)
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms < 1.0
    if not np.all(mask) or not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N, seed=0)
    
    return best_pts
