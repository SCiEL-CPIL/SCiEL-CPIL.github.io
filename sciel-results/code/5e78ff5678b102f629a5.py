import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from collections import defaultdict

N = 300

def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice and select n_points best points."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0
    
    # Try a few different spacings and rotations
    for attempt in range(15):
        base_spacing = 2.0 / (np.sqrt(n_points) + 1.0) * 0.95
        spacing = base_spacing * (0.92 + 0.02 * attempt)
        
        k = int(1.0 / spacing) + 6
        lattice_pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                lattice_pts.append((x, y))
        lattice_pts = np.array(lattice_pts)
        
        angle = rng.uniform(0, 2 * np.pi)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        rotated = np.column_stack([
            lattice_pts[:, 0] * cos_a - lattice_pts[:, 1] * sin_a,
            lattice_pts[:, 0] * sin_a + lattice_pts[:, 1] * cos_a
        ])
        
        norms = np.linalg.norm(rotated, axis=1)
        margin = spacing * 0.5
        mask = norms <= 1.0 - margin
        if mask.sum() < n_points:
            continue
        
        candidates = rotated[mask]
        cand_norms = norms[mask]
        
        if attempt > 0:
            candidates = candidates + rng.normal(0, spacing * 0.03, candidates.shape)
            norms2 = np.linalg.norm(candidates, axis=1)
            valid = norms2 < 1.0
            if valid.sum() < n_points:
                continue
            candidates = candidates[valid]
            cand_norms = norms2[valid]
        
        if len(candidates) > n_points:
            order = np.argsort(cand_norms, kind="stable")
            selected = candidates[order[:n_points]]
        else:
            selected = candidates
        
        d = cdist(selected, selected)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
        min_bound = 1.0 - np.linalg.norm(selected, axis=1).max()
        r = min(min_pair, min_bound)
        
        if r > best_r:
            best_r = r
            best_pts = selected.copy()
            
    if best_pts is None:
        spacing = 2.0 / (np.sqrt(n_points) + 1.0) * 0.95
        k = int(1.0 / spacing) + 6
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

def _compute_radius(pts):
    """Compute the packing radius for given points."""
    norms = np.linalg.norm(pts, axis=1)
    min_bound = 1.0 - norms.max()
    if min_bound <= 0:
        return 0.0
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() / 2.0
    return min(min_pair, min_bound)

def _soft_min_logsumexp(values, T):
    """Stable soft minimum: -T * logsumexp(-values/T)"""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    neg_vals = -values / T
    max_neg = np.max(neg_vals)
    if max_neg > 700:
        return values.min()
    exp_shifted = np.exp(neg_vals - max_neg)
    log_sum = np.log(np.sum(exp_shifted)) + max_neg
    return -T * log_sum

def _objective_and_grad(x, T, edge_indices=None):
    """
    Objective: minimize -softmin(constraints)
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms
    
    if edge_indices is None:
        d = cdist(pts, pts)
        iu = np.triu_indices(n, k=1)
        pair_half = d[iu] / 2.0
    else:
        pair_half = np.array([np.linalg.norm(pts[i] - pts[j]) / 2.0 for i, j in edge_indices])
        
    all_vals = np.concatenate([pair_half, boundary])
    
    obj = -_soft_min_logsumexp(all_vals, T)
    
    neg_vals_T = -all_vals / T
    max_nv = np.max(neg_vals_T)
    if max_nv > 700:
        min_idx = np.argmin(all_vals)
        grad_all = np.zeros_like(all_vals)
        grad_all[min_idx] = -1.0
    else:
        exp_shifted = np.exp(neg_vals_T - max_nv)
        weights = exp_shifted / np.sum(exp_shifted)
        grad_all = -weights
        
    grad_pts = np.zeros_like(pts)
    
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    grad_boundary = -pts / safe_norms[:, np.newaxis] * grad_all[len(pair_half):][:, np.newaxis]
    grad_pts += grad_boundary
    
    if edge_indices is None:
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        safe_dists = np.where(dists > 1e-10, dists, 1.0)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        
        W = np.zeros((n, n))
        n_pairs = len(pair_half)
        iu = np.triu_indices(n, k=1)
        W[iu] = grad_all[:n_pairs]
        W[iu[1], iu[0]] = grad_all[:n_pairs]
        
        weighted_unit = W[:, :, np.newaxis] * unit_vecs
        grad_pairs = 0.5 * np.sum(weighted_unit, axis=1)
        grad_pts += grad_pairs
    else:
        for k, (i, j) in enumerate(edge_indices):
            w = grad_all[k]
            if w == 0: continue
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            if d < 1e-10: continue
            unit_vec = diff / d
            grad_pts[i] += 0.5 * w * unit_vec
            grad_pts[j] -= 0.5 * w * unit_vec
            
    return obj, grad_pts.ravel()

def _force_relax_spatial(pts, n_steps=5000, deadline=None):
    """
    Force-directed relaxation using spatial hashing for O(N) neighbor search.
    This allows for many more steps within the time budget.
    """
    pos = pts.copy().astype(np.float64)
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    n = len(pos)
    
    r_est = best_r
    lr = 0.01
    cooldown = 0.9995
    target_margin = 1.01 # Target distance = 2 * r * target_margin

    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
            
        forces = np.zeros_like(pos)
        
        # Spatial Hashing
        # Cell size should be larger than the max interaction distance
        # We want to push apart points that are closer than 2*r_est * 1.2
        interaction_dist = 2.0 * r_est * 1.5
        cell_size = max(interaction_dist, 0.01)
        
        grid = defaultdict(list)
        for i in range(n):
            cx = int(pos[i, 0] / cell_size)
            cy = int(pos[i, 1] / cell_size)
            grid[(cx, cy)].append(i)
            
        # Calculate forces
        for i in range(n):
            cx = int(pos[i, 0] / cell_size)
            cy = int(pos[i, 1] / cell_size)
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    neighbors = grid.get((cx + dx, cy + dy), [])
                    for j in neighbors:
                        if j <= i:
                            continue
                        diff = pos[i] - pos[j]
                        dist_sq = np.dot(diff, diff)
                        if dist_sq < 1e-10:
                            continue
                        dist = np.sqrt(dist_sq)
                        
                        # We care about pairs closer than the target separation
                        target = 2.0 * r_est * target_margin
                        if dist < target:
                            excess = target - dist
                            # Force magnitude: strong repulsion when very close, softer when further
                            # Using a smooth potential derivative
                            force_mag = 10.0 * (excess / target) ** 2
                            unit_vec = diff / dist
                            forces[i] += force_mag * unit_vec
                            forces[j] -= force_mag * unit_vec
        
        # Boundary constraint
        norms = np.linalg.norm(pos, axis=1)
        wall_margin = 0.05
        in_wall = norms > (1.0 - wall_margin)
        if np.any(in_wall):
            excess_b = np.where(in_wall, norms - (1.0 - wall_margin), 0.0)
            f_mag = 50.0 * excess_b
            safe_norms = np.where(norms > 1e-10, norms, 1.0)
            boundary_force = -pos / safe_norms[:, np.newaxis] * f_mag[:, np.newaxis]
            forces += boundary_force
        
        # Adaptive step
        force_norm = np.linalg.norm(forces)
        if force_norm > 1e-10:
            step_scale = min(1.0, 0.005 / force_norm)
            pos = pos + lr * step_scale * forces
        else:
            pos = pos + lr * forces
            
        # Clamp inside circle
        norms = np.linalg.norm(pos, axis=1)
        mask_out = norms > 1.0
        if np.any(mask_out):
            pos[mask_out] = pos[mask_out] / norms[mask_out, np.newaxis] * 0.999
            
        if step % 50 == 0:
            r = _compute_radius(pos)
            if r > best_r:
                best_r = r
                best_pos = pos.copy()
                r_est = min(r, 0.06)
                
        lr *= cooldown
        
    r_final = _compute_radius(pos)
    if r_final > best_r:
        best_r = r_final
        best_pos = pos.copy()
        
    return best_pos, best_r

def _get_active_edges(pts, k=12):
    """Get the k nearest neighbors for each point to form a sparse graph."""
    n = len(pts)
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    edges = set()
    for i in range(n):
        k_neighbors = np.argsort(d[i])[:k]
        for j in k_neighbors:
            if i < j:
                edges.add((i, j))
            else:
                edges.add((j, i))
    return list(edges)

def _refine_with_scipy(pts, deadline):
    """Refine using scipy minimize with soft-min objective."""
    n = len(pts)
    x0 = pts.ravel()
    
    # Use a range of temperatures
    for T in [0.01, 0.005, 0.002, 0.001, 0.0005]:
        if time.time() > deadline:
            break
        
        # Get active edges
        edge_indices = _get_active_edges(pts, k=16)
        
        res = minimize(
            _objective_and_grad,
            x0,
            args=(T, edge_indices),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 1500, 'maxfun': 8000, 'maxls': 100, 'maxcor': 30}
        )
        
        pts = res.x.reshape(-1, 2)
        norms = np.linalg.norm(pts, axis=1)
        mask_out = norms > 1.0
        if np.any(mask_out):
            pts[mask_out] = pts[mask_out] / norms[mask_out, np.newaxis] * 0.999

    return pts

def _active_set_polish(pts, deadline, max_iter=50):
    """
    Simple active-set polish: identify the minimum constraint and push points apart.
    """
    n = len(pts)
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
            
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
            
        # Find the minimum constraint
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        min_pair_idx = np.unravel_index(np.argmin(d), d.shape)
        min_pair_dist = d[min_pair_idx] / 2.0
        
        norms = np.linalg.norm(pts, axis=1)
        min_bound_idx = np.argmin(1.0 - norms)
        min_bound_dist = 1.0 - norms[min_bound_idx]
        
        if min_pair_dist < min_bound_dist:
            # Push apart the closest pair
            i, j = min_pair_idx
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                break
            unit_vec = diff / dist
            step = 0.001
            pts[i] += step * unit_vec
            pts[j] -= step * unit_vec
        else:
            # Push inwards the closest to boundary
            i = min_bound_idx
            norm_i = norms[i]
            if norm_i < 1e-10:
                break
            unit_vec = -pts[i] / norm_i
            step = 0.001
            pts[i] += step * unit_vec
            
        # Ensure inside
        norms_new = np.linalg.norm(pts, axis=1)
        mask_out = norms_new > 1.0
        if np.any(mask_out):
            pts[mask_out] = pts[mask_out] / norms_new[mask_out, np.newaxis] * 0.9999

    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Initialization
    pts = _hex_init(N, seed=42)
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    # 2. Force relaxation with spatial hashing (Fast, many steps)
    pts, r_force = _force_relax_spatial(pts, n_steps=8000, deadline=deadline)
    if r_force > best_r:
        best_r = r_force
        best_pts = pts.copy()
    
    if time.time() > deadline:
        return best_pts
        
    # 3. SciPy refinement (Slow, precise)
    pts = _refine_with_scipy(pts, deadline)
    r_scipy = _compute_radius(pts)
    if r_scipy > best_r:
        best_r = r_scipy
        best_pts = pts.copy()
    
    if time.time() > deadline:
        return best_pts
        
    # 4. Active set polish (Fine tuning)
    pts = _active_set_polish(pts, deadline, max_iter=100)
    r_polish = _compute_radius(pts)
    if r_polish > best_r:
        best_r = r_polish
        best_pts = pts.copy()
        
    if time.time() > deadline:
        return best_pts
        
    # 5. Final short force relaxation to settle
    pts, r_final = _force_relax_spatial(pts, n_steps=2000, deadline=deadline)
    if r_final > best_r:
        best_r = r_final
        best_pts = pts.copy()
        
    return best_pts
