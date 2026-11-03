import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
import scipy.sparse as sp
import scipy.sparse.linalg as spla

N = 300

def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice and select n_points best points."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0
    
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
    Uses sparse edge_indices if provided.
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms
    
    if edge_indices is None:
        # Fallback to full (slow) if no edges provided, but we will always provide them
        d = cdist(pts, pts)
        iu = np.triu_indices(n, k=1)
        pair_half = d[iu] / 2.0
        edges_list = list(zip(iu[0], iu[1]))
    else:
        # Efficient vectorized computation for sparse edges
        i_idx, j_idx = zip(*edge_indices)
        i_idx = np.array(i_idx)
        j_idx = np.array(j_idx)
        diff = pts[i_idx] - pts[j_idx]
        dists = np.linalg.norm(diff, axis=1)
        pair_half = dists / 2.0
        
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
    
    # Gradient from boundary constraints
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    grad_boundary = -pts / safe_norms[:, np.newaxis] * grad_all[len(pair_half):][:, np.newaxis]
    grad_pts += grad_boundary
    
    # Gradient from pair constraints
    if edge_indices is None:
        # This path is slow, but kept for compatibility if needed
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
        # Vectorized gradient for sparse edges
        w_pairs = grad_all[:len(pair_half)]
        # diff is already computed above
        safe_dists = np.where(dists > 1e-10, dists, 1.0)
        unit_vecs = diff / safe_dists[:, np.newaxis]
        
        # grad_i += 0.5 * w * unit_vec
        # grad_j -= 0.5 * w * unit_vec
        weighted_units = 0.5 * w_pairs[:, np.newaxis] * unit_vecs
        
        # Accumulate into i and j
        np.add.at(grad_pts, i_idx, weighted_units)
        np.add.at(grad_pts, j_idx, -weighted_units)
            
    return obj, grad_pts.ravel()

def _force_relax(pts, n_steps=2000, deadline=None):
    """Fast force-directed relaxation with adaptive parameters."""
    pos = pts.copy().astype(np.float64)
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    n = len(pos)
    
    r_est = best_r
    lr = 0.006
    cooldown = 0.9998
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
            
        forces = np.zeros_like(pos)
        
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        target = 2.0 * r_est
        # Only apply repulsion to pairs that are too close
        too_close = dists < target * 1.2
        safe_dists = np.where(too_close, dists, 1.0)
        
        # Use a smoother force profile
        excess = np.where(too_close, target - dists, 0.0)
        force_mag = np.where(too_close, (excess / safe_dists) ** 1.5, 0.0)
        
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        
        forces += np.sum(force_mag[:, :, np.newaxis] * unit_vecs, axis=1)
        
        norms = np.linalg.norm(pos, axis=1)
        wall_margin = 0.05
        in_wall = norms > (1.0 - wall_margin)
        if np.any(in_wall):
            excess_b = np.where(in_wall, norms - (1.0 - wall_margin), 0.0)
            f_mag = 20.0 * excess_b
            safe_norms = np.where(norms > 1e-10, norms, 1.0)
            boundary_force = -pos / safe_norms[:, np.newaxis] * f_mag[:, np.newaxis]
            forces += boundary_force
        
        # Adaptive learning rate based on force magnitude
        force_norm = np.linalg.norm(forces)
        if force_norm > 1e-10:
            step_scale = min(1.0, 0.01 / force_norm)
            pos = pos + lr * step_scale * forces
        else:
            pos = pos + lr * forces
        
        norms = np.linalg.norm(pos, axis=1)
        mask_out = norms > 1.0
        if np.any(mask_out):
            pos[mask_out] = pos[mask_out] / norms[mask_out, np.newaxis] * 0.999
            
        if step % 30 == 0:
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

def _get_delaunay_edges(pts, k_extra=5):
    """
    Get edges from Delaunay triangulation plus k_extra nearest neighbors.
    This ensures we capture all potential contacts.
    """
    n = len(pts)
    try:
        tri = Delaunay(pts)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    if a < b:
                        edges.add((a, b))
                    else:
                        edges.add((b, a))
    except Exception:
        # Fallback to k-NN if Delaunay fails (e.g., duplicate points)
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        for i in range(n):
            k_neighbors = np.argsort(d[i])[:k_extra]
            for j in k_neighbors:
                if i < j:
                    edges.add((i, j))
                else:
                    edges.add((j, i))
        return list(edges)

    # Add k_extra nearest neighbors to be safe (Delaunay might miss some if points are not in general position or due to numerical issues)
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    for i in range(n):
        k_neighbors = np.argsort(d[i])[:k_extra]
        for j in k_neighbors:
            if i < j:
                edges.add((i, j))
            else:
                edges.add((j, i))
                
    return list(edges)

def _refine_with_scipy(pts, deadline):
    """Refine using scipy minimize with soft-min objective on sparse Delaunay graph."""
    n = len(pts)
    x0 = pts.ravel()
    
    # Use a sequence of temperatures.
    # Start with larger T to escape local minima, then decrease to sharpen the min.
    # The sparse edge set allows us to use smaller T efficiently.
    temperatures = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for T in temperatures:
        if time.time() > deadline:
            break
        
        # Get active edges from Delaunay + neighbors
        # We update the graph every temperature step to adapt to new geometry
        edge_indices = _get_delaunay_edges(pts, k_extra=8)
        
        res = minimize(
            _objective_and_grad,
            x0,
            args=(T, edge_indices),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 2000, 'maxfun': 10000, 'maxls': 100, 'maxcor': 50, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        pts = res.x.reshape(-1, 2)
        # Ensure points are inside the circle
        norms = np.linalg.norm(pts, axis=1)
        mask_out = norms > 1.0
        if np.any(mask_out):
            pts[mask_out] = pts[mask_out] / norms[mask_out, np.newaxis] * 0.999
        
        x0 = pts.ravel()

    return pts

def _active_set_newton(pts, deadline, max_iter=100):
    """
    Active-set refinement using gradient ascent on the minimum constraint.
    Uses efficient sparse distance computation.
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
            
        # Identify active constraints efficiently using k-NN
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        
        norms = np.linalg.norm(pts, axis=1)
        boundary_dists = 1.0 - norms
        
        # Active pair constraints: dists < 2*r + epsilon
        epsilon = 1e-5
        active_pairs = []
        iu = np.triu_indices(n, k=1)
        for i, j in zip(iu[0], iu[1]):
            if d[i, j] < 2.0 * r + epsilon:
                active_pairs.append((i, j))
                
        # Active boundary constraints: boundary_dists < epsilon
        active_bounds = [i for i in range(n) if boundary_dists[i] < epsilon]
                
        # If no active constraints, we're at a local optimum
        if len(active_pairs) == 0 and len(active_bounds) == 0:
            break
            
        # Compute gradient of the minimum constraint value
        grad = np.zeros_like(pts)
        
        for i, j in active_pairs:
            diff_ij = pts[i] - pts[j]
            d_ij = d[i, j]
            if d_ij > 1e-10:
                unit_vec = diff_ij / d_ij
                grad[i] += 0.5 * unit_vec
                grad[j] -= 0.5 * unit_vec
                
        for i in active_bounds:
            norm_i = norms[i]
            if norm_i > 1e-10:
                grad[i] -= pts[i] / norm_i
        
        # Adaptive step size based on gradient magnitude
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
            
        step_size = 0.002
        pts_new = pts + step_size * grad
        
        # Ensure points stay inside the circle
        norms_new = np.linalg.norm(pts_new, axis=1)
        mask_out = norms_new > 1.0
        if np.any(mask_out):
            pts_new[mask_out] = pts_new[mask_out] / norms_new[mask_out, np.newaxis] * 0.9999
            
        # Check if this step improves the radius
        r_new = _compute_radius(pts_new)
        if r_new > best_r:
            best_r = r_new
            best_pts = pts_new.copy()
            pts = pts_new
        else:
            # Try smaller step
            for _ in range(5):
                step_size *= 0.5
                pts_new = pts + step_size * grad
                norms_new = np.linalg.norm(pts_new, axis=1)
                mask_out = norms_new > 1.0
                if np.any(mask_out):
                    pts_new[mask_out] = pts_new[mask_out] / norms_new[mask_out, np.newaxis] * 0.9999
                r_new = _compute_radius(pts_new)
                if r_new > best_r:
                    best_r = r_new
                    best_pts = pts_new.copy()
                    pts = pts_new
                    break
            else:
                break
                
    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # 1. Initialization: Hexagonal Lattice with multiple attempts
    pts = _hex_init(N, seed=42)
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    # 2. Force-directed relaxation
    pts, r_force = _force_relax(pts, n_steps=2000, deadline=deadline)
    if r_force > best_r:
        best_r = r_force
        best_pts = pts.copy()
    
    if time.time() > deadline:
        return best_pts
    
    # 3. SciPy refinement with soft-min objective (Sparse Delaunay)
    pts = _refine_with_scipy(pts, deadline)
    r_scipy = _compute_radius(pts)
    if r_scipy > best_r:
        best_r = r_scipy
        best_pts = pts.copy()
    
    if time.time() > deadline:
        return best_pts
    
    # 4. Active-set refinement (Polish)
    pts = _active_set_newton(pts, deadline, max_iter=100)
    r_newton = _compute_radius(pts)
    if r_newton > best_r:
        best_r = r_newton
        best_pts = pts.copy()
    
    if time.time() > deadline:
        return best_pts
    
    # 5. Final force relaxation with smaller steps
    pts, r_final = _force_relax(pts, n_steps=1000, deadline=deadline)
    if r_final > best_r:
        best_r = r_final
        best_pts = pts.copy()
        
    # 6. Final Polish with Active Set again (if time permits)
    if time.time() < deadline - 10:
        pts = _active_set_newton(pts, deadline, max_iter=50)
        r_final2 = _compute_radius(pts)
        if r_final2 > best_r:
            best_r = r_final2
            best_pts = pts.copy()

    return best_pts
