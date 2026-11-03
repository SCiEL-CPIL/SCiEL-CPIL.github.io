import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 300

def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice and select n_points best points."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0
    
    for attempt in range(20):
        base_spacing = 2.0 / (np.sqrt(n_points) + 1.0) * 0.95
        spacing = base_spacing * (0.90 + 0.015 * attempt)
        
        k = int(1.0 / spacing) + 8
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
            jitter = rng.normal(0, spacing * 0.02, candidates.shape)
            candidates = candidates + jitter
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
        k = int(1.0 / spacing) + 8
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

def _get_delaunay_edges(pts):
    """Get Delaunay triangulation edges as a list of (i, j) pairs."""
    try:
        tri = Delaunay(pts)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    if a > b:
                        a, b = b, a
                    edges.add((a, b))
        return list(edges)
    except Exception:
        # Fallback to k-nearest neighbors
        n = len(pts)
        d = cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        edges = set()
        for i in range(n):
            k_neighbors = np.argsort(d[i])[:12]
            for j in k_neighbors:
                if i < j:
                    edges.add((i, j))
                else:
                    edges.add((j, i))
        return list(edges)

def _gauss_newton_contact(pts, deadline, max_iter=500):
    """
    Gauss-Newton refinement enforcing contact constraints.
    For each edge in the contact graph, try to make the distance exactly 2*r.
    For boundary points, try to make |p_i| = 1 - r.
    """
    n = len(pts)
    pos = pts.copy().astype(np.float64)
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        r = _compute_radius(pos)
        if r > best_r:
            best_r = r
            best_pos = pos.copy()
        
        # Get current contact graph via Delaunay
        edges = _get_delaunay_edges(pos)
        
        # Identify which constraints are active (binding)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        norms = np.linalg.norm(pos, axis=1)
        boundary_dists = 1.0 - norms
        
        # Active pairs: distance < 2*r + tolerance
        tol = 1e-6
        active_pairs = []
        iu = np.triu_indices(n, k=1)
        for i, j in zip(iu[0], iu[1]):
            if dists[i, j] < 2.0 * r + tol:
                active_pairs.append((i, j))
        
        # Active boundary: boundary_dist < tolerance
        active_bounds = []
        for i in range(n):
            if boundary_dists[i] < tol:
                active_bounds.append(i)
        
        if len(active_pairs) == 0 and len(active_bounds) == 0:
            break
        
        # Build the Gauss-Newton system
        # Variables: 2*n coordinates
        # Constraints: for each active pair, ||p_i - p_j||^2 - (2r)^2 = 0
        #              for each active bound, ||p_i||^2 - (1-r)^2 = 0
        
        n_constraints = len(active_pairs) + len(active_bounds)
        if n_constraints == 0:
            break
            
        # Jacobian of constraints w.r.t. positions
        # For pair (i,j): grad of (||p_i - p_j||^2 - 4r^2) 
        #   w.r.t. p_i = 2*(p_i - p_j)
        #   w.r.t. p_j = 2*(p_j - p_i) = -2*(p_i - p_j)
        # For bound i: grad of (||p_i||^2 - (1-r)^2)
        #   w.r.t. p_i = 2*p_i
        
        # We want to solve: J * delta_x = -f(x) where f(x) = constraint values
        # But we also want to increase r, so we add a term that pushes r up.
        
        # Actually, let's use a different approach:
        # Minimize the violation of constraints while maximizing r.
        # Use a simple iterative scheme:
        
        # For each active pair, compute the direction to increase the distance
        # For each active bound, compute the direction to decrease the norm
        
        delta = np.zeros_like(pos)
        
        for i, j in active_pairs:
            d_ij = pos[i] - pos[j]
            dist = np.linalg.norm(d_ij)
            if dist < 1e-12:
                continue
            # We want to increase dist to 2*r
            # Move i away from j and j away from i
            unit = d_ij / dist
            # The constraint violation is dist - 2*r (negative if too close)
            # We want to push them apart
            weight = 0.5 * (2.0 * r - dist)
            if weight > 0:
                delta[i] += unit * weight
                delta[j] -= unit * weight
        
        for i in active_bounds:
            norm_i = np.linalg.norm(pos[i])
            if norm_i < 1e-12:
                continue
            # We want to decrease the norm to 1-r
            # Move i toward the origin
            unit = pos[i] / norm_i
            weight = 0.5 * (norm_i - (1.0 - r))
            if weight > 0:
                delta[i] -= unit * weight
        
        # Apply the update with a step size
        # Use line search to ensure we don't make things worse
        step_size = 0.1
        improved = False
        for _ in range(30):
            pos_new = pos + step_size * delta
            
            # Ensure points stay inside the circle
            norms_new = np.linalg.norm(pos_new, axis=1)
            mask_out = norms_new > 1.0
            if np.any(mask_out):
                pos_new[mask_out] = pos_new[mask_out] / norms_new[mask_out, np.newaxis] * 0.999999
            
            r_new = _compute_radius(pos_new)
            if r_new > best_r:
                best_r = r_new
                best_pos = pos_new.copy()
                pos = pos_new
                improved = True
                break
            step_size *= 0.5
        
        if not improved:
            # Try a smaller step or different approach
            step_size = 0.01
            for _ in range(30):
                pos_new = pos + step_size * delta
                norms_new = np.linalg.norm(pos_new, axis=1)
                mask_out = norms_new > 1.0
                if np.any(mask_out):
                    pos_new[mask_out] = pos_new[mask_out] / norms_new[mask_out, np.newaxis] * 0.999999
                
                r_new = _compute_radius(pos_new)
                if r_new > best_r:
                    best_r = r_new
                    best_pos = pos_new.copy()
                    pos = pos_new
                    improved = True
                    break
                step_size *= 0.5
            
            if not improved:
                break
    
    return best_pos, best_r

def _force_relax(pts, n_steps=2000, deadline=None):
    """Fast force-directed relaxation."""
    pos = pts.copy().astype(np.float64)
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    n = len(pos)
    
    r_est = best_r
    lr = 0.008
    cooldown = 0.9995
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
            
        forces = np.zeros_like(pos)
        
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        target = 2.0 * r_est
        too_close = dists < target * 1.5
        safe_dists = np.where(too_close, dists, 1.0)
        
        excess = np.where(too_close, target - dists, 0.0)
        force_mag = np.where(too_close, excess / safe_dists, 0.0)
        
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        
        forces += np.sum(force_mag[:, :, np.newaxis] * unit_vecs, axis=1)
        
        norms = np.linalg.norm(pos, axis=1)
        wall_margin = 0.08
        in_wall = norms > (1.0 - wall_margin)
        if np.any(in_wall):
            excess_b = np.where(in_wall, norms - (1.0 - wall_margin), 0.0)
            f_mag = 15.0 * excess_b
            safe_norms = np.where(norms > 1e-10, norms, 1.0)
            boundary_force = -pos / safe_norms[:, np.newaxis] * f_mag[:, np.newaxis]
            forces += boundary_force
        
        pos = pos + lr * forces
        
        norms = np.linalg.norm(pos, axis=1)
        mask_out = norms > 1.0
        if np.any(mask_out):
            pos[mask_out] = pos[mask_out] / norms[mask_out, np.newaxis] * 0.998
            
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

def _refine_with_softmin(pts, deadline):
    """Refine using scipy minimize with soft-min objective on Delaunay edges."""
    n = len(pts)
    x0 = pts.ravel()
    
    def _objective(x):
        pts_local = x.reshape(-1, 2)
        norms = np.linalg.norm(pts_local, axis=1)
        boundary = 1.0 - norms
        
        # Use Delaunay edges for pair constraints
        try:
            tri = Delaunay(pts_local)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        a, b = simplex[i], simplex[j]
                        if a > b:
                            a, b = b, a
                        edges.add((a, b))
            edge_list = list(edges)
        except Exception:
            # Fallback
            d = cdist(pts_local, pts_local)
            np.fill_diagonal(d, np.inf)
            edge_list = []
            for i in range(n):
                k_neighbors = np.argsort(d[i])[:12]
                for j in k_neighbors:
                    if i < j:
                        edge_list.append((i, j))
                    else:
                        edge_list.append((j, i))
        
        pair_half = np.array([np.linalg.norm(pts_local[i] - pts_local[j]) / 2.0 for i, j in edge_list])
        
        all_vals = np.concatenate([pair_half, boundary])
        
        T = 0.005
        neg_vals = -all_vals / T
        max_neg = np.max(neg_vals)
        if max_neg > 700:
            return -all_vals.min()
        exp_shifted = np.exp(neg_vals - max_neg)
        log_sum = np.log(np.sum(exp_shifted)) + max_neg
        return -T * log_sum
    
    # Use L-BFGS-B
    res = minimize(
        _objective,
        x0,
        method='L-BFGS-B',
        options={'maxiter': 2000, 'maxfun': 10000, 'maxls': 100}
    )
    
    pts = res.x.reshape(-1, 2)
    norms = np.linalg.norm(pts, axis=1)
    mask_out = norms > 1.0
    if np.any(mask_out):
        pts[mask_out] = pts[mask_out] / norms[mask_out, np.newaxis] * 0.999999
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # 1. Initialization: Hexagonal Lattice with multiple attempts
    pts = _hex_init(N, seed=42)
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    # 2. Force-directed relaxation
    pts, r_force = _force_relax(pts, n_steps=3000, deadline=deadline)
    if r_force > best_r:
        best_r = r_force
        best_pts = pts.copy()
    
    if time.time() > deadline:
        return best_pts
    
    # 3. Gauss-Newton contact refinement
    pts, r_gn = _gauss_newton_contact(pts, deadline, max_iter=500)
    if r_gn > best_r:
        best_r = r_gn
        best_pts = pts.copy()
    
    if time.time() > deadline:
        return best_pts
    
    # 4. Soft-min refinement
    pts = _refine_with_softmin(pts, deadline)
    r_soft = _compute_radius(pts)
    if r_soft > best_r:
        best_r = r_soft
        best_pts = pts.copy()
    
    if time.time() > deadline:
        return best_pts
    
    # 5. Another round of Gauss-Newton
    pts, r_gn2 = _gauss_newton_contact(pts, deadline, max_iter=300)
    if r_gn2 > best_r:
        best_r = r_gn2
        best_pts = pts.copy()
    
    if time.time() > deadline:
        return best_pts
    
    # 6. Final force relaxation
    pts, r_final = _force_relax(pts, n_steps=1000, deadline=deadline)
    if r_final > best_r:
        best_r = r_final
        best_pts = pts.copy()
    
    if time.time() > deadline:
        return best_pts
    
    # 7. Final Gauss-Newton polish
    pts, r_polish = _gauss_newton_contact(pts, deadline, max_iter=200)
    if r_polish > best_r:
        best_r = r_polish
        best_pts = pts.copy()
    
    return best_pts
