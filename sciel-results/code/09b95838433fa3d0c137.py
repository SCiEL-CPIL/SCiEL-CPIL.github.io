import numpy as np
import time
from scipy.spatial import cKDTree

N = 300

def _compute_radius(pts):
    """Compute the packing radius for given points."""
    if len(pts) == 0:
        return 0.0
    norms = np.linalg.norm(pts, axis=1)
    min_bound = 1.0 - norms.max()
    if min_bound <= 0:
        return 0.0
    tree = cKDTree(pts)
    # Query k=2 to get nearest neighbor (excluding self)
    dists, _ = tree.query(pts, k=2)
    min_pair = dists[:, 1].min() / 2.0
    return min(min_pair, min_bound)

def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice and select n_points best points."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0
    
    for attempt in range(20):
        base_spacing = 2.0 / np.sqrt(n_points) * 0.95
        spacing = base_spacing * (0.90 + 0.02 * attempt)
        
        k = int(1.2 / spacing) + 5
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
            noise_scale = spacing * 0.02
            pts = pts + rng.normal(0, noise_scale, pts.shape)
            
        norms = np.linalg.norm(pts, axis=1)
        mask = norms < 1.0
        pts_inside = pts[mask]
        
        if len(pts_inside) < n_points:
            continue
            
        order = np.argsort(norms[mask], kind="stable")
        selected = pts_inside[order[:n_points]]
        
        r = _compute_radius(selected)
        
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

def _force_relax(pts, n_steps=2000, deadline=None):
    """Fast force-directed relaxation using KDTree for neighbor search."""
    pos = pts.copy().astype(np.float64)
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    n = len(pos)
    
    r_est = best_r
    lr = 0.005
    cooldown = 0.9995
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
            
        forces = np.zeros_like(pos)
        
        tree = cKDTree(pos)
        # Search for neighbors within 2*r_est + margin
        search_radius = 2.5 * r_est
        pairs = tree.query_pairs(search_radius, output_type='ndarray')
        
        for i, j in pairs:
            diff = pos[i] - pos[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                continue
            target = 2.0 * r_est
            if dist < target:
                excess = target - dist
                force_mag = 10.0 * excess / dist
                unit_vec = diff / dist
                forces[i] += force_mag * unit_vec
                forces[j] -= force_mag * unit_vec
        
        norms = np.linalg.norm(pos, axis=1)
        wall_margin = 0.05
        for i in range(n):
            if norms[i] > 1.0 - wall_margin:
                excess_b = norms[i] - (1.0 - wall_margin)
                if excess_b > 0:
                    f_mag = 20.0 * excess_b
                    if norms[i] > 1e-10:
                        direction = -pos[i] / norms[i]
                    else:
                        direction = np.array([1.0, 0.0])
                    forces[i] += f_mag * direction
        
        pos = pos + lr * forces
        
        norms = np.linalg.norm(pos, axis=1)
        mask_out = norms > 1.0
        if np.any(mask_out):
            pos[mask_out] = pos[mask_out] / norms[mask_out, np.newaxis] * 0.999
            
        if step % 100 == 0:
            r = _compute_radius(pos)
            if r > best_r:
                best_r = r
                best_pos = pos.copy()
                r_est = r
                lr = min(0.02, lr * 1.1)
                
        lr *= cooldown
        
    return best_pos, best_r

def _newton_refine(pts, deadline=None):
    """
    Active-set Newton refinement:
    1. Identify the minimum constraint (pair distance or boundary)
    2. Build a small active set of constraints near the minimum
    3. Solve a small QP-like system to move points to increase the minimum
    """
    pos = pts.copy().astype(np.float64)
    n = len(pos)
    best_r = _compute_radius(pos)
    
    max_iterations = 50
    for it in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
            
        # Compute current radius and identify critical constraints
        norms = np.linalg.norm(pos, axis=1)
        boundary = 1.0 - norms
        
        tree = cKDTree(pos)
        dists, indices = tree.query(pos, k=2)
        pair_dists = dists[:, 1]
        pair_indices = indices[:, 1]
        
        # Find the minimum of all constraints
        min_pair_idx = np.argmin(pair_dists)
        min_pair_val = pair_dists[min_pair_idx] / 2.0
        min_bound_idx = np.argmin(boundary)
        min_bound_val = boundary[min_bound_idx]
        
        current_r = min(min_pair_val, min_bound_val)
        
        # Build active set: constraints within 10% of the minimum
        threshold = current_r * 1.1 + 1e-10
        
        active_pairs = []
        for i in range(n):
            if pair_dists[i] / 2.0 <= threshold:
                active_pairs.append((i, pair_indices[i]))
        
        active_bounds = []
        for i in range(n):
            if boundary[i] <= threshold:
                active_bounds.append(i)
        
        if len(active_pairs) == 0 and len(active_bounds) == 0:
            break
            
        # Build the linear system for Newton step
        # We want to find delta_pos such that all active constraints increase
        # For a pair (i,j): d/dp_i (|p_i - p_j|) = (p_i - p_j)/|p_i - p_j|
        # For boundary i: d/dp_i (1 - |p_i|) = -p_i/|p_i|
        
        # Simple approach: gradient ascent on the soft minimum
        # But let's try a direct active set approach
        
        # Collect all active constraints
        all_active = []
        for (i, j) in active_pairs:
            diff = pos[i] - pos[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-10:
                grad_i = diff / dist
                grad_j = -diff / dist
                all_active.append((i, j, grad_i, grad_j, pair_dists[i] / 2.0))
        
        for i in active_bounds:
            norm_i = norms[i]
            if norm_i > 1e-10:
                grad_i = -pos[i] / norm_i
                all_active.append((i, -1, grad_i, None, boundary[i]))
        
        if len(all_active) == 0:
            break
            
        # Compute weighted gradient (weights proportional to how close to minimum)
        grad_sum = np.zeros_like(pos)
        for (i, j, gi, gj, val) in all_active:
            weight = 1.0 / (1.0 + (current_r - val) * 1000)
            grad_sum[i] += weight * gi
            if j >= 0 and gj is not None:
                grad_sum[j] += weight * gj
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad_sum)
        if grad_norm < 1e-15:
            break
        grad_dir = grad_sum / grad_norm
        
        # Line search
        step_size = 0.01
        improved = False
        for _ in range(10):
            new_pos = pos + step_size * grad_dir
            new_norms = np.linalg.norm(new_pos, axis=1)
            if np.any(new_norms >= 1.0):
                step_size *= 0.5
                continue
            new_r = _compute_radius(new_pos)
            if new_r > current_r + 1e-12:
                pos = new_pos
                best_r = max(best_r, new_r)
                improved = True
                break
            step_size *= 0.5
        
        if not improved:
            break
            
    return pos, best_r

def _anneal_restart(pts, deadline=None, n_restarts=5):
    """
    Perturbation-based restarts:
    Take the best solution, add random perturbations, and re-relax.
    This helps escape local minima.
    """
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    n = len(best_pts)
    rng = np.random.default_rng(12345)
    
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline:
            break
            
        # Perturb a random subset of points
        n_perturb = max(10, n // 5)
        idx = rng.choice(n, size=n_perturb, replace=False)
        scale = 0.01 * best_r
        
        perturbed = best_pts.copy()
        perturbed[idx] += rng.normal(0, scale, (n_perturb, 2))
        
        # Keep inside unit circle
        norms = np.linalg.norm(perturbed, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            perturbed[mask] = perturbed[mask] / norms[mask, np.newaxis] * 0.999
            
        # Relax the perturbed solution
        relaxed, r = _force_relax(perturbed, n_steps=1000, deadline=deadline)
        
        if r > best_r:
            best_r = r
            best_pts = relaxed.copy()
            
    return best_pts, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    best_pts = None
    best_r = 0.0
    
    # Phase 1: Multiple initializations
    inits = [
        _hex_init(N, seed=42),
        _hex_init(N, seed=123),
        _hex_init(N, seed=999),
        _hex_init(N, seed=777),
        _hex_init(N, seed=555),
    ]
    
    for pts in inits:
        if time.time() > deadline - 40.0:
            break
            
        r = _compute_radius(pts)
        
        # Force relaxation
        relaxed_pts, relaxed_r = _force_relax(pts, n_steps=2000, deadline=deadline - 30.0)
        
        if relaxed_r > r:
            pts = relaxed_pts
            r = relaxed_r
        
        # Newton refinement
        if time.time() < deadline - 20.0:
            newton_pts, newton_r = _newton_refine(pts, deadline=deadline - 15.0)
            if newton_r > r:
                pts = newton_pts
                r = newton_r
        
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
            
    if best_pts is None:
        best_pts = _hex_init(N, seed=0)
        best_r = _compute_radius(best_pts)
        
    # Phase 2: Anneal restarts from best solution
    if time.time() < deadline - 20.0:
        annealed_pts, annealed_r = _anneal_restart(
            best_pts, deadline=deadline - 10.0, n_restarts=8
        )
        if annealed_r > best_r:
            best_r = annealed_r
            best_pts = annealed_pts.copy()
        
        # Additional Newton refinement on the annealed result
        if time.time() < deadline - 5.0:
            final_pts, final_r = _newton_refine(best_pts, deadline=deadline - 2.0)
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
