"""
Packing 50 equal circles in a unit disk using soft-min optimization with
force-directed pre-conditioning and multi-stage beta annealing.

Combined approach:
- Soft-min objective (from inspiration) for smooth, differentiable optimization
- Force-directed relaxation as pre-conditioner
- Multi-stage beta annealing for better convergence
- Contact-graph-based active set refinement (from current program)
- Binary search on radius for feasibility checking
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
_START_TIME = None
_DEADLINE = 140.0  # safety margin before 150s kill


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    dists_origin = np.linalg.norm(pts, axis=1)
    if np.any(dists_origin >= 1.0):
        return 0.0
    
    r_cont = 1.0 - np.max(dists_origin)
    if r_cont < 0:
        return 0.0
        
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)
    
    return min(r_cont, r_pair)


def _soft_min_obj_and_grad(flat, beta):
    """
    Compute the soft-min objective and gradient for L-BFGS-B.
    Objective: minimize f(x) = (1/beta) * ln(sum(exp(-beta * x_k)))
    where x_k are the critical distances (pairwise/2 and boundary).
    Minimizing this is equivalent to maximizing the soft-min of x_k.
    """
    pts = flat.reshape(N, 2)
    
    # Calculate boundary distances: 1 - |c_i|
    dists_origin = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - dists_origin
    
    # Calculate pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
    
    iu = np.triu_indices(N, k=1)
    pair_dists = dist_mat[iu]
    
    # Combine all critical distances
    all_dists = np.concatenate([boundary_dists, pair_dists / 2.0])
    
    # Soft min calculation with numerical stability
    y = -beta * all_dists
    max_y = np.max(y)
    y_shifted = y - max_y
    sum_exp = np.sum(np.exp(y_shifted))
    log_sum_exp = max_y + np.log(sum_exp)
    obj_val = log_sum_exp / beta
    
    # Gradient
    weights = np.exp(y_shifted) / sum_exp
    
    grad_pts = np.zeros_like(pts)
    
    # 1. Boundary terms
    w_boundary = weights[0:N]
    safe_dists = np.where(dists_origin < 1e-8, 1.0, dists_origin)
    boundary_grad = (w_boundary / safe_dists)[:, np.newaxis] * pts
    grad_pts += boundary_grad
    
    # 2. Pairwise terms
    w_pairs = weights[N:]
    i_idx, m_idx = iu
    d_pairs = dist_mat[i_idx, m_idx]
    safe_d_pairs = np.where(d_pairs < 1e-8, 1.0, d_pairs)
    vecs = pts[i_idx] - pts[m_idx]
    unit_vecs = vecs / safe_d_pairs[:, np.newaxis]
    
    grad_contrib_i = -0.5 * w_pairs[:, np.newaxis] * unit_vecs
    grad_contrib_m = 0.5 * w_pairs[:, np.newaxis] * unit_vecs
    
    np.add.at(grad_pts, i_idx, grad_contrib_i)
    np.add.at(grad_pts, m_idx, grad_contrib_m)
    
    grad_flat = grad_pts.flatten()
    
    return obj_val, grad_flat


def _force_directed_relax(pts, n_steps=300, target_r=0.12):
    """
    Force-directed relaxation to improve packing.
    Pushes overlapping circles apart and pulls them toward the center.
    """
    pts = pts.copy()
    target_d = 2.0 * target_r
    
    for step in range(n_steps):
        if _START_TIME is not None and time.time() - _START_TIME > _DEADLINE - 5:
            break
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist_mat, np.inf)
        
        # Only consider pairs that are too close
        for i in range(N):
            for j in range(i + 1, N):
                d = dist_mat[i, j]
                if d < target_d and d > 1e-10:
                    overlap = target_d - d
                    direction = (pts[i] - pts[j]) / d
                    force_mag = overlap * 0.5
                    forces[i] += force_mag * direction
                    forces[j] -= force_mag * direction
        
        # Boundary constraint: push circles inward if they're outside
        dists_origin = np.linalg.norm(pts, axis=1)
        for i in range(N):
            if dists_origin[i] > 1.0 - target_r:
                direction = -pts[i] / dists_origin[i]
                excess = dists_origin[i] - (1.0 - target_r)
                forces[i] += excess * direction
        
        # Update positions with adaptive learning rate
        lr = 0.01 * (1.0 - step / max(n_steps, 1)) + 0.001
        pts += lr * forces
        
        # Project to unit disk
        dists = np.linalg.norm(pts, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / dists[mask, np.newaxis]
    
    return pts


def _hex_lattice_init(r_guess=0.12):
    """Generate a hexagonal lattice packing as initial guess."""
    spacing = 2.0 * r_guess
    pts = []
    k = int(1.0 / spacing) + 3
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= (1.0 - r_guess) ** 2:
                pts.append([x, y])
    pts = np.array(pts)
    if len(pts) >= N:
        radii = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(radii, kind='stable')[:N]
        return pts[idx]
    else:
        return _ring_init([1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.95])


def _ring_init(ring_sizes, ring_radii, angle_offsets=None):
    """Generate points from concentric rings."""
    points = np.zeros((N, 2))
    idx = 0
    if angle_offsets is None:
        angle_offsets = [0.0] * len(ring_sizes)
    for ring_idx, (size, radius) in enumerate(zip(ring_sizes, ring_radii)):
        offset = angle_offsets[ring_idx] if ring_idx < len(angle_offsets) else 0.0
        for j in range(size):
            angle = 2 * np.pi * j / size + offset
            points[idx] = [radius * np.cos(angle), radius * np.sin(angle)]
            idx += 1
    return points


def _known_structure_init():
    """
    Initialize with the known optimal structure: 1 + 6 + 12 + 18 + 13 rings.
    """
    ring_sizes = [1, 6, 12, 18, 13]
    ring_radii = [0.0, 0.24, 0.48, 0.72, 0.94]
    
    return _ring_init(ring_sizes, ring_radii)


def _soft_min_optimize(pts_init, beta_schedule, max_iter_per_stage=500):
    """
    Multi-stage soft-min optimization with beta annealing.
    """
    current_pts = pts_init.copy()
    
    for stage, beta in enumerate(beta_schedule):
        if _START_TIME is not None and time.time() - _START_TIME > _DEADLINE - 10:
            break
        
        def objective(x):
            return _soft_min_obj_and_grad(x, beta)
        
        x0 = current_pts.flatten()
        result = minimize(objective, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': max_iter_per_stage, 'ftol': 1e-14,
                                   'gtol': 1e-10, 'maxls': 50})
        current_pts = result.x.reshape(N, 2)
        
        # Project to ensure validity
        dists = np.linalg.norm(current_pts, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            current_pts[mask] = current_pts[mask] / dists[mask, np.newaxis]
    
    return current_pts


def _build_contact_graph(pts, r_target, k_nearest=6):
    """
    Build a contact graph: identify pairs of circles that are close to touching.
    """
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_mat, np.inf)
    
    threshold = 2.0 * r_target * 1.05
    iu = np.triu_indices(N, k=1)
    active_pairs = []
    for idx in range(len(iu[0])):
        if dist_mat[iu[0][idx], iu[1][idx]] <= threshold:
            active_pairs.append((iu[0][idx], iu[1][idx]))
    
    dists_origin = np.linalg.norm(pts, axis=1)
    boundary_active = []
    for i in range(N):
        if dists_origin[i] >= 1.0 - r_target * 1.05:
            boundary_active.append(i)
    
    return active_pairs, boundary_active


def _feasibility_search(pts_init, r_low, r_high, n_steps=8, max_iter_per_step=200):
    """
    Binary search on r to find the maximum feasible radius.
    """
    best_pts = pts_init.copy()
    best_r = _score(pts_init)
    
    for step in range(n_steps):
        if _START_TIME is not None and time.time() - _START_TIME > _DEADLINE - 15:
            break
        
        r_mid = (r_low + r_high) / 2.0
        r_mid = max(r_mid, best_r + 1e-6)
        
        # Use soft-min optimization to try to achieve r_mid
        pts_candidate = _soft_min_optimize(pts_init, [50, 100, 200], max_iter_per_stage=150)
        
        # Project to unit disk
        dists = np.linalg.norm(pts_candidate, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            pts_candidate[mask] = pts_candidate[mask] / dists[mask, np.newaxis]
        
        score_candidate = _score(pts_candidate)
        
        if score_candidate >= r_mid - 1e-8:
            r_low = r_mid
            best_pts = pts_candidate.copy()
            best_r = score_candidate
        else:
            r_high = r_mid
        
        if best_r > _score(pts_init):
            pts_init = best_pts.copy()
    
    return best_pts, best_r


def _contact_graph_refine(pts, r_target, n_iterations=30):
    """
    Refine packing using contact graph analysis with soft-min optimization.
    """
    pts = pts.copy()
    
    for it in range(n_iterations):
        if _START_TIME is not None and time.time() - _START_TIME > _DEADLINE - 10:
            break
        
        # Build contact graph
        active_pairs, boundary_active = _build_contact_graph(pts, r_target)
        
        if not active_pairs and not boundary_active:
            break
        
        # Run soft-min optimization with high beta (focus on critical distances)
        beta = 200 + 10 * it  # Increasingly sharp
        pts_new = _soft_min_optimize(pts, [beta], max_iter_per_stage=100)
        
        # Project to unit disk
        dists = np.linalg.norm(pts_new, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            pts_new[mask] = pts_new[mask] / dists[mask, np.newaxis]
        
        score_new = _score(pts_new)
        score_old = _score(pts)
        
        if score_new >= score_old - 1e-10:
            pts = pts_new
        else:
            # Try with relaxed target
            r_relaxed = r_target * 0.95
            pts_new = _soft_min_optimize(pts, [100], max_iter_per_stage=100)
            dists = np.linalg.norm(pts_new, axis=1)
            mask = dists > 1.0
            if np.any(mask):
                pts_new[mask] = pts_new[mask] / dists[mask, np.newaxis]
            score_new = _score(pts_new)
            if score_new > score_old:
                pts = pts_new
            else:
                break
    
    return pts


def _full_optimize(points, budget_seconds=25.0):
    """
    Full optimization pipeline:
    Phase 1: Force-directed pre-conditioning
    Phase 2: Multi-stage soft-min optimization with beta annealing
    Phase 3: Binary search on radius
    Phase 4: Contact graph refinement
    """
    t0 = time.time()
    best_points = points.copy()
    best_score = _score(points)
    
    # Phase 1: Force-directed pre-conditioning
    if time.time() - t0 < budget_seconds * 0.15:
        target_r = max(best_score, 0.12)
        pts_fd = _force_directed_relax(points, n_steps=200, target_r=target_r)
        score_fd = _score(pts_fd)
        if score_fd > best_score:
            best_score = score_fd
            best_points = pts_fd.copy()
    
    # Phase 2: Multi-stage soft-min optimization
    if time.time() - t0 < budget_seconds * 0.60:
        beta_schedule = [20, 50, 100, 200, 400]
        remaining = budget_seconds - (time.time() - t0)
        max_iter = max(100, int(remaining * 20))
        
        pts_opt = _soft_min_optimize(best_points, beta_schedule, max_iter_per_stage=max_iter)
        score_opt = _score(pts_opt)
        if score_opt > best_score:
            best_score = score_opt
            best_points = pts_opt.copy()
    
    # Phase 3: Binary search on radius
    if time.time() - t0 < budget_seconds * 0.85:
        r_low = max(best_score, 0.12)
        r_high = min(0.135, r_low + 0.012)
        
        remaining = budget_seconds - (time.time() - t0)
        n_steps = max(4, int(remaining / 4.0))
        
        pts_feas, r_feas = _feasibility_search(best_points, r_low, r_high,
                                                n_steps=n_steps, max_iter_per_step=150)
        if r_feas > best_score:
            best_score = r_feas
            best_points = pts_feas.copy()
    
    # Phase 4: Contact graph refinement
    if time.time() - t0 < budget_seconds * 0.98:
        r_target = best_score * 1.001
        pts_refined = _contact_graph_refine(best_points, r_target, n_iterations=20)
        score_refined = _score(pts_refined)
        if score_refined > best_score:
            best_score = score_refined
            best_points = pts_refined.copy()
    
    return best_points, best_score


def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_points = None
    best_score = 0.0
    
    rng = np.random.RandomState(42)
    
    # Strategy 1: Known structure initialization
    known_pts = _known_structure_init()
    known_score = _score(known_pts)
    if known_score > best_score:
        best_score = known_score
        best_points = known_pts.copy()
    
    # Strategy 2: Hex lattice initialization
    hex_pts = _hex_lattice_init(0.125)
    hex_score = _score(hex_pts)
    if hex_score > best_score:
        best_score = hex_score
        best_points = hex_pts.copy()
    
    # Strategy 3: Multiple ring configurations with full optimization
    ring_configs = [
        ([1, 6, 12, 18, 13], [0.0, 0.24, 0.48, 0.72, 0.94]),
        ([1, 6, 12, 18, 13], [0.0, 0.22, 0.45, 0.68, 0.90]),
        ([1, 6, 12, 18, 13], [0.0, 0.26, 0.52, 0.76, 0.96]),
    ]
    
    for config_idx, (ring_sizes, radii) in enumerate(ring_configs):
        if time.time() - _START_TIME > _DEADLINE - 40:
            break
        
        points = _ring_init(ring_sizes, radii)
        if config_idx > 0:
            points += rng.uniform(-0.005, 0.005, size=(N, 2))
        
        remaining = _DEADLINE - (time.time() - _START_TIME) - 25
        budget = min(22.0, remaining)
        if budget < 10:
            break
            
        pts_opt, score_opt = _full_optimize(points, budget_seconds=budget)
        if score_opt > best_score:
            best_score = score_opt
            best_points = pts_opt.copy()
    
    # Strategy 4: Multi-start with perturbation from best
    n_starts = 2
    for s in range(n_starts):
        if time.time() - _START_TIME > _DEADLINE - 20:
            break
        
        if s == 0:
            pts_init = best_points.copy()
            noise = rng.normal(0, 0.008, size=pts_init.shape)
        else:
            pts_init = _known_structure_init()
            noise = rng.normal(0, 0.012, size=pts_init.shape)
        
        pts_init += noise
        norms = np.hypot(pts_init[:, 0], pts_init[:, 1])
        r_max = 1.0 - max(best_score, 0.12)
        for i in range(N):
            if norms[i] > r_max:
                pts_init[i] *= r_max / norms[i]
        
        remaining = _DEADLINE - (time.time() - _START_TIME) - 10
        if remaining < 12:
            break
            
        pts_opt, score_opt = _full_optimize(pts_init, budget_seconds=min(18.0, remaining))
        if score_opt > best_score:
            best_score = score_opt
            best_points = pts_opt.copy()
    
    # Final contact graph refinement
    if best_points is not None and time.time() - _START_TIME < _DEADLINE - 5:
        r_current = _score(best_points)
        if r_current > 0.1:
            pts_final = _contact_graph_refine(best_points, r_current * 1.0005, n_iterations=15)
            score_final = _score(pts_final)
            if score_final > best_score:
                best_score = score_final
                best_points = pts_final.copy()
    
    if best_points is None:
        best_points = _known_structure_init()
    
    # Ensure validity
    dists = np.linalg.norm(best_points, axis=1)
    for i in range(N):
        if dists[i] >= 1.0:
            best_points[i] *= 0.999 / max(dists[i], 1e-10)
    
    return best_points


# EVOLVE-BLOCK-START
# The main construct function is defined above
# EVOLVE-BLOCK-END
