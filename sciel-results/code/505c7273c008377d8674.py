"""Improved packer: 12-fold symmetric parameterization with L-BFGS-B soft-min optimization."""
import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 300
SEED = 12345
DEADLINE = 115.0  # safety margin before the 120 s wall-clock limit
NSYM = 12  # 12-fold rotational symmetry


def _score(points):
    """Return the feasible common radius (0.0 if invalid)."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    norm = np.linalg.norm(points, axis=1)
    r_bound = 1.0 - norm
    if np.min(r_bound) < 0.0:
        return 0.0
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    r_pair = 0.5 * dists[:, 1].min()
    r = min(r_bound.min(), r_pair)
    return float(r) if r > 0.0 else 0.0


def _hex_lattice_init(spacing):
    """Generate a hexagonal lattice and pick N points inside the disk."""
    while True:
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        pts = np.array(rows)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        keep = rad <= 1.0 - spacing / 2.0
        pts = pts[keep]
        if len(pts) >= N:
            order = np.argsort(np.hypot(pts[:, 0], pts[:, 1]), kind="stable")
            return pts[order[:N]].copy()
        spacing *= 0.97


def _radial_ring_init():
    """Generate a ring-based initialization with 12-fold symmetry."""
    pts = []
    # Center point
    pts.append([0.0, 0.0])
    # Rings with 12-fold symmetry
    for ring in range(1, 20):
        r = ring * 0.055
        if r > 0.95:
            break
        n = 12 * ring  # 12 points per ring, increasing
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        for a in angles:
            pts.append([r * np.cos(a), r * np.sin(a)])
    pts = np.array(pts[:N])
    rad = np.linalg.norm(pts, axis=1)
    mask = rad > 0.99
    if np.any(mask):
        pts[mask] = pts[mask] / rad[mask, None] * 0.99
    return pts


def _compute_true_score(points):
    """Compute the true packing radius."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    norm = np.linalg.norm(points, axis=1)
    r_bound = 1.0 - norm
    if np.min(r_bound) < 0.0:
        return 0.0
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    r_pair = 0.5 * dists[:, 1].min()
    r = min(r_bound.min(), r_pair)
    return float(r) if r > 0.0 else 0.0


def _sym_params_to_points(params):
    """
    Convert symmetric parameter vector to N x 2 points.
    
    Parameters:
    - First parameter: center radius (if > 0, place a point at center)
    - Then for each of NSYM sectors:
      - For each ring in that sector: (radius, angular_offset)
    
    We use a structured approach:
    - params[0]: whether to include center (0 or 1)
    - Then groups of (r, theta) pairs, repeated NSYM times with rotation
    """
    points = []
    idx = 0
    
    # Center point
    if params[0] > 0.5:
        points.append([0.0, 0.0])
    
    # Remaining parameters define orbits under 12-fold rotation
    # Each orbit: (radius, base_angle) -> generates 12 points
    remaining = params[1:]
    n_orbits = len(remaining) // 2
    
    for o in range(n_orbits):
        r = remaining[2 * o]
        theta_base = remaining[2 * o + 1]
        for k in range(NSYM):
            theta = theta_base + 2 * np.pi * k / NSYM
            points.append([r * np.cos(theta), r * np.sin(theta)])
    
    pts = np.array(points[:N])
    if len(pts) < N:
        # Pad with points near boundary if needed
        while len(pts) < N:
            pts = np.vstack([pts, [[0.95, 0.0]]])
    return pts[:N]


def _points_to_sym_params(points):
    """
    Convert N x 2 points to symmetric parameter vector.
    This is approximate - we find the best 12-fold symmetric fit.
    """
    params = []
    
    # Check if center has a point
    center_dist = np.min(np.linalg.norm(points, axis=1))
    if center_dist < 0.02:
        params.append(1.0)
    else:
        params.append(0.0)
    
    # For each sector, find representative points
    # Assign each point to its sector
    angles = np.arctan2(points[:, 1], points[:, 0])
    angles = (angles + 2 * np.pi) % (2 * np.pi)
    sector_angles = 2 * np.pi / NSYM
    sectors = np.floor(angles / sector_angles).astype(int) % NSYM
    
    # For each sector, collect points and find orbits
    used = np.zeros(N, dtype=bool)
    orbits = []
    
    for sec in range(NSYM):
        sec_pts = points[sectors == sec]
        sec_angles = angles[sectors == sec]
        sec_indices = np.where(sectors == sec)[0]
        
        # Sort by radius
        radii = np.linalg.norm(sec_pts, axis=1)
        order = np.argsort(radii)
        
        for idx in order:
            i = sec_indices[idx]
            if used[i]:
                continue
            r = radii[idx]
            # Angular offset within sector
            theta_offset = sec_angles[idx] - sec * sector_angles
            orbits.append((r, theta_offset))
            # Mark all points in this orbit as used
            for k in range(NSYM):
                target_angle = theta_offset + 2 * np.pi * k / NSYM
                # Find nearest point
                target_x = r * np.cos(target_angle)
                target_y = r * np.sin(target_angle)
                dists = np.linalg.norm(points - np.array([target_x, target_y]), axis=1)
                nearest = np.argmin(dists)
                if dists[nearest] < 0.05:
                    used[nearest] = True
    
    # Deduplicate orbits (keep unique ones)
    unique_orbits = []
    for orbit in orbits:
        is_dup = False
        for uo in unique_orbits:
            if abs(orbit[0] - uo[0]) < 0.02 and abs(orbit[1] - uo[1]) < 0.1:
                is_dup = True
                break
        if not is_dup:
            unique_orbits.append(orbit)
    
    for r, theta in unique_orbits:
        params.append(r)
        params.append(theta)
    
    return np.array(params)


def _soft_min_obj_and_grad(params, beta=100.0):
    """
    Compute soft-min objective and gradient for the packing problem.
    Maximizes the minimum of (pair distances / 2, boundary distances).
    
    We minimize -soft_min where soft_min approximates the min.
    """
    points = _sym_params_to_points(params)
    if len(points) < 2:
        return 0.0, np.zeros_like(params)
    
    # Compute all relevant distances
    norms = np.linalg.norm(points, axis=1)
    boundary_dists = 1.0 - norms
    
    # Pairwise distances using KD-tree
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    pair_dists = 0.5 * dists[:, 1]
    
    # Combine all constraints
    all_dists = np.concatenate([boundary_dists, pair_dists])
    
    # Soft min: -log(sum(exp(-beta * x))) / beta
    # For numerical stability
    min_val = np.min(all_dists)
    shifted = all_dists - min_val
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min = min_val - log_sum / beta
    
    # Objective: we want to maximize soft_min, so minimize -soft_min
    obj = -soft_min
    
    # Gradient via finite differences (for simplicity, could do analytic)
    grad = np.zeros_like(params)
    eps = 1e-6
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        p_plus = _sym_params_to_points(params_plus)
        if len(p_plus) < 2:
            grad[i] = 0.0
            continue
        norms_plus = np.linalg.norm(p_plus, axis=1)
        bd_plus = 1.0 - norms_plus
        tree_plus = cKDTree(p_plus)
        d_plus, _ = tree_plus.query(p_plus, k=2)
        pd_plus = 0.5 * d_plus[:, 1]
        all_plus = np.concatenate([bd_plus, pd_plus])
        min_plus = np.min(all_plus)
        shifted_plus = all_plus - min_plus
        log_sum_plus = np.log(np.sum(np.exp(-beta * shifted_plus)))
        soft_min_plus = min_plus - log_sum_plus / beta
        grad[i] = -(soft_min_plus - soft_min) / eps
    
    return obj, grad


def _optimize_lbfgs(params_init, t_budget, beta_schedule=None):
    """
    Optimize using L-BFGS-B with soft-min objective.
    """
    if beta_schedule is None:
        beta_schedule = [10.0, 50.0, 200.0, 1000.0, 5000.0, 20000.0]
    
    start = time.time()
    params = params_init.copy()
    best_params = params.copy()
    best_score = _compute_true_score(_sym_params_to_points(params))
    
    for beta in beta_schedule:
        if time.time() - start > t_budget - 2:
            break
        
        def obj_and_grad(p):
            return _soft_min_obj_and_grad(p, beta=beta)
        
        try:
            result = minimize(
                lambda p: _soft_min_obj_and_grad(p, beta=beta)[0],
                params,
                jac=lambda p: _soft_min_obj_and_grad(p, beta=beta)[1],
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 500}
            )
            params = result.x
        except Exception:
            pass
        
        # Evaluate and track best
        current_points = _sym_params_to_points(params)
        current_score = _compute_true_score(current_points)
        if current_score > best_score:
            best_score = current_score
            best_params = params.copy()
    
    return best_params, best_score


def _gd_polish(points, t_budget, beta=50000.0):
    """
    Final gradient descent polish on the full 600-dimensional space.
    """
    start = time.time()
    pos = points.copy().astype(np.float64)
    best_pos = pos.copy()
    best_r = _score(pos)
    
    lr = 0.001
    patience = 0
    
    while time.time() - start < t_budget:
        # Compute gradient of -soft_min
        norms = np.linalg.norm(pos, axis=1)
        boundary_dists = 1.0 - norms
        
        tree = cKDTree(pos)
        dists, indices = tree.query(pos, k=2)
        pair_dists = 0.5 * dists[:, 1]
        
        # Soft min gradient
        all_dists = np.concatenate([boundary_dists, pair_dists])
        min_val = np.min(all_dists)
        shifted = all_dists - min_val
        weights = np.exp(-beta * shifted)
        weights /= np.sum(weights)
        
        # Gradient of -soft_min with respect to positions
        grad = np.zeros_like(pos)
        
        # Boundary contribution
        for i in range(N):
            w = weights[i]
            if norms[i] > 1e-10:
                grad[i] -= w * (pos[i] / norms[i])  # grad of (1 - norm) is -pos/norm
        
        # Pair contribution
        for i in range(N):
            j = indices[i, 1]
            w = weights[N + i]
            d = pos[i] - pos[j]
            dist = np.linalg.norm(d)
            if dist > 1e-10:
                grad[i] += w * 0.5 * (d / dist)
                grad[j] -= w * 0.5 * (d / dist)
        
        # Gradient descent step
        pos -= lr * grad
        
        # Project into unit circle
        rad = np.linalg.norm(pos, axis=1)
        mask = rad > 0.999
        if np.any(mask):
            pos[mask] = pos[mask] / rad[mask, None] * 0.999
        
        current_r = _score(pos)
        if current_r > best_r:
            best_r = current_r
            best_pos = pos.copy()
            patience = 0
        else:
            patience += 1
            if patience > 50:
                lr *= 0.9
                patience = 0
            if lr < 1e-6:
                break
    
    return best_pos, best_r


def construct():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # Generate initial points using different strategies
    init_points_list = [
        _hex_lattice_init(2.0 / (np.sqrt(N) + 1.0)),
        _radial_ring_init(),
    ]
    
    # Add a random initialization
    random_pts = rng.standard_normal((N, 2)) * 0.7
    rad = np.linalg.norm(random_pts, axis=1)
    mask = rad > 0.95
    if np.any(mask):
        random_pts[mask] = random_pts[mask] / rad[mask, None] * 0.95
    init_points_list.append(random_pts)
    
    best = init_points_list[0].copy()
    best_r = _score(best)
    
    # Phase 1: Symmetric optimization on each initialization
    for init_pts in init_points_list:
        if time.time() - t0 > DEADLINE - 20:
            break
        t_left = DEADLINE - (time.time() - t0)
        if t_left < 25:
            break
        t_alloc = min(t_left - 10, 30.0)
        
        # Convert to symmetric parameters
        sym_params = _points_to_sym_params(init_pts)
        
        # Optimize with L-BFGS-B
        sym_result, sym_score = _optimize_lbfgs(sym_params, t_alloc)
        
        # Convert back to points
        result_points = _sym_params_to_points(sym_result)
        
        if sym_score > best_r:
            best_r = sym_score
            best = result_points
    
    # Phase 2: Full-space gradient descent polish
    if time.time() - t0 < DEADLINE - 5:
        t_left = DEADLINE - (time.time() - t0)
        if t_left > 5:
            t_alloc = min(t_left - 2, 20.0)
            cand, r = _gd_polish(best, t_alloc)
            if r > best_r:
                best_r = r
                best = cand
    
    # Final safety check
    final_r = _score(best)
    if final_r <= 0.0:
        best = _hex_lattice_init(2.0 / (np.sqrt(N) + 1.0))

    return best
