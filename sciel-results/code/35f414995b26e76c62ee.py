import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 300

def _hex_init(n_points, seed=0):
    """Generate hexagonal lattice points within unit circle."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0

    for attempt in range(20):
        base_spacing = 2.0 / (np.sqrt(n_points) + 1.0) * 0.95
        spacing = base_spacing * (0.93 + 0.012 * attempt)

        k = int(1.0 / spacing) + 6
        rot_angle = rng.uniform(0, 2 * np.pi)
        cos_a = np.cos(rot_angle)
        sin_a = np.sin(rot_angle)

        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                x_rot = x * cos_a - y * sin_a
                y_rot = x * sin_a + y * cos_a
                r = np.hypot(x_rot, y_rot)
                if r <= 1.0 - spacing / 2.0:
                    rows.append((x_rot, y_rot, r))

        if len(rows) < n_points:
            continue

        pts = np.array([(r[0], r[1]) for r in rows])
        radii = np.array([r[2] for r in rows])
        order = np.argsort(radii, kind="stable")
        selected = pts[order[:n_points]]

        if attempt > 0:
            selected = selected + rng.normal(0, spacing * 0.03, selected.shape)
            norms = np.linalg.norm(selected, axis=1)
            mask = norms < 1.0
            selected = selected[mask]
            if len(selected) < n_points:
                continue
            selected = selected[:n_points]

        d = np.linalg.norm(selected[:, np.newaxis, :] - selected[np.newaxis, :, :], axis=2)
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
    d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() / 2.0
    return min(min_pair, min_bound)

def _voronoi_area_gradient(pts):
    """
    Compute gradient of sum of log(Voronoi cell area) for each point.
    This promotes uniform spacing (maximizes local density uniformity).
    Uses a smooth approximation based on nearest neighbors.
    """
    n = len(pts)
    grad = np.zeros_like(pts)
    
    # For each point, find k nearest neighbors
    tree = cKDTree(pts)
    k = min(8, n - 1)
    dists, indices = tree.query(pts, k=k + 1)  # +1 to include self
    
    for i in range(n):
        # Get neighbors excluding self
        nbr_idx = indices[i, 1:]
        nbr_dists = dists[i, 1:]
        
        # Estimate local area as product of distances to neighbors (smooth)
        # Gradient of log(area) w.r.t. point i: sum over neighbors of -unit_vec / dist
        for j_idx in range(len(nbr_idx)):
            j = int(nbr_idx[j_idx])
            d = nbr_dists[j_idx]
            if d < 1e-10:
                continue
            vec = pts[i] - pts[j]
            unit = vec / d
            # Gradient contribution: -unit / d (pushes point away from neighbor)
            grad[i] += -unit / d
            # By Newton's third law, also push neighbor
            grad[j] += unit / d
    
    return grad

def _voronoi_relax(pts, n_steps=3000, deadline=None, lr_init=0.005, cooldown=0.9995):
    """
    Relax points using gradient ascent on sum of log(Voronoi cell areas).
    This promotes uniform local density, avoiding contact-graph traps.
    """
    pos = pts.copy().astype(np.float64)
    n = len(pos)
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    
    lr = lr_init
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute Voronoi area gradient
        grad = _voronoi_area_gradient(pos)
        
        # Boundary constraint: push inward if too close to wall
        norms = np.linalg.norm(pos, axis=1)
        wall_margin = 0.05
        too_close = norms > (1.0 - wall_margin)
        if np.any(too_close):
            idx = np.where(too_close)[0]
            excess_b = norms[idx] - (1.0 - wall_margin)
            f_mag = 100.0 * excess_b ** 2
            safe_n = np.where(norms[idx] > 1e-10, norms[idx], 1.0)
            direction = -pos[idx] / safe_n[:, np.newaxis]
            grad[idx] += (f_mag / safe_n)[:, np.newaxis] * direction
        
        # Update positions
        pos += lr * grad
        lr *= cooldown
        
        # Keep inside container
        new_norms = np.linalg.norm(pos, axis=1)
        out = new_norms >= 1.0
        if np.any(out):
            pos[out] *= 0.999 / new_norms[out, np.newaxis]
        
        # Periodically check improvement
        if step % 100 == 0 or step == n_steps - 1:
            r_now = _compute_radius(pos)
            if r_now > best_r:
                best_r = r_now
                best_pos = pos.copy()
    
    return best_pos

def _soft_min_logsumexp(values, T):
    """Stable soft minimum."""
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

def _build_edges(pts):
    """Build sparse contact edge set using KDTree."""
    n = len(pts)
    if n < 3:
        iu = np.triu_indices(n, k=1)
        return iu[0], iu[1]
    
    try:
        norms = np.linalg.norm(pts, axis=1)
        typical_spacing = np.sqrt(np.pi / n)
        search_radius = typical_spacing * 2.5
        
        tree = cKDTree(pts)
        pairs = tree.query_pairs(search_radius)
        
        edge_set = set()
        for i, j in pairs:
            if i > j:
                i, j = j, i
            edge_set.add((i, j))
        
        edges = list(edge_set)
        
        if len(edges) < n:
            for i in range(n):
                nn = tree.query(pts[i], k=6, workers=-1)
                for j in nn[1][:5]:
                    if j != i:
                        a, b = (i, int(j)) if i < int(j) else (int(j), i)
                        edge_set.add((a, b))
            edges = list(edge_set)
        
        ei = np.array([e[0] for e in edges], dtype=np.int64)
        ej = np.array([e[1] for e in edges], dtype=np.int64)
        return ei, ej
    except Exception:
        iu = np.triu_indices(n, k=1)
        return iu[0], iu[1]

def _objective_and_grad(x, T, ei, ej):
    """Sparse soft-min objective with gradient."""
    pts = x.reshape(-1, 2)
    n = len(pts)

    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms

    diff = pts[ei] - pts[ej]
    d = np.linalg.norm(diff, axis=1)
    pair_half = d / 2.0

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

    n_pairs = len(pair_half)
    grad_pts = np.zeros_like(pts)

    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    grad_boundary = -pts / safe_norms[:, np.newaxis] * grad_all[n_pairs:][:, np.newaxis]
    grad_pts += grad_boundary

    w = grad_all[:n_pairs]
    safe_d = np.where(d > 1e-10, d, 1.0)
    unit = diff / safe_d[:, np.newaxis]
    contrib = (0.5 * w)[:, np.newaxis] * unit
    np.add.at(grad_pts, ei, contrib)
    np.add.at(grad_pts, ej, -contrib)

    return obj, grad_pts.ravel()

def _force_relax(pts, n_steps=3000, deadline=None):
    """Fast force-directed relaxation using KDTree contacts."""
    pos = pts.copy().astype(np.float64)
    n = len(pos)
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)

    lr = 0.02
    cooldown = 0.998
    r_est = best_r

    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break

        ei, ej = _build_edges(pos)
        forces = np.zeros_like(pos)

        diff = pos[ei] - pos[ej]
        d = np.linalg.norm(diff, axis=1)
        target = 2.0 * r_est
        mask = d < target
        safe_d = np.where(mask, d, 1.0)
        excess = np.where(mask, target - d, 0.0)
        force_mag = np.where(mask, 4.0 * excess ** 3 / safe_d, 0.0)
        unit = diff / safe_d[:, np.newaxis]
        contrib = force_mag[:, np.newaxis] * unit
        np.add.at(forces, ei, contrib)
        np.add.at(forces, ej, -contrib)

        norms = np.linalg.norm(pos, axis=1)
        wall_margin = 0.03
        too_close = norms > (1.0 - wall_margin)
        if np.any(too_close):
            idx = np.where(too_close)[0]
            excess_b = norms[idx] - (1.0 - wall_margin)
            f_mag = 50.0 * excess_b ** 3
            safe_n = np.where(norms[idx] > 1e-10, norms[idx], 1.0)
            direction = -pos[idx] / safe_n[:, np.newaxis]
            forces[idx] += (f_mag / safe_n)[:, np.newaxis] * direction

        pos += lr * forces
        lr *= cooldown

        new_norms = np.linalg.norm(pos, axis=1)
        out = new_norms >= 1.0
        if np.any(out):
            pos[out] *= 0.999 / new_norms[out, np.newaxis]

        if step % 200 == 0 or step == n_steps - 1:
            r_now = _compute_radius(pos)
            if r_now > best_r:
                best_r = r_now
                best_pos = pos.copy()
            r_est = 0.5 * (r_est + r_now)

    return best_pos

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    best_pts = None
    best_r = -1.0

    # Phase 1: Generate multiple hexagonal lattice initializations
    seeds = [42, 0, 1, 2, 3, 7, 13, 21]
    for seed in seeds:
        if time.time() > deadline - 80.0:
            break
        pts = _hex_init(N, seed=seed)
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    if best_pts is None:
        best_pts = _hex_init(N, seed=42)
        best_r = _compute_radius(best_pts)

    # Phase 2: Voronoi-area-based relaxation (NEW structural approach)
    if time.time() < deadline - 50.0:
        try:
            voronoi_relaxed = _voronoi_relax(best_pts, n_steps=4000, deadline=deadline - 45.0)
            norms = np.linalg.norm(voronoi_relaxed, axis=1)
            if np.all(norms < 1.0) and np.all(np.isfinite(voronoi_relaxed)):
                vor_r = _compute_radius(voronoi_relaxed)
                if vor_r > best_r:
                    best_r = vor_r
                    best_pts = voronoi_relaxed.copy()
        except Exception:
            pass

    # Phase 3: Force-directed relaxation for contact-based refinement
    if time.time() < deadline - 35.0:
        try:
            relaxed = _force_relax(best_pts, n_steps=3000, deadline=deadline - 30.0)
            norms = np.linalg.norm(relaxed, axis=1)
            if np.all(norms < 1.0) and np.all(np.isfinite(relaxed)):
                rel_r = _compute_radius(relaxed)
                if rel_r > best_r:
                    best_r = rel_r
                    best_pts = relaxed.copy()
        except Exception:
            pass

    # Phase 4: Sparse soft-min annealing
    temperatures = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005]
    x0 = best_pts.ravel()
    for T in temperatures:
        if time.time() > deadline - 20.0:
            break
        try:
            ei, ej = _build_edges(best_pts)
            maxiter = 300 if T <= 0.001 else 200
            result = minimize(
                lambda x: _objective_and_grad(x, T, ei, ej)[0],
                x0,
                jac=lambda x: _objective_and_grad(x, T, ei, ej)[1],
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-13, 'gtol': 1e-11, 'maxls': 100}
            )
            if time.time() > deadline:
                break
            new_pts = result.x.reshape(-1, 2)
            norms = np.linalg.norm(new_pts, axis=1)
            if np.any(norms >= 1.0) or not np.all(np.isfinite(new_pts)):
                continue
            new_r = _compute_radius(new_pts)
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()
                x0 = new_pts.ravel()
        except Exception:
            continue

    # Phase 5: Another Voronoi relaxation pass on refined configuration
    if time.time() < deadline - 15.0:
        try:
            voronoi_relaxed2 = _voronoi_relax(best_pts, n_steps=2000, deadline=deadline - 12.0)
            norms = np.linalg.norm(voronoi_relaxed2, axis=1)
            if np.all(norms < 1.0) and np.all(np.isfinite(voronoi_relaxed2)):
                vor_r2 = _compute_radius(voronoi_relaxed2)
                if vor_r2 > best_r:
                    best_r = vor_r2
                    best_pts = voronoi_relaxed2.copy()
        except Exception:
            pass

    # Phase 6: Final high-precision polish
    if time.time() < deadline - 5.0:
        try:
            T_final = 0.00001
            ei, ej = _build_edges(best_pts)
            result = minimize(
                lambda x: _objective_and_grad(x, T_final, ei, ej)[0],
                x0,
                jac=lambda x: _objective_and_grad(x, T_final, ei, ej)[1],
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-13, 'maxls': 200}
            )
            new_pts = result.x.reshape(-1, 2)
            norms = np.linalg.norm(new_pts, axis=1)
            if np.all(norms < 1.0) and np.all(np.isfinite(new_pts)):
                new_r = _compute_radius(new_pts)
                if new_r > best_r:
                    best_r = new_r
                    best_pts = new_pts.copy()
        except Exception:
            pass

    # Ensure validity
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms < 1.0
    if not np.all(mask) or not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N, seed=0)

    return best_pts
