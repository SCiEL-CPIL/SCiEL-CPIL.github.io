import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay, cKDTree

N = 300

def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice (with rotation) and select n_points best points."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0

    for attempt in range(14):
        base_spacing = 2.0 / (np.sqrt(n_points) + 1.0) * 0.95
        spacing = base_spacing * (0.95 + 0.015 * attempt)

        k = int(1.0 / spacing) + 5
        rows = []
        rot_angle = rng.uniform(0, 2 * np.pi)
        cos_a = np.cos(rot_angle)
        sin_a = np.sin(rot_angle)
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
            selected = selected + rng.normal(0, spacing * 0.04, selected.shape)
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

def _build_edges(pts, margin=0.15):
    """Build a sparse contact edge set using KDTree for speed."""
    n = len(pts)
    if n < 3:
        iu = np.triu_indices(n, k=1)
        return iu[0], iu[1]
    try:
        norms = np.linalg.norm(pts, axis=1)
        area = np.pi * 1.0
        typical_spacing = np.sqrt(area / n)
        search_radius = typical_spacing * 2.5
        
        tree = cKDTree(pts)
        pairs = tree.query_pairs(search_radius)
        
        edge_set = set()
        for i, j in pairs:
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            edge_set.add((a, b))
        
        edges = list(edge_set)
        if len(edges) < n:
            # Fallback: include nearest neighbours
            d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
            np.fill_diagonal(d, np.inf)
            k = 4
            for i in range(n):
                nn = np.argsort(d[i])[:k]
                for j in nn:
                    a, b = (i, int(j)) if i < int(j) else (int(j), i)
                    edge_set.add((a, b))
            edges = list(edge_set)
        
        if len(edges) < n:
            d = np.linalg.norm(pts[:, np.newaxis, :] - pts[np.newaxis, :, :], axis=2)
            np.fill_diagonal(d, np.inf)
            for i in range(n):
                nn = np.argsort(d[i])[:8]
                for j in nn:
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
    """
    Sparse soft-min objective. Constraints:
      - pairwise half-distances d_ij/2 for edges (i,j) in (ei,ej)
      - boundary distances 1 - |c_i|
    """
    pts = x.reshape(-1, 2)
    n = len(pts)

    norms = np.linalg.norm(pts, axis=1)
    boundary = 1.0 - norms  # (n,)

    diff = pts[ei] - pts[ej]
    d = np.linalg.norm(diff, axis=1)
    pair_half = d / 2.0  # (E,)

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

    # Boundary gradient
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    grad_boundary = -pts / safe_norms[:, np.newaxis] * grad_all[n_pairs:][:, np.newaxis]
    grad_pts += grad_boundary

    # Pairwise gradient (sparse)
    w = grad_all[:n_pairs]
    safe_d = np.where(d > 1e-10, d, 1.0)
    unit = diff / safe_d[:, np.newaxis]
    contrib = (0.5 * w)[:, np.newaxis] * unit
    np.add.at(grad_pts, ei, contrib)
    np.add.at(grad_pts, ej, -contrib)

    return obj, grad_pts.ravel()

def _force_relax(pts, n_steps=4000, deadline=None):
    """Fast force-directed relaxation using sparse Delaunay contacts + Hertzian overlap^4."""
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
        E = len(ei)
        forces = np.zeros_like(pos)

        diff = pos[ei] - pos[ej]
        d = np.linalg.norm(diff, axis=1)
        target = 2.0 * r_est
        mask = d < target
        safe_d = np.where(mask, d, 1.0)
        excess = np.where(mask, target - d, 0.0)
        # Hertzian overlap^4 -> force magnitude 4*(target-d)^3 / d
        force_mag = np.where(mask, 4.0 * excess ** 3 / safe_d, 0.0)
        unit = diff / safe_d[:, np.newaxis]
        contrib = force_mag[:, np.newaxis] * unit
        np.add.at(forces, ei, contrib)
        np.add.at(forces, ej, -contrib)

        # Boundary: push inward if too close to wall
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

        # Keep inside container
        new_norms = np.linalg.norm(pos, axis=1)
        out = new_norms >= 1.0
        if np.any(out):
            pos[out] *= 0.999 / new_norms[out, np.newaxis]

        # Periodically update best and r_est
        if step % 200 == 0 or step == n_steps - 1:
            r_now = _compute_radius(pos)
            if r_now > best_r:
                best_r = r_now
                best_pos = pos.copy()
            r_est = 0.5 * (r_est + r_now)

    return best_pos

def _newton_polish(pts, n_iter=100, deadline=None):
    """
    Damped Newton method to enforce exact touching conditions.
    Identifies critical constraints (active set) and solves for positions
    that maximize the minimum radius.
    """
    pos = pts.copy().astype(np.float64)
    n = len(pos)
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    
    # Identify critical constraints
    def _get_critical_constraints(p):
        norms = np.linalg.norm(p, axis=1)
        boundary = 1.0 - norms
        ei, ej = _build_edges(p)
        d = np.linalg.norm(p[ei] - p[ej], axis=1)
        pair_half = d / 2.0
        
        all_vals = np.concatenate([pair_half, boundary])
        min_val = all_vals.min()
        
        # Get indices of constraints close to minimum
        tol = 1e-4
        crit_pairs = []
        for k in range(len(pair_half)):
            if pair_half[k] <= min_val + tol:
                crit_pairs.append((int(ei[k]), int(ej[k])))
        
        crit_boundary = []
        for i in range(n):
            if boundary[i] <= min_val + tol:
                crit_boundary.append(i)
        
        return crit_pairs, crit_boundary, min_val
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        crit_pairs, crit_boundary, r_min = _get_critical_constraints(pos)
        
        if len(crit_pairs) == 0 and len(crit_boundary) == 0:
            break
        
        # Build linear system: for each critical pair (i,j), |p_i - p_j| = 2*r
        # For each critical boundary i, |p_i| = 1 - r
        # We solve for small displacements dp that increase r.
        
        # Simplified: use gradient descent on the soft-min with very low T
        # as a proxy for Newton step on the active set.
        T = 1e-5
        ei, ej = _build_edges(pos)
        obj, grad = _objective_and_grad(pos.ravel(), T, ei, ej)
        grad_pts = grad.reshape(-1, 2)
        
        # Damped step
        step_size = 0.01 / (1.0 + it * 0.1)
        new_pos = pos - step_size * grad_pts
        
        # Project inside container
        new_norms = np.linalg.norm(new_pos, axis=1)
        out = new_norms >= 1.0
        if np.any(out):
            new_pos[out] *= 0.999 / new_norms[out, np.newaxis]
        
        new_r = _compute_radius(new_pos)
        if new_r > best_r:
            best_r = new_r
            best_pos = new_pos.copy()
            pos = new_pos
        else:
            # Try smaller step
            new_pos = pos - 0.5 * step_size * grad_pts
            new_norms = np.linalg.norm(new_pos, axis=1)
            out = new_norms >= 1.0
            if np.any(out):
                new_pos[out] *= 0.999 / new_norms[out, np.newaxis]
            new_r = _compute_radius(new_pos)
            if new_r > best_r:
                best_r = new_r
                best_pos = new_pos.copy()
                pos = new_pos
            else:
                break
    
    return best_pos

def _tighten_min_constraint(pts, n_iter=50, step_size=0.001):
    """Directly push the most constrained points to increase the minimum constraint."""
    pts = pts.copy()
    for _ in range(n_iter):
        norms = np.linalg.norm(pts, axis=1)
        boundary = 1.0 - norms

        ei, ej = _build_edges(pts)
        d = np.linalg.norm(pts[ei] - pts[ej], axis=1)
        pair_half = d / 2.0

        min_pair_val = pair_half.min()
        min_boundary_val = boundary.min()

        if min_pair_val < min_boundary_val:
            k = int(np.argmin(pair_half))
            i, j = int(ei[k]), int(ej[k])
            vec = pts[i] - pts[j]
            dist = np.linalg.norm(vec)
            if dist < 1e-10:
                continue
            unit = vec / dist
            pts[i] += step_size * unit
            pts[j] -= step_size * unit
        else:
            i = int(np.argmin(boundary))
            norm_i = norms[i]
            if norm_i < 1e-10:
                continue
            unit = pts[i] / norm_i
            pts[i] -= step_size * unit

        new_norms = np.linalg.norm(pts, axis=1)
        mask = new_norms >= 1.0
        if np.any(mask):
            pts[mask] *= 0.999 / new_norms[mask, np.newaxis]

    return pts

def _local_refine(pts, n_iter=200, step_size=0.001, n_critical=20):
    """Local refinement on the most constrained points using the sparse objective."""
    pts = pts.copy()
    n = len(pts)

    for it in range(n_iter):
        norms = np.linalg.norm(pts, axis=1)
        boundary = 1.0 - norms
        ei, ej = _build_edges(pts)
        d = np.linalg.norm(pts[ei] - pts[ej], axis=1)
        pair_half = d / 2.0
        min_val = min(pair_half.min(), boundary.min())

        point_constraints = np.full(n, np.inf)
        point_constraints = np.minimum(point_constraints, boundary)
        for k in range(len(ei)):
            v = pair_half[k]
            point_constraints[ei[k]] = min(point_constraints[ei[k]], v)
            point_constraints[ej[k]] = min(point_constraints[ej[k]], v)

        order = np.argsort(point_constraints)
        crit_idx = order[:n_critical]

        if len(crit_idx) < 2:
            break

        T = 0.001
        for _ in range(5):
            obj, grad = _objective_and_grad(pts.ravel(), T, ei, ej)
            grad_pts = grad.reshape(-1, 2)
            for idx in crit_idx:
                pts[idx] -= step_size * grad_pts[idx]

            new_norms = np.linalg.norm(pts, axis=1)
            mask = new_norms >= 1.0
            if np.any(mask):
                pts[mask] *= 0.999 / new_norms[mask, np.newaxis]

    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    best_pts = None
    best_r = -1.0

    seeds = [42, 0, 1, 2, 3]
    for seed in seeds:
        if time.time() > deadline - 60.0:
            break
        pts = _hex_init(N, seed=seed)
        r = _compute_radius(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    if best_pts is None:
        best_pts = _hex_init(N, seed=42)
        best_r = _compute_radius(best_pts)

    # Fast force-directed relaxation first (cheap, many steps)
    if time.time() < deadline - 40.0:
        try:
            relaxed = _force_relax(best_pts, n_steps=4000, deadline=deadline - 35.0)
            norms = np.linalg.norm(relaxed, axis=1)
            if np.all(norms < 1.0) and np.all(np.isfinite(relaxed)):
                rel_r = _compute_radius(relaxed)
                if rel_r > best_r:
                    best_r = rel_r
                    best_pts = relaxed.copy()
        except Exception:
            pass

    # Sparse soft-min annealing
    temperatures = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]
    x0 = best_pts.ravel()
    for T in temperatures:
        if time.time() > deadline - 25.0:
            break
        try:
            ei, ej = _build_edges(best_pts)
            maxiter = 500 if T <= 0.001 else 300
            maxfun = 1500 if T <= 0.001 else 800
            result = minimize(
                lambda x: _objective_and_grad(x, T, ei, ej)[0],
                x0,
                jac=lambda x: _objective_and_grad(x, T, ei, ej)[1],
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-13, 'gtol': 1e-11,
                         'maxls': 100, 'maxfun': maxfun}
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

    # Newton polish to enforce exact touching conditions
    if time.time() < deadline - 20.0:
        try:
            polished = _newton_polish(best_pts, n_iter=100, deadline=deadline - 15.0)
            norms = np.linalg.norm(polished, axis=1)
            if np.all(norms < 1.0) and np.all(np.isfinite(polished)):
                pol_r = _compute_radius(polished)
                if pol_r > best_r:
                    best_r = pol_r
                    best_pts = polished.copy()
                    x0 = best_pts.ravel()
        except Exception:
            pass

    # Tighten pass
    if time.time() < deadline - 15.0:
        try:
            tightened = _tighten_min_constraint(best_pts, n_iter=150, step_size=0.0005)
            norms = np.linalg.norm(tightened, axis=1)
            if np.all(norms < 1.0) and np.all(np.isfinite(tightened)):
                tight_r = _compute_radius(tightened)
                if tight_r > best_r:
                    best_r = tight_r
                    best_pts = tightened.copy()
        except Exception:
            pass

    # Local refinement on critical points
    if time.time() < deadline - 12.0:
        try:
            refined = _local_refine(best_pts, n_iter=100, step_size=0.0008, n_critical=30)
            norms = np.linalg.norm(refined, axis=1)
            if np.all(norms < 1.0) and np.all(np.isfinite(refined)):
                ref_r = _compute_radius(refined)
                if ref_r > best_r:
                    best_r = ref_r
                    best_pts = refined.copy()
        except Exception:
            pass

    # Final high-precision polish
    if time.time() < deadline - 8.0:
        try:
            T_final = 0.00001
            ei, ej = _build_edges(best_pts)
            result = minimize(
                lambda x: _objective_and_grad(x, T_final, ei, ej)[0],
                x0,
                jac=lambda x: _objective_and_grad(x, T_final, ei, ej)[1],
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-13,
                         'maxls': 200, 'maxfun': 8000}
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

    # Final tighten
    if time.time() < deadline - 3.0:
        try:
            tightened = _tighten_min_constraint(best_pts, n_iter=100, step_size=0.0002)
            norms = np.linalg.norm(tightened, axis=1)
            if np.all(norms < 1.0) and np.all(np.isfinite(tightened)):
                tight_r = _compute_radius(tightened)
                if tight_r > best_r:
                    best_r = tight_r
                    best_pts = tightened.copy()
        except Exception:
            pass

    # Second local refinement
    if time.time() < deadline - 1.0:
        try:
            refined = _local_refine(best_pts, n_iter=50, step_size=0.0005, n_critical=20)
            norms = np.linalg.norm(refined, axis=1)
            if np.all(norms < 1.0) and np.all(np.isfinite(refined)):
                ref_r = _compute_radius(refined)
                if ref_r > best_r:
                    best_r = ref_r
                    best_pts = refined.copy()
        except Exception:
            pass

    # Ensure validity
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms < 1.0
    if not np.all(mask) or not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N, seed=0)

    return best_pts
