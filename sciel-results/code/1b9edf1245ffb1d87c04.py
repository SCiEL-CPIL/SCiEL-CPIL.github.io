"""Combined program: hex/ring init + explicit-gradient soft-min annealing + high-precision polish."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 300

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return max(0.0, 1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    iu = np.triu_indices(n, k=1)
    diff = points[iu[0]] - points[iu[1]]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    min_pair_dist = np.min(dists) / 2.0
    boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = np.min(boundary_dists)
    return min(min_pair_dist, min_boundary)

def _hex_lattice_init(n):
    """Initialize points in a hexagonal lattice clipped to the unit circle."""
    best_pts = None
    best_score = -1.0
    for scale in [0.9, 0.95, 1.0, 1.05, 1.1]:
        spacing = 2.0 / (np.sqrt(n) * scale)
        nx = int(2.0 / spacing) + 3
        ny = int(2.0 / (spacing * np.sqrt(3.0) / 2.0)) + 3
        x_coords = (np.arange(-nx // 2, nx // 2 + 1) + 0.5) * spacing
        y_coords = (np.arange(-ny // 2, ny // 2 + 1)) * spacing * np.sqrt(3.0) / 2.0
        pts_list = []
        for i, y in enumerate(y_coords):
            offset = 0.0 if i % 2 == 0 else spacing / 2.0
            for x in x_coords:
                px = x + offset
                py = y
                if np.hypot(px, py) <= 1.0 - spacing / 2.0:
                    pts_list.append([px, py])
        if len(pts_list) < n:
            continue
        pts = np.array(pts_list)
        norms = np.linalg.norm(pts, axis=1)
        idx = np.argsort(norms)
        pts = pts[idx[:n]]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists) / 2.0
        boundary_dists = 1.0 - np.linalg.norm(pts, axis=1)
        min_boundary = np.min(boundary_dists)
        score = min(min_pair_dist, min_boundary)
        if score > best_score:
            best_score = score
            best_pts = pts
    if best_pts is None:
        best_pts = np.random.rand(n, 2) * 2 - 1
        norms = np.linalg.norm(best_pts, axis=1)
        best_pts = best_pts * (0.9 / norms)[:, None]
    best_pts = best_pts + np.random.normal(0, 1e-4, best_pts.shape)
    return best_pts

def _ring_decomposition_init(n, num_rings=None, ring_spread=0.92):
    """Initialize points in concentric rings."""
    if num_rings is None:
        num_rings = int(np.sqrt(n) / 2.0) + 3
    points = [[0.0, 0.0]]
    remaining = n - 1
    r_est = 1.0 / np.sqrt(n)
    ring_radii = [ring_spread * (i + 1) / (num_rings + 1) for i in range(num_rings)]
    circumferences = [2.0 * np.pi * rr for rr in ring_radii]
    total_circ = sum(circumferences)
    ring_counts = []
    for i, circ in enumerate(circumferences):
        count = int(remaining * circ / total_circ)
        ring_counts.append(max(1, count))
    diff = remaining - sum(ring_counts)
    if diff > 0:
        for _ in range(diff):
            ring_counts[-1] += 1
    elif diff < 0:
        for i in range(len(ring_counts) - 1, -1, -1):
            if ring_counts[i] > 1 and -diff > 0:
                ring_counts[i] -= 1
                diff += 1
    for i, (count, rr) in enumerate(zip(ring_counts, ring_radii)):
        if count == 0:
            continue
        d_theta = 2.0 * np.pi / count
        offset = np.pi / count * (i % 2)
        for k in range(count):
            theta = k * d_theta + offset
            points.append([rr * np.cos(theta), rr * np.sin(theta)])
    points = np.array(points[:n])
    points += np.random.normal(0, 1e-4, points.shape)
    return points

def _soft_min_and_grad(values, beta):
    """Compute soft-min and its gradient weights using log-sum-exp trick."""
    z = -beta * values
    max_z = np.max(z)
    exp_vals = np.exp(z - max_z)
    sum_exp = np.sum(exp_vals)
    log_sum_exp = max_z + np.log(sum_exp)
    soft_min_val = -log_sum_exp / beta
    weights = exp_vals / sum_exp
    return soft_min_val, weights

def _objective_and_grad(x, beta):
    """Compute negative soft-min objective and its gradient efficiently."""
    p = x.reshape(-1, 2)
    n = len(p)
    iu = np.triu_indices(n, k=1)
    diff = p[iu[0]] - p[iu[1]]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    pair_vals = dists / 2.0
    norms = np.linalg.norm(p, axis=1)
    boundary_dists = 1.0 - norms
    all_vals = np.concatenate([pair_vals, boundary_dists])
    r_soft, weights = _soft_min_and_grad(all_vals, beta)
    objective = -r_soft
    grad = np.zeros_like(p)
    w_pairs = weights[:len(pair_vals)]
    i_idx, j_idx = iu
    safe_pair_dists = np.where(pair_vals > 1e-12, pair_vals, 1.0)
    coeff = w_pairs * 0.25 / safe_pair_dists
    vecs = p[i_idx] - p[j_idx]
    np.add.at(grad, i_idx, coeff[:, np.newaxis] * vecs)
    np.add.at(grad, j_idx, -coeff[:, np.newaxis] * vecs)
    w_bounds = weights[len(pair_vals):]
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    bound_grad_vecs = -p / safe_norms[:, np.newaxis]
    grad += w_bounds[:, np.newaxis] * bound_grad_vecs
    grad_obj = -grad
    return objective, grad_obj

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin for 150s limit

    # Try both initializations and pick the better one
    hex_pts = _hex_lattice_init(N)
    hex_score = _compute_score(hex_pts)
    
    ring_pts = _ring_decomposition_init(N)
    ring_score = _compute_score(ring_pts)
    
    if ring_score > hex_score:
        best_x = ring_pts.copy()
        best_score = ring_score
    else:
        best_x = hex_pts.copy()
        best_score = hex_score

    bounds = [(-1.0, 1.0)] * (2 * N)

    # Annealing optimization with explicit gradients
    beta_schedule = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]
    
    current_x = best_x.copy()
    
    for beta in beta_schedule:
        if time.time() > deadline - 20:
            break
        
        def neg_soft_min(x):
            obj, _ = _objective_and_grad(x, beta)
            return obj
        
        def neg_soft_min_grad(x):
            _, g = _objective_and_grad(x, beta)
            return g
        
        try:
            res = minimize(neg_soft_min, current_x, jac=neg_soft_min_grad,
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10,
                                    'maxls': 50})
            
            if res.fun < neg_soft_min(current_x):
                current_x = res.x.copy()
                current_raw = _compute_score(current_x)
                if current_raw > best_score:
                    best_score = current_raw
                    best_x = current_x.copy()
        except Exception:
            continue

    # Final high-precision polish
    if time.time() < deadline - 10:
        # Nelder-Mead for kink handling
        try:
            def raw_score(x):
                return -_compute_score(x.reshape(-1, 2))
            
            res = minimize(raw_score, best_x, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12,
                                    'adaptive': True})
            if res.fun < -best_score:
                best_x = res.x.copy()
                best_score = -res.fun
        except Exception:
            pass

    # One more L-BFGS-B pass with tight tolerance
    if time.time() < deadline - 5:
        try:
            def raw_score(x):
                return -_compute_score(x.reshape(-1, 2))
            
            # Use finite-difference gradient for the raw score
            res = minimize(raw_score, best_x, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < -best_score:
                best_x = res.x.copy()
                best_score = -res.fun
        except Exception:
            pass

    # Final validation
    best_x = best_x.reshape(-1, 2)
    norms = np.linalg.norm(best_x, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_x[mask] *= (1.0 / norms[mask] - 0.001)[:, np.newaxis]
    
    if not np.all(np.isfinite(best_x)):
        return hex_pts

    return best_x

# EVOLVE-BLOCK-END
