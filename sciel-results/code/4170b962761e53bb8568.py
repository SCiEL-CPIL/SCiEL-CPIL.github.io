"""Packing 300 circles in a unit circle: fast soft-min objective + multi-stage refinement pipeline."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# Precompute upper-triangle pair indices once (shared across all objective calls)
_IU = np.triu_indices(N, k=1)
_I_IDX = _IU[0]
_J_IDX = _IU[1]
_N_PAIRS = len(_I_IDX)

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the minimum radius for a given set of points (vectorized)."""
    n = len(points)
    if n < 2:
        return min(1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0

    vecs = points[_I_IDX] - points[_J_IDX]
    pair_dists = np.sqrt(np.sum(vecs**2, axis=1))
    min_pair_dist = np.min(pair_dists) / 2.0

    norms = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - norms)

    return min(min_pair_dist, min_boundary)

def _hex_lattice_init(n):
    """Initialize points in a hexagonal lattice clipped to the unit circle."""
    best_pts = None
    best_score = -1.0

    for scale in [0.85, 0.9, 0.95, 1.0, 1.05, 1.1]:
        spacing = 2.0 / (np.sqrt(n) * scale)
        nx = int(2.0 / spacing) + 4
        ny = int(2.0 / (spacing * np.sqrt(3.0) / 2.0)) + 4

        x_coords = (np.arange(-nx//2, nx//2 + 1) + 0.5) * spacing
        y_coords = (np.arange(-ny//2, ny//2 + 1)) * spacing * np.sqrt(3.0) / 2.0

        pts_list = []
        for i, y in enumerate(y_coords):
            offset = 0.0 if i % 2 == 0 else spacing / 2.0
            for x in x_coords:
                px = x + offset
                py = y
                if np.hypot(px, py) <= 1.0 - spacing / 2.0 + 1e-9:
                    pts_list.append([px, py])

        if len(pts_list) < n:
            continue

        pts = np.array(pts_list)
        norms = np.linalg.norm(pts, axis=1)
        idx = np.argsort(norms)
        pts = pts[idx[:n]]

        score = _compute_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()

    if best_pts is None:
        best_pts = np.random.rand(n, 2) * 2 - 1
        norms = np.linalg.norm(best_pts, axis=1)
        best_pts = best_pts * (0.9 / norms)[:, None]

    best_pts = best_pts + np.random.normal(0, 1e-4, best_pts.shape)
    return best_pts

def _ring_decomposition_init(n, num_rings=None, ring_spread=0.92):
    """Initialize points in concentric rings with optimized radial distances."""
    if num_rings is None:
        num_rings = int(np.sqrt(n) / 2.0) + 3

    points = [[0.0, 0.0]]
    remaining = n - 1

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
    """Soft-min and its gradient weights using the log-sum-exp trick."""
    values = np.asarray(values)
    z = -beta * values
    max_z = np.max(z)
    exp_vals = np.exp(z - max_z)
    sum_exp = np.sum(exp_vals)
    log_sum_exp = max_z + np.log(sum_exp)
    soft_min_val = -log_sum_exp / beta
    weights = exp_vals / sum_exp
    return soft_min_val, weights

def _objective_and_grad(x, beta):
    """Negative soft-min objective over all pair and boundary constraints, plus gradient.

    Vectorized: uses precomputed pair indices and direct vector differences.
    """
    p = x.reshape(-1, 2)

    vecs = p[_I_IDX] - p[_J_IDX]
    pair_dists = np.sqrt(np.sum(vecs**2, axis=1)) / 2.0

    norms = np.linalg.norm(p, axis=1)
    bound_dists = 1.0 - norms

    # Unified log-sum-exp over pairs and boundary
    z_pairs = -beta * pair_dists
    z_bounds = -beta * bound_dists

    max_z = max(np.max(z_pairs), np.max(z_bounds))
    exp_pairs = np.exp(z_pairs - max_z)
    exp_bounds = np.exp(z_bounds - max_z)

    sum_exp = np.sum(exp_pairs) + np.sum(exp_bounds)
    log_sum_exp = max_z + np.log(sum_exp)

    val = log_sum_exp / beta  # = -soft_min

    w_pairs = exp_pairs / sum_exp
    w_bounds = exp_bounds / sum_exp

    grad = np.zeros_like(p)

    # Pair contributions: d(pair_dist)/d(p_i) = (p_i - p_j)/|p_i - p_j|
    safe_d = np.where(pair_dists > 1e-12, pair_dists, 1.0)
    coeff = w_pairs * 0.25 / safe_d
    np.add.at(grad, _I_IDX, coeff[:, np.newaxis] * vecs)
    np.add.at(grad, _J_IDX, -coeff[:, np.newaxis] * vecs)

    # Boundary contributions: d(bound_dist)/d(p_i) = -p_i/|p_i|
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    bound_grad_vecs = -p / safe_norms[:, np.newaxis]
    grad += w_bounds[:, np.newaxis] * bound_grad_vecs

    return val, -grad

def _spring_mechanics_objective(x, target_r, k_spring=100.0, beta=1000.0):
    """Rigid-body mechanics objective: spring violations + soft-min clearance."""
    p = x.reshape(-1, 2)

    vecs = p[_I_IDX] - p[_J_IDX]
    pair_dists = np.sqrt(np.sum(vecs**2, axis=1))

    spring_violations = np.maximum(2.0 * target_r - pair_dists, 0.0)
    spring_energy = 0.5 * k_spring * np.sum(spring_violations**2)

    norms = np.linalg.norm(p, axis=1)
    boundary_dists = 1.0 - norms
    r_soft, _ = _soft_min_and_grad(boundary_dists, beta)

    return spring_energy - r_soft

def _spring_mechanics_grad(x, target_r, k_spring=100.0, beta=1000.0):
    """Gradient of the spring mechanics objective."""
    p = x.reshape(-1, 2)

    vecs = p[_I_IDX] - p[_J_IDX]
    pair_dists = np.sqrt(np.sum(vecs**2, axis=1))

    grad = np.zeros_like(p)

    compressed = pair_dists < 2.0 * target_r
    if np.any(compressed):
        comp_i = _I_IDX[compressed]
        comp_j = _J_IDX[compressed]
        comp_d = pair_dists[compressed]
        comp_vecs = vecs[compressed]
        safe_d = np.where(comp_d > 1e-12, comp_d, 1e-12)
        mag = k_spring * (2.0 * target_r - comp_d)
        np.add.at(grad, comp_i, mag[:, np.newaxis] * comp_vecs / safe_d[:, np.newaxis])
        np.add.at(grad, comp_j, -mag[:, np.newaxis] * comp_vecs / safe_d[:, np.newaxis])

    norms = np.linalg.norm(p, axis=1)
    boundary_dists = 1.0 - norms
    _, weights = _soft_min_and_grad(boundary_dists, beta)
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    bound_grad_vecs = -p / safe_norms[:, np.newaxis]
    grad += weights[:, np.newaxis] * bound_grad_vecs

    return grad

def _local_optimize_softmin(points, time_limit=30.0, beta_schedule=None, max_iter=500):
    """Local optimization via L-BFGS-B with the fast soft-min objective."""
    start_time = time.time()
    if beta_schedule is None:
        beta_schedule = [10.0, 50.0, 200.0, 1000.0, 5000.0]

    current_points = points.copy()
    current_score = _compute_score(current_points)

    for beta in beta_schedule:
        if time.time() - start_time > time_limit:
            break
        try:
            if beta >= 10000:
                opts = {'maxiter': max_iter, 'gtol': 1e-10, 'ftol': 1e-14, 'maxls': 100, 'disp': False}
            elif beta >= 1000:
                opts = {'maxiter': max_iter, 'gtol': 1e-9, 'ftol': 1e-13, 'maxls': 80, 'disp': False}
            else:
                opts = {'maxiter': max_iter, 'gtol': 1e-8, 'ftol': 1e-12, 'maxls': 50, 'disp': False}

            res = minimize(
                lambda x: _objective_and_grad(x, beta)[0],
                current_points.flatten(),
                jac=lambda x: _objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options=opts
            )

            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            if candidate_score > current_score:
                current_points = candidate
                current_score = candidate_score
        except Exception:
            continue

    return current_points, current_score

def _local_optimize_spring(points, time_limit=30.0, target_r=None, k_spring=50.0,
                           beta_schedule=None, max_iter=200):
    """Local optimization with the spring mechanics objective."""
    start_time = time.time()
    if target_r is None:
        target_r = _compute_score(points)
    if beta_schedule is None:
        beta_schedule = [100.0, 500.0, 2000.0, 10000.0]

    current_points = points.copy()
    current_score = _compute_score(current_points)

    for beta in beta_schedule:
        if time.time() - start_time > time_limit:
            break
        try:
            opts = {'maxiter': max_iter, 'gtol': 1e-10, 'ftol': 1e-14, 'maxls': 100, 'disp': False}
            res = minimize(
                lambda x: _spring_mechanics_objective(x, target_r, k_spring, beta),
                current_points.flatten(),
                jac=lambda x: _spring_mechanics_grad(x, target_r, k_spring, beta),
                method='L-BFGS-B',
                options=opts
            )
            candidate = res.x.reshape(-1, 2)
            candidate_score = _compute_score(candidate)
            if candidate_score > current_score:
                current_points = candidate
                current_score = candidate_score
                target_r = current_score
        except Exception:
            continue

    return current_points, current_score

def _force_directed_refine(points, iterations=300, time_limit=10.0):
    """Force-directed local optimization: push apart overlaps, pull toward boundary."""
    start_time = time.time()
    points = points.copy()
    n = len(points)

    for it in range(iterations):
        if time.time() - start_time > time_limit:
            break

        vecs = points[_I_IDX] - points[_J_IDX]
        pair_dists = np.sqrt(np.sum(vecs**2, axis=1))
        min_pair = np.min(pair_dists) / 2.0
        norms = np.linalg.norm(points, axis=1)
        min_boundary = np.min(1.0 - norms)
        r = min(min_pair, min_boundary)

        forces = np.zeros_like(points)

        close_mask = pair_dists < 2.0 * r * 1.5
        if np.any(close_mask):
            close_i = _I_IDX[close_mask]
            close_j = _J_IDX[close_mask]
            close_d = pair_dists[close_mask]
            close_vecs = vecs[close_mask]
            safe_d = np.where(close_d > 1e-10, close_d, 1e-10)
            mag = (2.0 * r - close_d) / safe_d
            forces[close_i] += mag[:, np.newaxis] * close_vecs
            forces[close_j] -= mag[:, np.newaxis] * close_vecs

        if min_boundary <= min_pair:
            for i in range(n):
                norm_i = norms[i]
                if norm_i > 0.01:
                    radial_dir = points[i] / norm_i
                    if 1.0 - norm_i < r * 1.1:
                        forces[i] -= radial_dir * 0.01
                    else:
                        forces[i] += radial_dir * 0.005

        step = 0.1 * r / (1.0 + it * 0.01)
        new_points = points + step * forces

        new_norms = np.linalg.norm(new_points, axis=1)
        for i in range(n):
            if new_norms[i] > 1.0 - 1e-10:
                new_points[i] = new_points[i] / new_norms[i] * (1.0 - 1e-10)

        new_score = _compute_score(new_points)
        if new_score >= r - 1e-8:
            points = new_points

    return points, _compute_score(points)

def _global_rotation_search(points, n_angles=36, time_limit=5.0):
    """Rotate the whole configuration to find better alignment."""
    start_time = time.time()
    best_points = points.copy()
    best_score = _compute_score(points)

    for angle in np.linspace(0, 2*np.pi, n_angles, endpoint=False):
        if time.time() - start_time > time_limit:
            break
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        rotated = points @ rot.T
        score = _compute_score(rotated)
        if score > best_score:
            best_score = score
            best_points = rotated

    return best_points, best_score

def _radial_displacement_refine(points, time_limit=10.0):
    """Refine radial positions of boundary circles."""
    start_time = time.time()
    n = len(points)

    norms = np.linalg.norm(points, axis=1)
    boundary_mask = norms > 0.7
    if not np.any(boundary_mask):
        return points, _compute_score(points)

    boundary_indices = np.where(boundary_mask)[0]

    best_points = points.copy()
    best_score = _compute_score(points)

    for delta_r in np.linspace(-0.02, 0.02, 21):
        if time.time() - start_time > time_limit:
            break
        candidate = points.copy()
        for idx in boundary_indices:
            norm_i = norms[idx]
            if norm_i > 1e-6:
                radial_dir = candidate[idx] / norm_i
                candidate[idx] += delta_r * radial_dir

        new_norms = np.linalg.norm(candidate, axis=1)
        for i in range(n):
            if new_norms[i] > 1.0 - 1e-10:
                candidate[i] = candidate[i] / new_norms[i] * (1.0 - 1e-10)

        score = _compute_score(candidate)
        if score > best_score:
            best_score = score
            best_points = candidate

    return best_points, best_score

def _clamp(points):
    """Clamp any point outside the unit circle back inside."""
    norms = np.linalg.norm(points, axis=1)
    for i in range(len(points)):
        if norms[i] > 1.0:
            points[i] = points[i] / norms[i] * (1.0 - 1e-12)
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 147.0  # safety margin before the 150s kill

    # Multiple initializations
    initializations = [
        _hex_lattice_init(N),
        _ring_decomposition_init(N, 8, 0.92),
        _ring_decomposition_init(N, 10, 0.90),
    ]

    best_points = None
    best_score = -np.inf
    for points in initializations:
        if time.time() > deadline - 60:
            break
        score = _compute_score(points)
        if score > best_score:
            best_points = points.copy()
            best_score = score

    if best_points is None:
        best_points = _hex_lattice_init(N)
        best_score = _compute_score(best_points)

    # Stage 1: global rotation search
    if time.time() < deadline - 100:
        remaining_time = min(5.0, deadline - 100 - time.time())
        if remaining_time > 1.0:
            best_points, best_score = _global_rotation_search(best_points, n_angles=36, time_limit=remaining_time)

    # Stage 2: coarse soft-min optimization (fast objective)
    if time.time() < deadline - 70:
        remaining_time = min(35.0, deadline - 70 - time.time())
        if remaining_time > 10.0:
            best_points, best_score = _local_optimize_softmin(
                best_points, time_limit=remaining_time,
                beta_schedule=[10.0, 50.0, 200.0, 1000.0], max_iter=300)

    # Stage 3: spring mechanics (coarse)
    if time.time() < deadline - 45:
        remaining_time = min(20.0, deadline - 45 - time.time())
        if remaining_time > 5.0:
            best_points, best_score = _local_optimize_spring(
                best_points, time_limit=remaining_time, k_spring=50.0,
                beta_schedule=[500.0, 2000.0], max_iter=150)

    # Stage 4: radial displacement refinement
    if time.time() < deadline - 30:
        remaining_time = min(10.0, deadline - 30 - time.time())
        if remaining_time > 3.0:
            best_points, best_score = _radial_displacement_refine(best_points, time_limit=remaining_time)

    # Stage 5: fine soft-min optimization
    if time.time() < deadline - 18:
        remaining_time = min(15.0, deadline - 18 - time.time())
        if remaining_time > 5.0:
            best_points, best_score = _local_optimize_softmin(
                best_points, time_limit=remaining_time,
                beta_schedule=[2000.0, 10000.0], max_iter=300)

    # Stage 6: force-directed refinement
    if time.time() < deadline - 8:
        remaining_time = min(6.0, deadline - 8 - time.time())
        if remaining_time > 2.0:
            fd_points, fd_score = _force_directed_refine(best_points, iterations=300, time_limit=remaining_time)
            if fd_score > best_score:
                best_points = fd_points
                best_score = fd_score

    # Stage 7: high-precision polish
    if time.time() < deadline - 3:
        remaining_time = min(4.0, deadline - 3 - time.time())
        if remaining_time > 1.0:
            best_points, best_score = _local_optimize_softmin(
                best_points, time_limit=remaining_time,
                beta_schedule=[50000.0, 200000.0], max_iter=300)

    # Final validation and clamping
    best_points = _clamp(best_points)
    best_score = _compute_score(best_points)

    if not np.all(np.isfinite(best_points)):
        best_points = np.nan_to_num(best_points, nan=0.0, posinf=0.9, neginf=-0.9)
        best_points = _clamp(best_points)
        best_score = _compute_score(best_points)

    return best_points

# EVOLVE-BLOCK-END
