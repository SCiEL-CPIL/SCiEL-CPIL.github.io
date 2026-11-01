import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = time.time() + 142.0  # 8s safety margin before 150s kill


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if pts is None or len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists > 1.0 + 1e-9):
        return 0.0
    r_cont = 1.0 - np.max(dists)
    if r_cont < 0:
        return 0.0
    if N < 2:
        return r_cont
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dmat = np.sqrt(dx ** 2 + dy ** 2)
    np.fill_diagonal(dmat, np.inf)
    min_pair = np.min(dmat)
    r_pair = min_pair / 2.0
    return min(r_cont, r_pair)


def _soft_min_obj_and_grad(flat, beta):
    """
    Vectorized soft-min surrogate: minimize -soft_min(radius) where radius is
    the min of all pairwise half-distances and boundary distances.
    """
    pts = flat.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e6, np.zeros(2 * N)

    # Boundary distances
    dists_origin = np.hypot(pts[:, 0], pts[:, 1])
    boundary_dists = 1.0 - dists_origin

    # Pairwise distances (vectorized)
    iu = np.triu_indices(N, k=1)
    dx = pts[iu[0], 0] - pts[iu[1], 0]
    dy = pts[iu[0], 1] - pts[iu[1], 1]
    d_pairs = np.sqrt(dx * dx + dy * dy)
    half_d_pairs = 0.5 * d_pairs

    # All critical radii
    all_radii = np.concatenate([boundary_dists, half_d_pairs])

    # Soft min: x_max - (1/beta) * log(sum(exp(-beta*(x_k - x_max))))
    x_max = np.max(all_radii)
    shifted = -beta * (all_radii - x_max)
    shifted -= np.max(shifted)  # numerical stability
    sum_exp = np.sum(np.exp(shifted))
    log_sum_exp = np.max(shifted) + np.log(sum_exp)
    soft_min_val = x_max - log_sum_exp / beta

    obj_val = -soft_min_val

    # Weights for gradient
    weights = np.exp(shifted) / sum_exp

    grad = np.zeros(2 * N)

    # Boundary gradient contribution
    w_b = weights[0:N]
    safe_d = np.where(dists_origin < 1e-10, 1.0, dists_origin)
    # d(boundary_dist)/d(pts[i]) = -pts[i] / dists_origin[i]
    # d(-soft_min)/d(pts) = +sum w_k * d(radius_k)/d(pts)
    # For boundary: d(1-|p|)/d(p) = -p/|p|
    # So gradient of -soft_min w.r.t. pts includes: w_b * (-pts/safe_d)
    # But we want to MAXIMIZE soft_min, so minimize -soft_min
    # grad(-soft_min) = -sum w_k * d(radius_k)/d(pts)
    # For boundary: -w_b * (-pts/safe_d) = w_b * pts/safe_d
    # Wait, let me redo:
    # soft_min = x_max - (1/beta)*log(sum exp(-beta*(x_k - x_max)))
    # d(soft_min)/d(pts) = sum_k w_k * d(x_k)/d(pts) where w_k = exp(-beta*(x_k-x_max))/sum
    # We minimize -soft_min, so grad = -sum_k w_k * d(x_k)/d(pts)
    # For boundary: d(x_k)/d(pts_i) = -pts_i / dists_origin[i]
    # So grad contribution = -w_b[i] * (-pts_i / safe_d[i]) = w_b[i] * pts_i / safe_d[i]
    # This pushes points AWAY from center when boundary is tight -- WRONG direction
    # Let me re-examine: we want to maximize min radius. If boundary is the min,
    # we want to move points inward. d(1-|p|)/d(p) = -p/|p|. 
    # grad(soft_min) w.r.t. pts_i includes w_b[i] * (-pts_i/|p_i|)
    # grad(-soft_min) = -w_b[i] * (-pts_i/|p_i|) = w_b[i] * pts_i/|p_i|
    # This is outward... that's wrong. Let me reconsider.
    
    # Actually: soft_min approximates min(x_k). d(min)/d(pts) ≈ d(x_min)/d(pts)
    # If boundary is min, d(1-|p_i|)/d(p_i) = -p_i/|p_i| (inward direction)
    # To maximize soft_min, we want to move in direction +d(soft_min)/d(pts)
    # = w_b[i] * (-p_i/|p_i|) which is inward. Correct!
    # But we minimize -soft_min, so grad(-soft_min) = -d(soft_min)/d(pts)
    # = -w_b[i] * (-p_i/|p_i|) = w_b[i] * p_i/|p_i| (outward)
    # L-BFGS-B minimizes, so it will move AGAINST the gradient = inward. Correct!
    
    grad[:N*2:2] += w_b * pts[:, 0] / safe_d
    grad[1::2] += w_b * pts[:, 1] / safe_d

    # Pairwise gradient contribution
    w_p = weights[N:]
    safe_dp = np.where(d_pairs < 1e-10, 1.0, d_pairs)
    # d(half_d_pair)/d(pts_i) = (pts_i - pts_j) / (2 * d_pair)
    # grad(-soft_min) contribution = -w_p * d(half_d)/d(pts)
    # For point i: -w_p * (pts_i - pts_j)/(2*d)
    # For point j: -w_p * (pts_j - pts_i)/(2*d) = w_p * (pts_i - pts_j)/(2*d)
    
    unit_x = dx / safe_dp
    unit_y = dy / safe_dp
    
    # Contribution to point i (negative direction)
    gi_x = -0.5 * w_p * unit_x
    gi_y = -0.5 * w_p * unit_y
    # Contribution to point j (positive direction)  
    gj_x = 0.5 * w_p * unit_x
    gj_y = 0.5 * w_p * unit_y
    
    np.add.at(grad, 2 * iu[0], gi_x)
    np.add.at(grad, 2 * iu[0] + 1, gi_y)
    np.add.at(grad, 2 * iu[1], gj_x)
    np.add.at(grad, 2 * iu[1] + 1, gj_y)

    return obj_val, grad


def _hex_lattice_init():
    """Generate a hexagonal lattice packing as initial guess."""
    best_pts = None
    best_score = -1.0

    for spacing in [0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29]:
        k = int(1.0 / spacing) + 4
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])

        mask = radius <= 1.0
        points = points[mask]
        radius = radius[mask]

        if len(points) >= N:
            idx = np.argsort(radius, kind="stable")[:N]
            pts = points[idx]
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts

    if best_pts is not None:
        return best_pts

    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.9, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _random_init():
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.95, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _ring_init(n_rings=5):
    """Ring-based initialization matching known optimal structure."""
    best_pts = None
    best_score = -1.0

    ring_configs = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 21, 10],
        [1, 6, 12, 15, 16],
        [1, 6, 12, 14, 17],
    ]

    for config in ring_configs:
        if sum(config) != N:
            continue

        points = [[0.0, 0.0]]
        ring_idx = 0
        for ring_size in config[1:]:
            ring_idx += 1
            avg_spacing = np.sqrt(np.pi * 1.0**2 / (N * np.pi))
            ring_radius = ring_idx * avg_spacing * 0.95
            angles = np.linspace(0, 2 * np.pi, ring_size, endpoint=False)
            angles += np.random.uniform(-0.05, 0.05, ring_size)
            for angle in angles:
                x = ring_radius * np.cos(angle)
                y = ring_radius * np.sin(angle)
                points.append([x, y])

        pts = np.array(points[:N])
        dists = np.hypot(pts[:, 0], pts[:, 1])
        mask = dists > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / dists[mask, np.newaxis]

        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts

    if best_pts is not None:
        return best_pts
    return _random_init()


def _dynamics_refine(pts, max_iters=3000, dt=0.001, damping=0.95):
    """
    Short dynamics simulation to escape local optima.
    Uses vectorized force computation.
    """
    pts = pts.copy().astype(np.float64)
    vel = np.zeros_like(pts)
    best_score = _score(pts)
    best_pts = pts.copy()

    iu = np.triu_indices(N, k=1)
    force_scale = 30.0
    wall_scale = 20.0

    for it in range(max_iters):
        if time.time() > _DEADLINE - 15.0:
            break

        force = np.zeros_like(pts)

        # Current best radius estimate
        dists_origin = np.hypot(pts[:, 0], pts[:, 1])
        r_cont = 1.0 - np.max(dists_origin)
        dx = pts[iu[0], 0] - pts[iu[1], 0]
        dy = pts[iu[0], 1] - pts[iu[1], 1]
        d_pairs = np.sqrt(dx * dx + dy * dy)
        r_pair = 0.5 * np.min(d_pairs)
        r_target = min(r_cont, r_pair) * 1.005

        # Pairwise repulsion (vectorized)
        d_safe = np.where(d_pairs < 1e-10, 1e-10, d_pairs)
        overlap = np.maximum(0, 2 * r_target - d_pairs)
        f_mag = force_scale * overlap / d_safe
        fx = f_mag * dx
        fy = f_mag * dy

        np.add.at(force, iu[0], np.column_stack([fx, fy]))
        np.add.at(force, iu[1], -np.column_stack([fx, fy]))

        # Wall force
        excess = np.maximum(0, dists_origin + r_target - 1.0)
        f_wall = wall_scale * excess
        safe_dist = np.where(dists_origin < 1e-10, 1e-10, dists_origin)
        force[:, 0] -= f_wall * pts[:, 0] / safe_dist
        force[:, 1] -= f_wall * pts[:, 1] / safe_dist

        # Update
        vel = damping * vel + dt * force
        pts = pts + dt * vel

        # Project inside
        nr = np.hypot(pts[:, 0], pts[:, 1])
        mask = nr > 1.0
        if np.any(mask):
            pts[mask] /= nr[mask, None] * 1.0001

        current_score = _score(pts)
        if current_score > best_score:
            best_score = current_score
            best_pts = pts.copy()

        if it % 200 == 0 and it > 0:
            r_target *= 1.002

    return best_pts, best_score


def _optimize_from_start(pts, beta_schedule, max_iter_per_beta=300):
    """Run L-BFGS-B with beta annealing."""
    current_pts = pts.copy()
    best_pts = current_pts.copy()
    best_score = _score(current_pts)

    for beta in beta_schedule:
        if time.time() > _DEADLINE - 12.0:
            break

        try:
            res = minimize(
                _soft_min_obj_and_grad,
                current_pts.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': max_iter_per_beta, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            new_pts = res.x.reshape(N, 2)
            dists = np.hypot(new_pts[:, 0], new_pts[:, 1])
            mask = dists > 1.0
            if np.any(mask):
                new_pts[mask] = new_pts[mask] / dists[mask, np.newaxis]

            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                current_pts = new_pts.copy()
        except Exception:
            pass

    return best_pts, best_score


def construct():
    start_time = time.time()

    # Multiple initializations
    inits = []
    inits.append(_hex_lattice_init())
    inits.append(_ring_init())
    for _ in range(5):
        inits.append(_random_init())

    best_pts = inits[0]
    best_score = _score(best_pts)

    # Beta schedules
    beta_sched_1 = [10, 50, 200, 800, 3000, 10000]
    beta_sched_2 = [50, 200, 1000, 5000, 20000]

    # Optimize from each initialization
    for init_idx, init_pts in enumerate(inits):
        if time.time() > _DEADLINE - 25.0:
            break

        if init_idx % 2 == 0:
            schedule = beta_sched_1
        else:
            schedule = beta_sched_2

        # Dynamics pre-refinement for non-trivial inits
        if init_idx > 0 and time.time() < _DEADLINE - 30.0:
            dyn_pts, dyn_score = _dynamics_refine(init_pts, max_iters=1500)
            if dyn_score > _score(init_pts):
                init_pts = dyn_pts

        candidate_pts, candidate_score = _optimize_from_start(init_pts, schedule, max_iter_per_beta=200)

        if candidate_score > best_score:
            best_score = candidate_score
            best_pts = candidate_pts.copy()

    # High-beta polish
    if time.time() < _DEADLINE - 20.0:
        for beta in [50000, 200000, 500000, 1000000]:
            if time.time() > _DEADLINE - 10.0:
                break
            try:
                res = minimize(
                    _soft_min_obj_and_grad,
                    best_pts.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(-1.0, 1.0)] * (2 * N),
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-11, 'disp': False}
                )
                new_pts = res.x.reshape(N, 2)
                dists = np.hypot(new_pts[:, 0], new_pts[:, 1])
                mask = dists > 1.0
                if np.any(mask):
                    new_pts[mask] = new_pts[mask] / dists[mask, np.newaxis]
                new_score = _score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
            except Exception:
                pass

    # Final dynamics + gradient combo
    if time.time() < _DEADLINE - 8.0:
        dyn_pts, dyn_score = _dynamics_refine(best_pts, max_iters=1000)
        if dyn_score > best_score:
            best_score = dyn_score
            best_pts = dyn_pts.copy()

        if time.time() < _DEADLINE - 4.0:
            try:
                res = minimize(
                    _soft_min_obj_and_grad,
                    best_pts.flatten(),
                    args=(500000,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(-1.0, 1.0)] * (2 * N),
                    options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12, 'disp': False}
                )
                new_pts = res.x.reshape(N, 2)
                dists = np.hypot(new_pts[:, 0], new_pts[:, 1])
                mask = dists > 1.0
                if np.any(mask):
                    new_pts[mask] = new_pts[mask] / dists[mask, np.newaxis]
                new_score = _score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
            except Exception:
                pass

    return best_pts
