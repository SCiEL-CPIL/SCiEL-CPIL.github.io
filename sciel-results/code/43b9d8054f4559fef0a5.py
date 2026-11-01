"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 150.0
SAFETY_MARGIN = 10.0


def _compute_score(points):
    """Compute the score: min of half-min-pair-distance and min distance to boundary."""
    n = len(points)
    if n == 0:
        return 0.0

    # Min distance to boundary
    dists_to_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = float(np.min(dists_to_boundary))

    # Min pair distance via vectorized computation
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = float(np.min(dist_matrix))

    return min(min_pair / 2.0, min_boundary)


def _compute_score_fast(points, cache=None):
    """Compute score with optional cache. Returns (score, dist_matrix, boundary_dists)."""
    n = len(points)
    if n == 0:
        return 0.0, None, None

    # Min distance to boundary
    norms = np.linalg.norm(points, axis=1)
    dists_to_boundary = 1.0 - norms
    min_boundary = float(np.min(dists_to_boundary))

    # Min pair distance
    if cache is not None and cache.shape == (n, n):
        dist_matrix = cache
    else:
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist_matrix, np.inf)
    min_pair = float(np.min(dist_matrix))

    return min(min_pair / 2.0, min_boundary), dist_matrix, dists_to_boundary


def _soft_min_gradient(pos, t=80.0):
    """
    Compute gradient of the soft-min of all pairwise and boundary distances.
    We want to maximize the minimum distance, so we minimize the soft-min.
    Returns (gradient, soft_min_value, actual_min).
    """
    n = len(pos)
    pos = pos.reshape(n, 2)

    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)

    # Boundary distances
    norms = np.linalg.norm(pos, axis=1)
    boundary_dists = 1.0 - norms

    # Flatten pairwise distances (upper triangle)
    iu = np.triu_indices(n, k=1)
    pair_dists = dist_matrix[iu]

    # All distances
    all_dists = np.concatenate([pair_dists, boundary_dists])

    # Compute weights using log-sum-exp for numerical stability
    d_min = np.min(all_dists)
    shifted = all_dists - d_min
    exp_terms = np.exp(-t * shifted)
    S = np.sum(exp_terms)

    # Gradient of soft-min: f = (1/t)*log(S) + d_min
    # df/dx = -(1/S) * sum_k exp(-t*(d_k - d_min)) * d(d_k)/dx
    #        = -sum_k w_k * d(d_k)/dx  where w_k = exp(-t*(d_k-d_min))/S

    grad = np.zeros_like(pos)

    # Pairwise gradient
    pair_weights = exp_terms[:len(pair_dists)] / S
    diff_pairs = diff[iu]
    norm_pairs = pair_dists[:, np.newaxis]
    norm_pairs[norm_pairs < 1e-12] = 1e-12
    unit_vecs = diff_pairs / norm_pairs

    i_idx, j_idx = iu
    np.add.at(grad, i_idx, -pair_weights[:, np.newaxis] * unit_vecs)
    np.add.at(grad, j_idx, pair_weights[:, np.newaxis] * unit_vecs)

    # Boundary gradient
    boundary_weights = exp_terms[len(pair_dists):] / S
    norms_col = norms[:, np.newaxis]
    norms_col[norms_col < 1e-12] = 1e-12
    unit_pos = pos / norms_col
    # d(1-|x|)/dx = -x/|x|, so contribution is -w * (-unit_pos) = w * unit_pos
    grad += boundary_weights[:, np.newaxis] * unit_pos

    soft_min_val = (1.0 / t) * np.log(S) + d_min
    return grad, soft_min_val, d_min


def _local_optimize(points, max_iter=500, lr=0.005, deadline=None):
    """
    Local optimization using L-BFGS-B on the soft-min of distances with
    annealed temperature. Maximizes the minimum pairwise and boundary distance.
    """
    n = len(points)
    best_points = points.copy()
    best_score = _compute_score(best_points)
    flat = best_points.reshape(-1).copy()

    # Anneal temperature: start smooth (low t) -> sharp (high t)
    t_start = 20.0
    t_end = 300.0

    def _obj_and_grad(flat_x, t_val):
        pts = flat_x.reshape(n, 2)
        # Soft penalty for being outside unit circle
        norms = np.linalg.norm(pts, axis=1)
        outside = np.maximum(norms - 1.0, 0.0)
        penalty = 100.0 * np.sum(outside ** 2)
        pen_grad = 200.0 * np.where((norms - 1.0) > 0, 1.0, 0.0)
        # pts shape (n,2), norms shape (n,)
        # d(outside^2)/d(pts) = 2*outside * pts/norms for outside>0
        pen_grad_pts = np.zeros_like(pts)
        mask_out = norms > 1.0
        if np.any(mask_out):
            pen_grad_pts[mask_out] = 2.0 * 100.0 * (norms[mask_out] - 1.0)[:, np.newaxis] * pts[mask_out] / norms[mask_out, np.newaxis]

        grad_sm, sm_val, _ = _soft_min_gradient(pts, t_val)
        # We want to maximize the soft-min, so minimize -soft_min
        obj = -sm_val + penalty
        grad = -grad_sm + pen_grad_pts
        return obj, grad.reshape(-1)

    # Multi-phase L-BFGS-B with annealed temperature
    phases = [
        (t_start, t_start * 3, 80),   # smooth, many iterations
        (t_start * 3, t_start * 10, 100),
        (t_start * 10, t_end, 150),
    ]

    for phase_idx, (t_lo, t_hi, max_iter_phase) in enumerate(phases):
        if deadline is not None and time.time() > deadline:
            break

        # Create a closure with fixed temperature for this phase
        # Actually, let's use a fixed t per phase for L-BFGS-B
        t_phase = (t_lo + t_hi) / 2.0

        def _obj(flat_x):
            return _obj_and_grad(flat_x, t_phase)[0]

        def _grad(flat_x):
            return _obj_and_grad(flat_x, t_phase)[1]

        try:
            res = minimize(
                _obj, flat, jac=_grad,
                method='L-BFGS-B',
                options={'maxiter': max_iter_phase, 'maxfun': max_iter_phase * 3,
                         'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            if res.x is not None:
                new_pts = res.x.reshape(n, 2)
                # Clamp to unit circle
                norms = np.linalg.norm(new_pts, axis=1)
                mask = norms > 1.0
                if np.any(mask):
                    new_pts[mask] = new_pts[mask] / norms[mask, np.newaxis]
                score = _compute_score(new_pts)
                if score > best_score:
                    best_score = score
                    best_points = new_pts.copy()
                    flat = best_points.reshape(-1).copy()
        except Exception:
            pass

    # Final exact score
    best_score = _compute_score(best_points)
    return best_points


def _hex_lattice_init(spacing, angle_deg=0):
    """Generate hexagonal lattice points within the unit disk."""
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    k = int(1.0 / spacing) + 3
    rows = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            # Rotate
            xr = x * cos_a - y * sin_a
            yr = x * sin_a + y * cos_a
            rows.append((xr, yr))

    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]

    if len(points) < N:
        return None

    # Sort by distance from center and take N closest
    radius = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(radius, kind="stable")[:N]
    return points[idx]


def _radial_ring_init():
    """
    Place circles in concentric rings with optimal spacing.
    Uses a greedy approach: place rings from center outward, fitting as many
    circles as possible in each ring given the target spacing.
    """
    best_points = None
    best_score = -np.inf

    # Try different base spacings
    for base_spacing in [0.10, 0.105, 0.11, 0.115, 0.12, 0.125, 0.13]:
        points = []
        r = 0.0  # Start at center
        placed = 0

        # Place one at center
        if N >= 1:
            points.append((0.0, 0.0))
            placed += 1

        # Place rings outward
        ring_idx = 1
        while placed < N:
            r = ring_idx * base_spacing
            if r > 0.95:
                break
            # Max circles in this ring: floor(2*pi*r / base_spacing)
            max_in_ring = int(2 * np.pi * r / base_spacing)
            if max_in_ring < 1:
                max_in_ring = 1
            # Don't exceed remaining
            count = min(max_in_ring, N - placed)
            # Stagger offset based on ring index
            offset = (ring_idx % 2) * np.pi / max(1, count)
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + offset
            for a in angles:
                points.append((r * np.cos(a), r * np.sin(a)))
            placed += count
            ring_idx += 1

        if len(points) != N:
            # Trim or pad
            if len(points) > N:
                points = points[:N]
            else:
                # Add remaining near center
                while len(points) < N:
                    points.append((0.0, 0.0))
                points = points[:N]

        points = np.array(points)
        # Verify all inside unit circle
        norms = np.linalg.norm(points, axis=1)
        if np.any(norms > 1.0):
            points[norms > 1.0] = points[norms > 1.0] / norms[norms > 1.0, np.newaxis] * 0.99

        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()

    return best_points


def _random_init():
    """Random initialization within unit disk."""
    points = np.random.randn(N, 2)
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, np.newaxis]
    # Scale to be within unit circle with some margin
    points *= 0.95
    return points


def _grid_init():
    """
    Initialize with a grid-like pattern inside the unit disk.
    This provides a more structured starting point than pure random.
    """
    # Create a grid and select points inside the unit circle
    step = 0.12
    k = int(1.0 / step) + 1
    points = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = i * step
            y = j * step
            if x * x + y * y <= 0.95 * 0.95:
                # Add small deterministic jitter to break symmetry
                jitter_x = 0.01 * np.sin(i * 7.3 + j * 11.1)
                jitter_y = 0.01 * np.cos(i * 13.7 + j * 5.9)
                points.append((x + jitter_x, y + jitter_y))

    points = np.array(points)
    if len(points) < N:
        return None

    # Sort by distance from center and take N closest
    radius = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(radius, kind="stable")[:N]
    return points[idx]


def _polish(points, deadline=None, max_iter=300):
    """
    High-precision polish pass using L-BFGS-B with very sharp soft-min temperature.
    """
    n = len(points)
    best_points = points.copy()
    best_score = _compute_score(best_points)
    flat = best_points.reshape(-1).copy()

    # Use very sharp temperature for precise optimization
    t_sharp = 500.0

    def _obj_and_grad(flat_x, t_val):
        pts = flat_x.reshape(n, 2)
        norms = np.linalg.norm(pts, axis=1)
        outside = np.maximum(norms - 1.0, 0.0)
        penalty = 100.0 * np.sum(outside ** 2)
        pen_grad_pts = np.zeros_like(pts)
        mask_out = norms > 1.0
        if np.any(mask_out):
            pen_grad_pts[mask_out] = 2.0 * 100.0 * (norms[mask_out] - 1.0)[:, np.newaxis] * pts[mask_out] / norms[mask_out, np.newaxis]

        grad_sm, sm_val, _ = _soft_min_gradient(pts, t_val)
        obj = -sm_val + penalty
        grad = -grad_sm + pen_grad_pts
        return obj, grad.reshape(-1)

    try:
        res = minimize(
            lambda x: _obj_and_grad(x, t_sharp)[0],
            flat,
            jac=lambda x: _obj_and_grad(x, t_sharp)[1],
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'maxfun': max_iter * 3,
                     'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
        )
        if res.x is not None:
            new_pts = res.x.reshape(n, 2)
            norms = np.linalg.norm(new_pts, axis=1)
            mask = norms > 1.0
            if np.any(mask):
                new_pts[mask] = new_pts[mask] / norms[mask, np.newaxis]
            score = _compute_score(new_pts)
            if score > best_score:
                best_score = score
                best_points = new_pts.copy()
    except Exception:
        pass

    return best_points


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN

    # Generate multiple initial configurations
    configs = []

    # Hexagonal lattice with different angles - try more angles and spacings
    for angle in [0, 10, 20, 30, 40, 50, 60]:
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        for _ in range(15):
            pts = _hex_lattice_init(spacing, angle)
            if pts is not None:
                configs.append(pts)
                break
            spacing *= 0.97

    # Radial ring initialization
    ring_pts = _radial_ring_init()
    if ring_pts is not None:
        configs.append(ring_pts)

    # Random initializations
    for _ in range(2):
        configs.append(_random_init())

    # Deterministic grid-like initialization with jitter
    grid_pts = _grid_init()
    if grid_pts is not None:
        configs.append(grid_pts)

    # Optimize each configuration and keep the best
    best_points = None
    best_score = -np.inf
    optimized_configs = []

    # Sort configs by initial score to prioritize promising ones
    scored_configs = []
    for cfg in configs:
        s = _compute_score(cfg)
        scored_configs.append((s, cfg))
    scored_configs.sort(key=lambda x: -x[0])

    for init_score, cfg in scored_configs:
        if time.time() > deadline - 15:  # Leave time for polish
            break

        # Local optimization
        optimized = _local_optimize(cfg, max_iter=300, deadline=deadline - 15)
        score = _compute_score(optimized)

        if score > best_score:
            best_score = score
            best_points = optimized.copy()

        # Keep top 3 configurations for potential re-optimization
        if len(optimized_configs) < 3:
            optimized_configs.append((score, optimized.copy()))
        else:
            # Replace worst if this is better
            worst_idx = np.argmin([s for s, _ in optimized_configs])
            if score > optimized_configs[worst_idx][0]:
                optimized_configs[worst_idx] = (score, optimized.copy())

    # If no configuration was optimized, use the first one as fallback
    if best_points is None:
        best_points = configs[0] if configs else np.zeros((N, 2))

    # Final high-precision polish pass if time permits
    if time.time() < deadline - 5.0:
        polished = _polish(best_points, deadline=deadline, max_iter=300)
        polished_score = _compute_score(polished)
        if polished_score > best_score:
            best_points = polished
            best_score = polished_score

    # If time remains, try polishing other good configurations
    if time.time() < deadline - 3.0 and len(optimized_configs) > 1:
        for score, pts in optimized_configs[1:]:
            if time.time() > deadline - 3.0:
                break
            if score >= best_score * 0.98:  # Only polish if close to best
                polished = _polish(pts, deadline=deadline, max_iter=150)
                polished_score = _compute_score(polished)
                if polished_score > best_score:
                    best_points = polished
                    best_score = polished_score

    return best_points
# EVOLVE-BLOCK-END
