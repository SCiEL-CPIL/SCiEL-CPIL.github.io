"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 150.0
SAFETY_MARGIN = 15.0


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


def _soft_min_objective_and_grad(x, t):
    """
    Objective: negative soft-min of all pairwise and boundary distances.
    Minimizing this maximizes the soft-min, approximating max-min packing.
    Returns (objective_value, gradient).
    """
    n = x.shape[0]
    pos = x.reshape(n, 2)

    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)

    # Boundary distances
    boundary_dists = 1.0 - np.linalg.norm(pos, axis=1)

    # Flatten pairwise distances (upper triangle)
    iu = np.triu_indices(n, k=1)
    pair_dists = dist_matrix[iu]

    # All distances
    all_dists = np.concatenate([pair_dists, boundary_dists])

    # Log-sum-exp for numerical stability
    d_min = np.min(all_dists)
    shifted = all_dists - d_min
    exp_terms = np.exp(-t * shifted)
    S = np.sum(exp_terms)

    # Soft-min value: (1/t)*log(S) + d_min
    soft_min_val = (1.0 / t) * np.log(S) + d_min

    # Gradient of soft-min w.r.t. positions
    grad_pos = np.zeros_like(pos)

    # Pairwise contribution
    pair_weights = exp_terms[:len(pair_dists)] / S
    diff_pairs = diff[iu]
    norm_pairs = pair_dists[:, np.newaxis]
    norm_pairs[norm_pairs < 1e-12] = 1e-12
    unit_vecs = diff_pairs / norm_pairs

    i_idx, j_idx = iu
    np.add.at(grad_pos, i_idx, -pair_weights[:, np.newaxis] * unit_vecs)
    np.add.at(grad_pos, j_idx, pair_weights[:, np.newaxis] * unit_vecs)

    # Boundary contribution
    boundary_weights = exp_terms[len(pair_dists):] / S
    norms = np.linalg.norm(pos, axis=1)[:, np.newaxis]
    norms[norms < 1e-12] = 1e-12
    unit_pos = pos / norms
    grad_pos += boundary_weights[:, np.newaxis] * unit_pos

    # We minimize -soft_min, so gradient is -grad_pos
    grad_flat = -grad_pos.flatten()

    # Objective is -soft_min
    obj_val = -soft_min_val

    return obj_val, grad_flat


def _local_optimize_lbfgs(points, t_start=40.0, t_end=150.0, deadline=None):
    """
    Local optimization using L-BFGS-B on the negative soft-min objective
    with cooldown-based temperature annealing.
    """
    n = points.shape[0]
    x0 = points.flatten()
    best_points = points.copy()
    best_score = _compute_score(best_points)

    # Use L-BFGS-B with vectorized objective/gradient
    t = t_start
    # Number of annealing stages
    n_stages = 4
    t_schedule = np.linspace(t_start, t_end, n_stages)

    for stage in range(n_stages):
        if deadline is not None and time.time() > deadline:
            break

        t = t_schedule[stage]

        # For each stage, run L-BFGS-B with a limited budget
        stage_budget = max(1, int(30 * (1.0 - stage * 0.15)))  # decreasing maxiter

        def fun_and_grad(x):
            return _soft_min_objective_and_grad(x, t)

        try:
            result = minimize(
                fun_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': stage_budget,
                    'ftol': 1e-10,
                    'gtol': 1e-6,
                    'maxls': 20,
                }
            )
            x0 = result.x
            new_points = x0.reshape(n, 2)

            # Keep points inside unit circle
            norms = np.linalg.norm(new_points, axis=1)
            mask = norms > 1.0
            if np.any(mask):
                new_points[mask] = new_points[mask] / norms[mask, np.newaxis]

            score = _compute_score(new_points)
            if score > best_score:
                best_score = score
                best_points = new_points.copy()

        except Exception:
            pass

        # Check deadline between stages
        if deadline is not None and time.time() > deadline:
            break

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
    Place circles in concentric rings. Each ring has a radial distance and
    an angular offset.
    """
    best_points = None
    best_score = -np.inf

    # Try different ring structures
    for n_rings in range(5, 16):
        # Distribute N circles among rings
        # Outer rings can hold more circles
        weights = []
        for i in range(n_rings):
            r_i = (i + 1) / n_rings * 0.95
            weights.append(2 * np.pi * r_i)

        total_w = sum(weights)
        counts = []
        remaining = N
        for i in range(n_rings - 1):
            c_i = max(1, int(round(N * weights[i] / total_w)))
            c_i = min(c_i, remaining - (n_rings - 1 - i))
            counts.append(c_i)
            remaining -= c_i
        counts.append(remaining)

        # Now compute positions
        points = []
        for i, count in enumerate(counts):
            if count <= 0:
                continue
            r_i = (i + 1) / n_rings * 0.95
            # Angular offset to stagger rings
            offset = (i * np.pi) / max(1, count)
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + offset
            for a in angles:
                points.append((r_i * np.cos(a), r_i * np.sin(a)))

        if len(points) != N:
            continue

        points = np.array(points)
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


def _final_polish(points, deadline=None):
    """
    High-precision polish pass using L-BFGS-B with very high temperature
    to focus tightly on the minimum distance constraint.
    """
    n = points.shape[0]
    x0 = points.flatten()
    best_points = points.copy()
    best_score = _compute_score(best_points)

    # Very high temperature for tight focus on the minimum
    t = 300.0

    def fun_and_grad(x):
        return _soft_min_objective_and_grad(x, t)

    try:
        result = minimize(
            fun_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 50,
                'ftol': 1e-12,
                'gtol': 1e-8,
                'maxls': 20,
            }
        )
        new_points = result.x.reshape(n, 2)

        # Keep points inside unit circle
        norms = np.linalg.norm(new_points, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            new_points[mask] = new_points[mask] / norms[mask, np.newaxis]

        score = _compute_score(new_points)
        if score > best_score:
            best_points = new_points.copy()
            best_score = score

    except Exception:
        pass

    return best_points


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN

    # Generate multiple initial configurations
    configs = []

    # Hexagonal lattice with different angles
    for angle in [0, 15, 30, 45, 60]:
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        for _ in range(20):
            pts = _hex_lattice_init(spacing, angle)
            if pts is not None:
                configs.append(pts)
                break
            spacing *= 0.98

    # Radial ring initialization
    ring_pts = _radial_ring_init()
    if ring_pts is not None:
        configs.append(ring_pts)

    # Random initializations
    for _ in range(3):
        configs.append(_random_init())

    # Optimize each configuration and keep the best
    best_points = None
    best_score = -np.inf

    for cfg in configs:
        if time.time() > deadline:
            break

        # Local optimization with L-BFGS-B and annealing
        optimized = _local_optimize_lbfgs(cfg, t_start=40.0, t_end=150.0, deadline=deadline)
        score = _compute_score(optimized)

        if score > best_score:
            best_score = score
            best_points = optimized.copy()

    # If no configuration was optimized, use the first one as fallback
    if best_points is None:
        best_points = configs[0] if configs else np.zeros((N, 2))

    # Final polish pass if there's time
    if time.time() < deadline - 10.0:
        polished = _final_polish(best_points, deadline=deadline)
        polished_score = _compute_score(polished)
        if polished_score > best_score:
            best_points = polished

    return best_points
# EVOLVE-BLOCK-END
