"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_score(points):
    """Compute the score: min of half-min-pair-distance and min distance to boundary."""
    n = len(points)
    if n == 0:
        return 0.0

    dists_to_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = float(np.min(dists_to_boundary))

    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = float(np.min(dist_matrix))

    return min(min_pair / 2.0, min_boundary)


def _soft_min_objective(x, t=50.0):
    """
    Soft-min objective for minimization.
    We want to maximize the minimum distance, so we minimize the negative soft-min.
    Returns (value, gradient).
    """
    n = N
    pos = x.reshape(n, 2)

    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist_matrix = np.sqrt(dist_sq)

    # Boundary distances
    norms = np.linalg.norm(pos, axis=1)
    boundary_dists = 1.0 - norms

    # Collect all distances
    iu = np.triu_indices(n, k=1)
    pair_dists = dist_matrix[iu]

    all_dists = np.concatenate([pair_dists, boundary_dists])

    d_min = np.min(all_dists)
    shifted = all_dists - d_min
    exp_terms = np.exp(-t * shifted)
    S = np.sum(exp_terms)
    soft_min_val = (1.0 / t) * np.log(S) + d_min

    # Compute gradient
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
    norms_safe = norms[:, np.newaxis]
    norms_safe[norms_safe < 1e-12] = 1e-12
    unit_pos = pos / norms_safe
    grad += boundary_weights[:, np.newaxis] * unit_pos

    # We want to maximize soft_min, so objective is -soft_min
    value = -soft_min_val
    grad_obj = -grad.reshape(-1)

    return value, grad_obj


def _physics_optimize(points, max_steps=3000, deadline=None):
    """
    Physics-based optimization: circles repel each other and the wall.
    Uses velocity Verlet integration with damping.
    """
    n = len(points)
    pos = points.astype(np.float64).copy()
    vel = np.zeros_like(pos)
    dt = 0.002
    damping = 0.995
    best_pos = pos.copy()
    best_score = _compute_score(pos)

    # Target radius based on initial score
    r_target = _compute_score(pos)

    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break

        # Compute forces
        forces = np.zeros_like(pos)

        # Pairwise repulsion (softened to avoid singularity)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        dist = np.sqrt(dist_sq)

        # Repulsive force: F = 1/d^2 for d < 2*r_target, else 0
        # Use a smooth potential
        min_dist = 2.0 * r_target
        mask = dist < min_dist * 1.5
        # Only compute forces for nearby pairs for efficiency
        # But for N=300, full computation is fine

        # Soft repulsion: F = k / d^3 * diff_vec for d < cutoff
        cutoff = min_dist * 2.0
        force_mask = dist < cutoff
        force_mask[~np.isfinite(dist)] = False

        # Compute forces vectorized
        inv_dist_cubed = np.zeros_like(dist)
        valid = force_mask & (dist > 1e-10)
        inv_dist_cubed[valid] = 1.0 / (dist[valid] ** 3)

        force_mat = inv_dist_cubed[:, :, np.newaxis] * diff
        forces -= np.sum(force_mat, axis=1)

        # Wall repulsion
        norms = np.linalg.norm(pos, axis=1)
        wall_mask = norms > 1.0 - r_target
        if np.any(wall_mask):
            wall_force = np.zeros_like(pos)
            wall_force[wall_mask] = (pos[wall_mask] / norms[wall_mask, np.newaxis]) * (norms[wall_mask] - (1.0 - r_target)) * 10.0
            forces -= wall_force

        # Update velocity and position
        vel = vel * damping + forces * dt
        pos = pos + vel * dt

        # Constrain to unit disk
        norms = np.linalg.norm(pos, axis=1)
        mask_out = norms > 1.0 - r_target
        if np.any(mask_out):
            pos[mask_out] = pos[mask_out] / norms[mask_out, np.newaxis] * (1.0 - r_target)
            vel[mask_out] = 0.0

        # Compute score
        score = _compute_score(pos)
        if score > best_score:
            best_score = score
            best_pos = pos.copy()
            r_target = score

    return best_pos


def _l_bfgs_optimize(points, deadline=None, t_start=30.0, t_end=200.0):
    """
    Optimize using L-BFGS-B on the soft-min objective.
    Adaptive temperature schedule.
    """
    x0 = points.reshape(-1)
    best_x = x0.copy()
    best_score = _compute_score(points)

    # Schedule of temperatures
    temps = [t_start, 50.0, 100.0, 150.0, t_end]

    for t in temps:
        if deadline is not None and time.time() > deadline:
            break

        def obj_grad(x):
            val, grad = _soft_min_objective(x, t=t)
            return val, grad

        try:
            result = minimize(
                obj_grad, x0, jac=True, method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            x0 = result.x
            score = _compute_score(x0.reshape(N, 2))
            if score > best_score:
                best_score = score
                best_x = x0.copy()
        except Exception:
            pass

    return best_x.reshape(N, 2)


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
            xr = x * cos_a - y * sin_a
            yr = x * sin_a + y * cos_a
            rows.append((xr, yr))

    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    mask = radius <= 1.0 - spacing / 2.0
    points = points[mask]

    if len(points) < N:
        return None

    radius = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(radius, kind="stable")[:N]
    return points[idx]


def _perturbed_hex_init():
    """
    Initialize on a dense hexagonal lattice with random perturbations to break symmetry.
    """
    # Estimate spacing for ~300 points in unit disk
    # Area of disk = pi, area per point ~ pi/300, hex cell area ~ s^2 * sqrt(3)/2
    # s^2 * sqrt(3)/2 = pi/300 => s = sqrt(2*pi/(300*sqrt(3)))
    s_est = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))

    # Try several spacings
    best_points = None
    best_score = -np.inf

    for s_factor in [0.95, 0.98, 1.0, 1.02, 1.05]:
        s = s_est * s_factor
        pts = _hex_lattice_init(s, angle_deg=0)
        if pts is None:
            continue

        # Add small random perturbation
        rng = np.random.default_rng(42)
        perturbation = rng.normal(0, s * 0.1, size=pts.shape)
        pts_pert = pts + perturbation

        # Constrain to unit disk
        norms = np.linalg.norm(pts_pert, axis=1)
        mask = norms > 1.0 - s * 0.5
        if np.any(mask):
            pts_pert[mask] = pts_pert[mask] / norms[mask, np.newaxis] * (1.0 - s * 0.5)

        if len(pts_pert) != N:
            continue

        score = _compute_score(pts_pert)
        if score > best_score:
            best_score = score
            best_points = pts_pert.copy()

    return best_points


def _random_init():
    """Random initialization inside the unit disk."""
    rng = np.random.default_rng(123)
    angles = rng.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(rng.uniform(0, 1, N)) * 0.95
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return points


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Leave 15s safety margin

    best_points = None
    best_score = -np.inf

    # Generate multiple initializations
    inits = []

    # 1. Perturbed hexagonal lattice (primary)
    hex_pts = _perturbed_hex_init()
    if hex_pts is not None:
        score = _compute_score(hex_pts)
        inits.append((score, hex_pts))

    # 2. Hexagonal lattice with various angles
    for angle_deg in [0, 15, 30, 45]:
        s_est = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
        for s_factor in [0.97, 1.0, 1.03]:
            s = s_est * s_factor
            pts = _hex_lattice_init(s, angle_deg)
            if pts is not None:
                score = _compute_score(pts)
                inits.append((score, pts))

    # 3. Random initialization
    rand_pts = _random_init()
    score = _compute_score(rand_pts)
    inits.append((score, rand_pts))

    # Sort by score, keep top few
    inits.sort(key=lambda x: -x[0])
    top_inits = inits[:5] if len(inits) >= 5 else inits

    # Physics-based optimization on best initializations
    for score, pts in top_inits:
        if time.time() > deadline - 30:
            break

        optimized = _physics_optimize(pts, max_steps=2000, deadline=deadline - 20)
        opt_score = _compute_score(optimized)

        if opt_score > best_score:
            best_score = opt_score
            best_points = optimized.copy()

    # L-BFGS-B refinement on the best result
    if best_points is not None and time.time() < deadline - 15:
        refined = _l_bfgs_optimize(best_points, deadline=deadline - 5)
        ref_score = _compute_score(refined)
        if ref_score > best_score:
            best_score = ref_score
            best_points = refined.copy()

    # Final physics polish
    if best_points is not None and time.time() < deadline - 10:
        polished = _physics_optimize(best_points, max_steps=1000, deadline=deadline - 3)
        pol_score = _compute_score(polished)
        if pol_score > best_score:
            best_score = pol_score
            best_points = polished.copy()

    # Final L-BFGS polish
    if best_points is not None and time.time() < deadline - 5:
        final_refined = _l_bfgs_optimize(best_points, deadline=deadline - 1)
        final_score = _compute_score(final_refined)
        if final_score > best_score:
            best_score = final_score
            best_points = final_refined.copy()

    # Fallback if nothing worked
    if best_points is None:
        s_est = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0)))
        pts = _hex_lattice_init(s_est, 0)
        if pts is not None:
            best_points = pts
        else:
            angles = np.linspace(0, 2 * np.pi * N, N)
            radii = np.linspace(0, 0.95, N)
            best_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    # Ensure all points are strictly inside the unit circle
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_points[mask] = best_points[mask] / norms[mask, np.newaxis] * 0.9999

    return best_points
# EVOLVE-BLOCK-END
