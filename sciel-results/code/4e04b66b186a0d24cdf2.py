import time
import numpy as np
from scipy.optimize import minimize

N = 50
_START_TIME = None
_DEADLINE = 138.0  # safety margin before 150s kill


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

    if N < 2:
        return r_cont

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


def _velocity_verlet_relax(pts, target_r, n_steps=500, damping_rate=0.9995):
    """
    Velocity-Verlet integrator with adaptive damping for packing optimization.
    Uses short-range repulsive forces and confining harmonic potential.
    """
    pts = pts.copy()
    velocities = np.zeros_like(pts)
    target_d = 2.0 * target_r
    iu = np.triu_indices(N, k=1)
    i_idx, m_idx = iu

    # Force parameters
    repulsion_strength = 10.0
    boundary_strength = 5.0
    centering_strength = 0.1

    for step in range(n_steps):
        if _START_TIME is not None and time.time() - _START_TIME > _DEADLINE - 10:
            break

        # Compute forces
        forces = np.zeros_like(pts)

        # Pairwise repulsive forces (short-range)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist_mat, np.inf)

        d_pairs = dist_mat[iu]
        mask_pair = (d_pairs < target_d * 1.5) & (d_pairs > 1e-10)
        if np.any(mask_pair):
            ii = i_idx[mask_pair]
            mm = m_idx[mask_pair]
            d = d_pairs[mask_pair]
            vecs = pts[ii] - pts[mm]
            units = vecs / d[:, np.newaxis]
            # Strong repulsion when overlapping, weaker when close
            overlaps = np.maximum(0, target_d - d)
            close = np.maximum(0, target_d * 1.5 - d)
            strengths = repulsion_strength * (overlaps * 2.0 + close * 0.1)
            np.add.at(forces, ii, units * strengths[:, np.newaxis])
            np.add.at(forces, mm, -units * strengths[:, np.newaxis])

        # Boundary constraint: harmonic potential pushing inward
        dists_origin = np.linalg.norm(pts, axis=1)
        boundary_limit = 1.0 - target_r
        mask_bnd = dists_origin > boundary_limit
        if np.any(mask_bnd):
            bi = np.where(mask_bnd)[0]
            excess = dists_origin[bi] - boundary_limit
            unit_out = pts[bi] / np.maximum(dists_origin[bi], 1e-8)[:, np.newaxis]
            forces[bi] -= boundary_strength * excess[:, np.newaxis] * unit_out

        # Weak centering force to prevent drift
        forces -= centering_strength * pts

        # Velocity-Verlet integration
        damping = damping_rate ** step
        velocities = damping * velocities + 0.002 * forces
        pts += 0.002 * velocities

        # Project back into unit disk
        dists = np.linalg.norm(pts, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / dists[mask, np.newaxis]

    return pts


def _force_directed_relax(pts, r_target, steps=200, lr=0.005):
    """
    Force-directed relaxation: push circles apart and away from boundary.
    Vectorized implementation for speed.
    """
    pts = pts.copy()
    iu = np.triu_indices(N, k=1)
    i_idx, m_idx = iu

    for step in range(steps):
        if _START_TIME is not None and time.time() - _START_TIME > _DEADLINE - 10:
            break

        dists_origin = np.linalg.norm(pts, axis=1)
        boundary_dists = 1.0 - dists_origin

        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist_mat, np.inf)

        force = np.zeros_like(pts)

        d_pairs = dist_mat[iu]

        # Vectorized pairwise forces
        mask_pair = (d_pairs < 2 * r_target) & (d_pairs > 1e-10)
        if np.any(mask_pair):
            ii = i_idx[mask_pair]
            mm = m_idx[mask_pair]
            d = d_pairs[mask_pair]
            vecs = pts[ii] - pts[mm]
            units = vecs / d[:, np.newaxis]
            strengths = (2 * r_target - d) * 0.5
            np.add.at(force, ii, units * strengths[:, np.newaxis])
            np.add.at(force, mm, -units * strengths[:, np.newaxis])

        # Vectorized boundary forces
        mask_bnd = boundary_dists < r_target
        if np.any(mask_bnd):
            bi = np.where(mask_bnd)[0]
            safe_dists = np.where(dists_origin[bi] < 1e-8, 1.0, dists_origin[bi])
            unit_out = pts[bi] / safe_dists[:, np.newaxis]
            strengths = r_target - boundary_dists[bi]
            force[bi] -= unit_out * strengths[:, np.newaxis]

        pts += lr * force

        # Project back into unit disk
        dists = np.linalg.norm(pts, axis=1)
        mask = dists > 1.0 - r_target
        if np.any(mask):
            pts[mask] = pts[mask] * (1.0 - r_target) / dists[mask, np.newaxis]

    return pts


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


def _ring_init():
    """
    Initialize circles in concentric rings matching the known optimal structure.
    Known best packing uses rings: 1 (center) + 6 + 12 + 18 + 13 = 50
    """
    pts = np.zeros((N, 2))

    # Ring 0: center
    pts[0] = [0.0, 0.0]

    # Ring 1: 6 circles
    r1 = 0.25
    angles1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    pts[1:7, 0] = r1 * np.cos(angles1)
    pts[1:7, 1] = r1 * np.sin(angles1)

    # Ring 2: 12 circles
    r2 = 0.50
    angles2 = np.linspace(0, 2 * np.pi, 12, endpoint=False) + np.pi / 12
    pts[7:19, 0] = r2 * np.cos(angles2)
    pts[7:19, 1] = r2 * np.sin(angles2)

    # Ring 3: 18 circles
    r3 = 0.72
    angles3 = np.linspace(0, 2 * np.pi, 18, endpoint=False)
    pts[19:37, 0] = r3 * np.cos(angles3)
    pts[19:37, 1] = r3 * np.sin(angles3)

    # Ring 4: 13 circles
    r4 = 0.87
    angles4 = np.linspace(0, 2 * np.pi, 13, endpoint=False) + np.pi / 26
    pts[37:50, 0] = r4 * np.cos(angles4)
    pts[37:50, 1] = r4 * np.sin(angles4)

    return pts


def _random_init():
    """Random initialization with bias toward uniform area coverage."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.95, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _perturb(pts, scale=0.01):
    """Add small random perturbation to escape local optima."""
    return pts + np.random.normal(0, scale, size=pts.shape)


def _project_to_disk(pts):
    """Project points back into the unit disk."""
    dists = np.linalg.norm(pts, axis=1)
    mask = dists > 1.0
    if np.any(mask):
        pts = pts.copy()
        pts[mask] = pts[mask] / dists[mask, np.newaxis]
    return pts


def _soft_min_optimize(pts, beta_schedule, max_iter_per_beta=300):
    """Run L-BFGS-B with a schedule of increasing beta values."""
    current_pts = pts.copy()
    best_pts = current_pts.copy()
    best_score = _score(current_pts)

    for beta in beta_schedule:
        if _START_TIME is not None and time.time() - _START_TIME > _DEADLINE - 15:
            break

        try:
            # Tighter tolerances for higher beta values
            if beta >= 1000:
                ftol = 1e-13
                gtol = 1e-9
                max_iter = max_iter_per_beta
            else:
                ftol = 1e-12
                gtol = 1e-8
                max_iter = max_iter_per_beta

            res = minimize(
                _soft_min_obj_and_grad,
                current_pts.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False, 'maxls': 50}
            )

            new_pts = res.x.reshape(N, 2)
            new_pts = _project_to_disk(new_pts)

            new_score = _score(new_pts)

            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                current_pts = new_pts.copy()

        except Exception:
            pass

    return best_pts, best_score


def _full_optimize(pts):
    """
    Full optimization pipeline:
    1. Velocity-Verlet relaxation (momentum-based)
    2. Force-directed relaxation (gradient-based)
    3. Soft-min L-BFGS-B with beta annealing
    4. Final high-precision polish
    """
    initial_score = _score(pts)

    # Step 1: Velocity-Verlet relaxation with momentum
    vv_pts = _velocity_verlet_relax(pts, initial_score, n_steps=400, damping_rate=0.9995)
    vv_score = _score(vv_pts)

    if vv_score > initial_score:
        pts = vv_pts
        initial_score = vv_score

    # Step 2: Force-directed relaxation
    relaxed = _force_directed_relax(pts, initial_score, steps=150, lr=0.005)
    relaxed_score = _score(relaxed)

    if relaxed_score > initial_score:
        pts = relaxed
        initial_score = relaxed_score

    # Step 3: Soft-min L-BFGS-B with beta annealing
    beta_schedule = [20, 100, 500, 2000, 10000, 50000]
    pts, score = _soft_min_optimize(pts, beta_schedule, max_iter_per_beta=400)

    # Step 4: Final high-precision polish
    if _START_TIME is not None and time.time() - _START_TIME < _DEADLINE - 8:
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                pts.flatten(),
                args=(200000,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12, 'disp': False, 'maxls': 100}
            )
            new_pts = res.x.reshape(N, 2)
            new_pts = _project_to_disk(new_pts)
            new_score = _score(new_pts)
            if new_score > score:
                score = new_score
                pts = new_pts
        except Exception:
            pass

    return pts, score


def _simulated_annealing_kick(pts, r_current, temperature=0.02, cooldown=0.995):
    """
    Structural change: use simulated annealing to escape local optima.
    """
    pts = pts.copy()
    current_score = r_current
    best_pts = pts.copy()
    best_score = current_score
    T = temperature

    for step in range(200):
        if _START_TIME is not None and time.time() - _START_TIME > _DEADLINE - 15:
            break

        # Generate a perturbed configuration
        scale = T * 0.5
        perturbed = pts + np.random.normal(0, scale, size=pts.shape)
        perturbed = _project_to_disk(perturbed)

        # Evaluate the perturbed configuration
        new_score = _score(perturbed)

        # Accept or reject based on simulated annealing criterion
        if new_score >= current_score:
            pts = perturbed
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = perturbed.copy()
        else:
            delta = current_score - new_score
            if np.random.random() < np.exp(-delta / max(T, 1e-10)):
                pts = perturbed
                current_score = new_score

        # Cool down
        T *= cooldown

    return best_pts, best_score


def construct():
    global _START_TIME
    _START_TIME = time.time()

    # Multiple diverse initializations
    inits = []
    inits.append(_hex_lattice_init())
    inits.append(_ring_init())
    for _ in range(5):
        inits.append(_random_init())

    best_pts = inits[0]
    best_score = _score(best_pts)

    # Sort inits by initial score to try best ones first
    init_scores = [(i, _score(init_pts)) for i, init_pts in enumerate(inits)]
    init_scores.sort(key=lambda x: x[1], reverse=True)

    for init_idx, (orig_idx, init_score) in enumerate(init_scores):
        if time.time() - _START_TIME > _DEADLINE - 25:
            break

        init_pts = inits[orig_idx]

        # Add perturbation for non-top inits to escape local minima
        if orig_idx > 0:
            perturbed = _perturb(init_pts, scale=0.012)
        else:
            perturbed = init_pts

        candidate_pts, candidate_score = _full_optimize(perturbed)

        if candidate_score > best_score:
            best_score = candidate_score
            best_pts = candidate_pts.copy()

    # Simulated annealing phase to escape local optima
    if time.time() - _START_TIME < _DEADLINE - 25:
        for sa_iter in range(4):
            if time.time() - _START_TIME > _DEADLINE - 25:
                break
            temperatures = [0.03, 0.02, 0.015, 0.01]
            T = temperatures[sa_iter % len(temperatures)]
            sa_pts, sa_score = _simulated_annealing_kick(best_pts, best_score, temperature=T)
            if sa_score > best_score:
                best_score = sa_score
                best_pts = sa_pts.copy()

    # Perturbation-retry loop: kick the best solution and re-optimize
    if time.time() - _START_TIME < _DEADLINE - 20:
        for kick_iter in range(6):
            if time.time() - _START_TIME > _DEADLINE - 20:
                break
            if kick_iter % 3 == 0:
                perturbed = _perturb(best_pts, scale=0.012)
            elif kick_iter % 3 == 1:
                perturbed = best_pts.copy()
                dists = np.linalg.norm(perturbed, axis=1)
                scales = 0.006 + 0.02 * dists
                perturbed += np.random.normal(0, 1, size=perturbed.shape) * scales[:, np.newaxis]
            else:
                perturbed = best_pts.copy()
                angle = np.random.uniform(0, 2 * np.pi)
                direction = np.array([np.cos(angle), np.sin(angle)])
                perturbed += np.random.normal(0, 0.008, size=perturbed.shape) + 0.004 * direction[np.newaxis, :]

            candidate_pts, candidate_score = _full_optimize(perturbed)
            if candidate_score > best_score:
                best_score = candidate_score
                best_pts = candidate_pts.copy()

    # Final high-beta polish on the best result
    if time.time() - _START_TIME < _DEADLINE - 8:
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                best_pts.flatten(),
                args=(500000,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-13, 'disp': False, 'maxls': 100}
            )
            new_pts = res.x.reshape(N, 2)
            new_pts = _project_to_disk(new_pts)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
        except Exception:
            pass

    # Gradient-based refinement as a final polish
    if time.time() - _START_TIME < _DEADLINE - 5:
        r_est = best_score
        for lr_try in [0.003, 0.002, 0.001]:
            if time.time() - _START_TIME > _DEADLINE - 3:
                break
            refined = _force_directed_relax(best_pts, r_est, steps=80, lr=lr_try)
            refined_score = _score(refined)
            if refined_score > best_score:
                best_score = refined_score
                best_pts = refined.copy()
            else:
                break

    return best_pts
