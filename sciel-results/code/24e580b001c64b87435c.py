import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = time.time() + 142.0  # 8s safety margin before 150s kill


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0

    dists_origin = np.linalg.norm(pts, axis=1)
    if np.any(dists_origin > 1.0 + 1e-9):
        return 0.0

    r_cont = 1.0 - np.max(dists_origin)
    if r_cont < 0:
        return 0.0

    if N < 2:
        return r_cont

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)

    return min(r_cont, r_pair)


def _soft_min_obj_and_grad(flat, beta):
    """
    Compute the soft-min objective and gradient for L-BFGS-B.
    Objective: minimize f(x) = (1/beta) * ln(sum(exp(-beta * x_k)))
    where x_k are the critical distances (pairwise and boundary).
    """
    pts = flat.reshape(N, 2)

    dists_origin = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - dists_origin

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)

    iu = np.triu_indices(N, k=1)
    pair_dists = dist_mat[iu]

    all_dists = np.concatenate([boundary_dists, pair_dists / 2.0])

    y = -beta * all_dists
    max_y = np.max(y)
    y_shifted = y - max_y
    sum_exp = np.sum(np.exp(y_shifted))
    log_sum_exp = max_y + np.log(sum_exp)

    obj_val = log_sum_exp / beta

    weights = np.exp(y_shifted) / sum_exp

    grad_pts = np.zeros_like(pts)

    w_boundary = weights[0:N]
    safe_dists = np.where(dists_origin < 1e-8, 1.0, dists_origin)
    boundary_grad = (w_boundary / safe_dists)[:, np.newaxis] * pts
    grad_pts += boundary_grad

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


def _ring_init(n_rings=5):
    """
    Generate a ring-based initialization matching the known optimal structure:
    1 center + concentric rings of circles.
    """
    best_pts = None
    best_score = -1.0

    # Known optimal has roughly: 1 + 6 + 12 + 18 + 13 = 50
    ring_configs = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 17, 14],
        [1, 6, 13, 17, 13],
        [1, 7, 12, 17, 13],
        [1, 6, 12, 18, 12, 1],
    ]

    for config in ring_configs:
        pts_list = []
        # Center
        pts_list.append([0.0, 0.0])
        for ring_idx, count in enumerate(config[1:], start=1):
            radius = 0.25 * ring_idx
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            # Slight rotation offset for alternating rings
            if ring_idx % 2 == 0:
                angles += np.pi / count
            for a in angles:
                pts_list.append([radius * np.cos(a), radius * np.sin(a)])

        if len(pts_list) >= N:
            pts = np.array(pts_list[:N])
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts

    if best_pts is not None:
        return best_pts

    # Fallback: uniform random
    return _random_init()


def _random_init():
    """Random initialization with bias toward uniform area coverage."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.95, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _jittered_ring_init():
    """Ring init with random jitter to escape symmetry."""
    pts = _ring_init()
    # Add small random perturbations
    jitter = np.random.normal(0, 0.015, size=pts.shape)
    pts += jitter
    # Project back into unit disk
    dists = np.linalg.norm(pts, axis=1)
    mask = dists > 0.95
    if np.any(mask):
        pts[mask] = pts[mask] * 0.95 / dists[mask, np.newaxis]
    return pts


def _perturb(pts, scale=0.01):
    """Add small random perturbation to escape local optima."""
    return pts + np.random.normal(0, scale, size=pts.shape)


def _optimize_from_start(pts, beta_schedule, max_iter_per_beta=300):
    """Run L-BFGS-B with a schedule of increasing beta values."""
    current_pts = pts.copy()
    best_pts = current_pts.copy()
    best_score = _score(current_pts)

    for beta in beta_schedule:
        if time.time() > _DEADLINE - 15.0:
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

            dists = np.linalg.norm(new_pts, axis=1)
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


def _force_directed_refine(pts, r_target, steps=300, lr=0.002, repulsive_power=4):
    """
    Force-directed layout with inverse-distance^repulsive_power repulsion.
    Combined with hard constraint projection for feasibility.
    """
    pts = pts.copy()
    for step in range(steps):
        if time.time() > _DEADLINE - 10.0:
            break

        force = np.zeros_like(pts)

        # Pairwise repulsive forces: F ~ 1/d^repulsive_power
        iu = np.triu_indices(N, k=1)
        i_idx, m_idx = iu
        vecs = pts[i_idx] - pts[m_idx]
        d_pairs = np.linalg.norm(vecs, axis=1)
        safe_d = np.where(d_pairs < 1e-10, 1e-10, d_pairs)
        unit_vecs = vecs / safe_d[:, np.newaxis]

        # Repulsion strength: stronger when closer than 2*r_target
        # Use a smooth potential: F = k * (2*r_target / d)^repulsive_power
        ratio = 2 * r_target / safe_d
        # Only apply strong repulsion when overlapping or close
        mask_close = safe_d < 2.5 * r_target
        repulsion = np.zeros_like(safe_d)
        repulsion[mask_close] = ratio[mask_close] ** repulsive_power

        force_i = unit_vecs * repulsion[:, np.newaxis]
        force_m = -unit_vecs * repulsion[:, np.newaxis]

        np.add.at(force, i_idx, force_i)
        np.add.at(force, m_idx, force_m)

        # Boundary attraction: pull toward center if too close to edge
        dists_origin = np.linalg.norm(pts, axis=1)
        boundary_dists = 1.0 - dists_origin
        mask_boundary = boundary_dists < r_target
        if np.any(mask_boundary):
            safe_dist = np.where(dists_origin[mask_boundary] < 1e-10, 1.0, dists_origin[mask_boundary])
            unit_out = pts[mask_boundary] / safe_dist[:, np.newaxis]
            # Pull inward proportional to violation
            pull = (r_target - boundary_dists[mask_boundary])[:, np.newaxis] * unit_out
            force[mask_boundary] -= pull

        # Adaptive learning rate: reduce as we converge
        effective_lr = lr * (1.0 - step / steps * 0.5)
        pts += effective_lr * force

        # Hard constraint projection: keep inside unit disk
        dists = np.linalg.norm(pts, axis=1)
        mask_out = dists > 1.0 - r_target
        if np.any(mask_out):
            pts[mask_out] = pts[mask_out] * (1.0 - r_target) / dists[mask_out, np.newaxis]

    return pts


def construct():
    start_time = time.time()

    # Multiple diverse initializations
    inits = []
    inits.append(_hex_lattice_init())
    inits.append(_ring_init())
    inits.append(_jittered_ring_init())
    for _ in range(6):
        inits.append(_random_init())

    best_pts = inits[0]
    best_score = _score(best_pts)

    # Beta schedules for annealing - more aggressive for later stages
    beta_schedule_1 = [10, 30, 100, 300, 1000, 3000]
    beta_schedule_2 = [50, 200, 1000, 3000, 10000]

    # Optimize from each initialization
    for init_idx, init_pts in enumerate(inits):
        if time.time() > _DEADLINE - 25.0:
            break

        # Use different beta schedules for different inits to diversify search
        if init_idx % 2 == 0:
            schedule = beta_schedule_1
        else:
            schedule = beta_schedule_2

        # Add perturbation for non-hex/ring inits to escape local minima
        if init_idx > 0:
            perturbed = _perturb(init_pts, scale=0.015)
            candidate_pts, candidate_score = _optimize_from_start(perturbed, schedule, max_iter_per_beta=250)
        else:
            candidate_pts, candidate_score = _optimize_from_start(init_pts, schedule, max_iter_per_beta=400)

        if candidate_score > best_score:
            best_score = candidate_score
            best_pts = candidate_pts.copy()

    # Final high-beta polish on the best result with tighter tolerances
    if time.time() < _DEADLINE - 20.0:
        for beta in [5000, 20000, 50000, 100000]:
            if time.time() > _DEADLINE - 12.0:
                break
            try:
                res = minimize(
                    _soft_min_obj_and_grad,
                    best_pts.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True,
                    bounds=[(-1.0, 1.0)] * (2 * N),
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'disp': False}
                )
                new_pts = res.x.reshape(N, 2)
                dists = np.linalg.norm(new_pts, axis=1)
                mask = dists > 1.0
                if np.any(mask):
                    new_pts[mask] = new_pts[mask] / dists[mask, np.newaxis]

                new_score = _score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts

            except Exception:
                pass

    # Force-directed refinement with inverse-distance^4 repulsion
    if time.time() < _DEADLINE - 8.0:
        r_est = best_score
        refined = _force_directed_refine(best_pts, r_est, steps=200, lr=0.002, repulsive_power=4)
        refined_score = _score(refined)
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined

    # Final ultra-tight L-BFGS polish
    if time.time() < _DEADLINE - 3.0:
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                best_pts.flatten(),
                args=(100000,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-1.0, 1.0)] * (2 * N),
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-13, 'disp': False}
            )
            new_pts = res.x.reshape(N, 2)
            dists = np.linalg.norm(new_pts, axis=1)
            mask = dists > 1.0
            if np.any(mask):
                new_pts[mask] = new_pts[mask] / dists[mask, np.newaxis]

            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts

        except Exception:
            pass

    return best_pts
