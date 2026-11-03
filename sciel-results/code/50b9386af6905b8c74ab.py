"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radius(pts):
    """Packing radius = min(min_pair_dist/2, min_boundary)."""
    if len(pts) < 2:
        return 0.0
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    min_boundary = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
    return min(min_pair_dist / 2.0, min_boundary)


def _init_hex(spacing):
    """Generate hexagonal lattice points within the unit disk."""
    k = int(1.0 / spacing) + 3
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= 1.0 - spacing / 2.0:
                points.append((x, y))
    return np.array(points) if points else np.empty((0, 2))


def _hertzian_dynamics(pts, max_steps=6000, dt=0.004, k_repel=400.0, k_boundary=200.0,
                       damping=0.995, threshold=0.16, target_margin=0.02,
                       deadline=None):
    """
    Rigid-body dynamics with a smooth Hertzian repulsive potential (F ~ overlap^4)
    that vanishes at the contact distance, evolved via velocity-Verlet with damping.
    The potential energy is smooth and differentiable, so the force field is well
    behaved and symplectic integration keeps the energy stable (no hard-constraint
    stiffness). Particles relax to a non-overlapping equilibrium.
    """
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    n = len(best_pts)

    pos = pts.copy()
    v = np.zeros_like(pos)

    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break

        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)

        mask = dists < threshold
        np.fill_diagonal(mask, False)
        overlap = np.where(mask, threshold - dists, 0.0)
        force_mag = k_repel * (overlap ** 4)

        safe_dists = np.where(dists > 1e-12, dists, 1.0)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        forces = np.sum(force_mag[:, :, np.newaxis] * unit_vecs, axis=1)

        # Smooth boundary repulsion (quadratic near the edge)
        norms = np.hypot(pos[:, 0], pos[:, 1])
        margin = 1.0 - norms
        boundary_active = margin < target_margin
        safe_norms = np.where(norms > 1e-12, norms, 1.0)
        inward_dir = -pos / safe_norms[:, np.newaxis]
        bf = np.where(boundary_active,
                      k_boundary * ((target_margin - margin) / target_margin) ** 2,
                      0.0)
        forces += inward_dir * bf[:, np.newaxis]

        # velocity-Verlet with damping
        v = damping * v + dt * forces
        pos = pos + dt * v

        # Hard boundary clamp (safety net only)
        norms_cand = np.hypot(pos[:, 0], pos[:, 1])
        too_far = norms_cand > 0.9999
        if np.any(too_far):
            pos[too_far] *= 0.9999 / norms_cand[too_far]
            v[too_far] *= 0.3

        cand_r = _compute_radius(pos)
        if cand_r > best_r:
            best_r = cand_r
            best_pts = pos.copy()

        force_norm = np.hypot(forces[:, 0], forces[:, 1])
        if np.max(force_norm) < 1e-9:
            break

    return best_pts, best_r


def _soft_min_optimize(pts, deadline, eps_start=30.0, eps_end=400.0,
                       lr_start=0.003, lr_end=1e-5, max_iter=4000):
    """
    Gradient ascent on a smooth log-sum-exp surrogate of the nonsmooth min objective.
    Maximizes a soft-min of all pair distances and boundary margins, which is
    differentiable everywhere, so we can do gradient steps that steadily lift the
    weakest constraints (the true bottleneck of the packing).
    """
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    n = len(best_pts)

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        progress = it / max_iter
        eps = eps_start * (eps_end / eps_start) ** progress
        lr = lr_start * (lr_end / lr_start) ** progress

        diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
        b_dists = 1.0 - radii

        # Pairwise gradient of soft-min (maximize): push apart the closest pairs
        weights_pair = np.exp(-eps * dists)
        np.fill_diagonal(weights_pair, 0.0)
        safe_dists = np.where(dists > 1e-12, dists, 1.0)
        unit_vecs = diff / safe_dists[:, :, np.newaxis]
        grad_pair = eps * np.sum(weights_pair[:, :, np.newaxis] * unit_vecs, axis=1)

        # Boundary gradient: push outward where close to edge
        weights_bound = np.exp(-eps * b_dists)
        safe_norms = np.where(radii > 1e-12, radii, 1.0)
        unit_radii = best_pts / safe_norms[:, np.newaxis]
        grad_bound = -eps * weights_bound[:, np.newaxis] * unit_radii

        total_grad = grad_pair + grad_bound
        grad_norm = np.hypot(total_grad[:, 0], total_grad[:, 1])
        grad_norm[grad_norm < 1e-12] = 1.0
        total_grad /= grad_norm[:, np.newaxis]

        candidate = best_pts + lr * total_grad
        norms_cand = np.hypot(candidate[:, 0], candidate[:, 1])
        too_far = norms_cand > 0.9999
        if np.any(too_far):
            candidate[too_far] *= 0.9999 / norms_cand[too_far]

        cand_r = _compute_radius(candidate)
        if cand_r > best_r:
            best_r = cand_r
            best_pts = candidate.copy()
        else:
            lr *= 0.98

    return best_pts, best_r


def _polish(pts, deadline):
    """Final high-precision polish: very small steps, high-eps soft-min."""
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)

    for lr in [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]:
        if deadline is not None and time.time() > deadline:
            break
        for it in range(250):
            if deadline is not None and time.time() > deadline:
                break
            eps = 250.0
            diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
            b_dists = 1.0 - radii

            weights_pair = np.exp(-eps * dists)
            np.fill_diagonal(weights_pair, 0.0)
            safe_dists = np.where(dists > 1e-12, dists, 1.0)
            unit_vecs = diff / safe_dists[:, :, np.newaxis]
            grad_pair = eps * np.sum(weights_pair[:, :, np.newaxis] * unit_vecs, axis=1)

            weights_bound = np.exp(-eps * b_dists)
            safe_norms = np.where(radii > 1e-12, radii, 1.0)
            unit_radii = best_pts / safe_norms[:, np.newaxis]
            grad_bound = -eps * weights_bound[:, np.newaxis] * unit_radii

            total_grad = grad_pair + grad_bound
            gn = np.hypot(total_grad[:, 0], total_grad[:, 1])
            gn[gn < 1e-12] = 1.0
            total_grad /= gn[:, np.newaxis]

            candidate = best_pts + lr * total_grad
            norms_cand = np.hypot(candidate[:, 0], candidate[:, 1])
            too_far = norms_cand > 0.9999
            if np.any(too_far):
                candidate[too_far] *= 0.9999 / norms_cand[too_far]

            cand_r = _compute_radius(candidate)
            if cand_r > best_r:
                best_r = cand_r
                best_pts = candidate.copy()

    return best_pts, best_r


def construct():
    deadline = time.time() + 142.0  # safety margin under the 150s kill

    best_pts = None
    best_r = 0.0

    # 1) Hex-lattice seeding over a range of spacings; keep the best N-point set.
    for spacing in np.linspace(0.09, 0.13, 30):
        if time.time() > deadline:
            break
        pts = _init_hex(spacing)
        if len(pts) < N:
            continue
        dists = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(dists, kind='stable')[:N]
        candidate = pts[idx]
        r = _compute_radius(candidate)
        if r > best_r:
            best_r = r
            best_pts = candidate.copy()

    if best_pts is None:
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            keep = radius <= 1.0 - spacing / 2.0
            points, radius = points[keep], radius[keep]
            if len(points) >= N:
                best_pts = points[np.argsort(radius, kind='stable')[:N]]
                best_r = _compute_radius(best_pts)
                break
            spacing *= 0.98

    # 2) Hertzian dynamics: smooth repulsive potential, symplectic relaxation.
    if time.time() < deadline - 20.0:
        best_pts, best_r = _hertzian_dynamics(
            best_pts, max_steps=6000, dt=0.004, k_repel=400.0, k_boundary=200.0,
            damping=0.995, threshold=0.16, target_margin=0.02,
            deadline=deadline - 20.0)

    # 3) Soft-min (log-sum-exp) gradient ascent on the smooth surrogate.
    if time.time() < deadline - 12.0:
        best_pts, best_r = _soft_min_optimize(
            best_pts, deadline=deadline - 12.0,
            eps_start=30.0, eps_end=400.0, lr_start=0.003, lr_end=1e-5, max_iter=4000)

    # 4) Final high-precision polish.
    if time.time() < deadline - 2.0:
        best_pts, best_r = _polish(best_pts, deadline=deadline - 2.0)

    return best_pts
# EVOLVE-BLOCK-END
