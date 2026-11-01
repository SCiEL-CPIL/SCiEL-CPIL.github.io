"""Optimize 50 equal circle centres in the unit disk via ring-structure + force-directed + L-BFGS-B."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
START_TIME = None
DEADLINE = 135.0  # safety margin before 150s kill


def _score(points):
    """Compute packing radius: min(min pairwise dist/2, min (1-|c_i|))."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists = np.linalg.norm(points, axis=1)
    if np.any(dists >= 1.0):
        return 0.0
    boundary_margin = np.min(1.0 - dists)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist_mat, np.inf)
    min_pair_dist = np.min(dist_mat)
    return min(min_pair_dist / 2.0, boundary_margin)


def _ring_init(ring_sizes, ring_radii, angle_offsets=None):
    """Generate points from concentric rings."""
    points = np.zeros((N, 2))
    idx = 0
    if angle_offsets is None:
        angle_offsets = [0.0] * len(ring_sizes)
    for ring_idx, (size, radius) in enumerate(zip(ring_sizes, ring_radii)):
        offset = angle_offsets[ring_idx] if ring_idx < len(angle_offsets) else 0.0
        for j in range(size):
            angle = 2 * np.pi * j / size + offset
            points[idx] = [radius * np.cos(angle), radius * np.sin(angle)]
            idx += 1
    return points


def _hex_lattice_init(r_guess):
    """Hexagonal lattice initialization."""
    spacing = 2.0 * r_guess
    pts = []
    k = int(1.0 / spacing) + 3
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= (1.0 - r_guess) ** 2:
                pts.append([x, y])
    pts = np.array(pts)
    if len(pts) >= N:
        radii = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(radii, kind='stable')[:N]
        return pts[idx]
    else:
        return _ring_init([1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.95])


def _force_directed(points, max_iters=2000, dt=0.005):
    """Force-directed optimization with strong repulsion for close pairs."""
    points = points.copy()
    n = points.shape[0]
    best_score = _score(points)
    best_points = points.copy()

    for it in range(max_iters):
        if time.time() - START_TIME > DEADLINE:
            break

        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        d2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(d2, np.inf)
        d = np.sqrt(d2)

        # Forces: strong repulsion for close pairs
        forces = np.zeros_like(points)
        current_r = _score(points)
        threshold = 2.5 * current_r if current_r > 0.01 else 0.4

        for i in range(n):
            for j in range(i + 1, n):
                if d[i, j] < threshold:
                    f = diff[i, j] / (d[i, j] ** 3 + 1e-8)
                    forces[i] += f
                    forces[j] -= f

        # Boundary constraint: push inward if too close to boundary
        dists = np.linalg.norm(points, axis=1)
        for i in range(n):
            if dists[i] > 0.95:
                forces[i] -= 0.5 * points[i] / dists[i]

        # Update
        points += dt * forces

        # Project inside unit circle
        dists = np.linalg.norm(points, axis=1)
        for i in range(n):
            if dists[i] > 0.99:
                points[i] *= 0.99 / dists[i]

        score = _score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()

        # Decay step size
        dt *= 0.999

    return best_points, best_score


def _soft_max_obj_grad(pts, r_target, soft_beta):
    """
    Soft-max constraint violation objective with vectorized gradient.
    f(pts) = soft_max([|c_i|-(1-r), 2r-d_ij])
    """
    n = len(pts)
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    cont_viol = norms - (1.0 - r_target)

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_viol = 2.0 * r_target - dists
    iu = np.triu_indices(n, k=1)
    pair_viol_flat = pair_viol[iu]

    all_viol = np.concatenate([cont_viol, pair_viol_flat])

    max_v = np.max(all_viol)
    if max_v > 50.0 / soft_beta:
        return max_v, np.zeros_like(pts)

    exp_v = np.exp(soft_beta * (all_viol - max_v))
    sum_exp = np.sum(exp_v)
    soft_max_val = max_v + np.log(sum_exp) / soft_beta
    weights = exp_v / sum_exp

    grad = np.zeros_like(pts)

    # Container gradient (vectorized)
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    unit_vecs = pts / safe_norms[:, np.newaxis]
    grad += weights[:n, np.newaxis] * unit_vecs

    # Pairwise gradient (vectorized via np.add.at)
    iw = weights[n:]
    valid = dists[iu] > 1e-10
    if np.any(valid):
        iw_valid = iw[valid]
        iu_valid = (iu[0][valid], iu[1][valid])
        d_valid = dists[iu_valid]
        diff_valid = pts[iu_valid[0]] - pts[iu_valid[1]]
        unit_diff = diff_valid / d_valid[:, np.newaxis]

        np.add.at(grad, iu_valid[0], -iw_valid[:, np.newaxis] * unit_diff)
        np.add.at(grad, iu_valid[1], iw_valid[:, np.newaxis] * unit_diff)

    return soft_max_val, grad


def _optimize_for_radius(pts_init, r, soft_beta, max_iter=2000, tol=1e-10):
    """Minimize soft constraint violation for fixed r using L-BFGS-B."""
    n = pts_init.shape[0]

    def objective(x):
        pts = x.reshape(n, 2)
        val, grad = _soft_max_obj_grad(pts, r, soft_beta)
        return val, grad.flatten()

    x0 = pts_init.flatten()
    result = minimize(objective, x0, method='L-BFGS-B', jac=True,
                      options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8,
                               'maxls': 50})
    pts_opt = result.x.reshape(n, 2)
    return pts_opt, result.fun


def _adaptive_beta_optimize(pts_init, r, start_beta=20.0, end_beta=5000.0,
                            n_stages=4, max_iter_per_stage=1500, tol=1e-10):
    """Optimize with adaptive increasing beta across stages."""
    current_pts = pts_init.copy()
    current_val = np.inf
    for stage in range(n_stages):
        if time.time() - START_TIME > DEADLINE - 5:
            break
        frac = stage / max(n_stages - 1, 1)
        beta = start_beta * (end_beta / start_beta) ** frac
        max_iter = max_iter_per_stage if stage < n_stages - 1 else max_iter_per_stage * 2
        current_pts, current_val = _optimize_for_radius(current_pts, r, beta,
                                                         max_iter=max_iter, tol=tol)
    return current_pts, current_val


def _lbfgs_refine(points, max_iter=2000):
    """L-BFGS-B refinement of the packing."""
    def objective(flat):
        pts = flat.reshape(N, 2)
        return -_score(pts)

    x0 = points.flatten()
    result = minimize(objective, x0, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8,
                               'maxls': 50})
    return result.x.reshape(N, 2), -result.fun


def _optimize_ring_structure(ring_sizes, initial_radii, max_iter=3000):
    """Optimize ring radii and angle offsets using L-BFGS-B on the ring parameters."""
    n_rings = len(ring_sizes)

    def ring_objective(params):
        radii = params[:n_rings]
        offsets = params[n_rings:]
        radii = np.abs(radii)
        points = _ring_init(ring_sizes, radii, offsets)
        return -_score(points)

    x0 = np.concatenate([initial_radii, np.zeros(n_rings)])
    result = minimize(ring_objective, x0, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10,
                               'maxls': 100})
    radii = np.abs(result.x[:n_rings])
    offsets = result.x[n_rings:]
    points = _ring_init(ring_sizes, radii, offsets)
    return points, -result.fun


def _full_optimize(points, budget_seconds=30.0):
    """Full optimization pipeline: force-directed + L-BFGS-B."""
    t0 = time.time()
    best_points = points.copy()
    best_score = _score(points)

    # Phase 1: Force-directed
    if time.time() - t0 < budget_seconds * 0.4:
        pts_f, score_f = _force_directed(points, max_iters=1500, dt=0.004)
        if score_f > best_score:
            best_score = score_f
            best_points = pts_f.copy()

    # Phase 2: L-BFGS-B refinement
    if time.time() - t0 < budget_seconds * 0.8:
        pts_lb, score_lb = _lbfgs_refine(best_points, max_iter=1500)
        if score_lb > best_score:
            best_score = score_lb
            best_points = pts_lb.copy()

    # Phase 3: Final fine-tune
    if time.time() - t0 < budget_seconds * 0.95:
        pts_final, score_final = _lbfgs_refine(best_points, max_iter=500)
        if score_final > best_score:
            best_score = score_final
            best_points = pts_final.copy()

    return best_points, best_score


def construct():
    global START_TIME
    START_TIME = time.time()

    best_points = None
    best_score = 0.0

    rng = np.random.RandomState(42)

    # Known optimal structure: 1 + 6 + 12 + 18 + 13 = 50
    ring_configs = [
        ([1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.95]),
        ([1, 6, 12, 18, 13], [0.0, 0.22, 0.45, 0.68, 0.90]),
        ([1, 6, 12, 18, 13], [0.0, 0.28, 0.55, 0.80, 0.98]),
        ([1, 7, 12, 17, 13], [0.0, 0.25, 0.50, 0.75, 0.95]),
        ([1, 6, 13, 17, 13], [0.0, 0.25, 0.50, 0.75, 0.95]),
        ([1, 6, 12, 18, 13], [0.0, 0.20, 0.42, 0.65, 0.88]),
    ]

    # Also try hex lattice init
    hex_pts = _hex_lattice_init(0.12)
    hex_score = _score(hex_pts)
    if hex_score > best_score:
        best_score = hex_score
        best_points = hex_pts.copy()

    # Try ring configs with full optimization
    for config_idx, (ring_sizes, radii) in enumerate(ring_configs):
        if time.time() - START_TIME > DEADLINE - 25:
            break

        points = _ring_init(ring_sizes, radii)
        if config_idx > 0:
            points += rng.uniform(-0.01, 0.01, size=(N, 2))

        pts_opt, score_opt = _full_optimize(points, budget_seconds=min(20.0, DEADLINE - (time.time() - START_TIME) - 20))
        if score_opt > best_score:
            best_score = score_opt
            best_points = pts_opt.copy()

    # Binary search on radius with soft-max optimization
    r_low = 0.10
    r_high = 0.145

    for _ in range(4):
        if time.time() - START_TIME > DEADLINE - 35:
            break
        r_mid = (r_low + r_high) / 2.0
        pts_init = _ring_init([1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.95])

        found_feasible = False
        for beta in [50.0, 200.0, 1000.0]:
            if time.time() - START_TIME > DEADLINE - 15:
                break
            pts_opt, val = _optimize_for_radius(pts_init, r_mid, beta, max_iter=1000)
            score = _score(pts_opt)
            if score >= r_mid - 1e-8:
                found_feasible = True
                if score > best_score:
                    best_score = score
                    best_points = pts_opt.copy()
                break

        if found_feasible:
            r_low = r_mid
        else:
            r_high = r_mid

        if r_high - r_low < 1e-5:
            break

    # Multi-start with symmetry breaking and adaptive beta
    n_starts = 3
    for s in range(n_starts):
        if time.time() - START_TIME > DEADLINE - 25:
            break
        r_target = max(r_low, 0.12)
        if s == 0:
            pts_init = best_points.copy()
        elif s == 1:
            pts_init = _hex_lattice_init(r_target)
        else:
            pts_init = best_points.copy()
            noise = rng.normal(0, 0.03 * (s + 1), size=pts_init.shape)
            pts_init += noise
            norms = np.hypot(pts_init[:, 0], pts_init[:, 1])
            for i in range(N):
                if norms[i] > 1.0 - r_target:
                    pts_init[i] *= (1.0 - r_target) / norms[i]

        pts_opt, val = _adaptive_beta_optimize(pts_init, r_target,
                                                start_beta=20.0, end_beta=5000.0,
                                                n_stages=3, max_iter_per_stage=1000)
        score = _score(pts_opt)
        if score > best_score:
            best_score = score
            best_points = pts_opt.copy()

    # Final high-precision polish with very high beta
    if best_points is not None and time.time() - START_TIME < DEADLINE - 5:
        r_current = _score(best_points)
        if r_current > 0.05:
            pts_polish, _ = _optimize_for_radius(best_points, r_current, soft_beta=50000.0,
                                                  max_iter=500, tol=1e-12)
            score_polish = _score(pts_polish)
            if score_polish > best_score:
                best_score = score_polish
                best_points = pts_polish.copy()

    if best_points is None:
        best_points = _ring_init([1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.95])

    # Ensure validity
    dists = np.linalg.norm(best_points, axis=1)
    for i in range(N):
        if dists[i] >= 1.0:
            best_points[i] *= 0.99 / max(dists[i], 1e-10)

    return best_points


# EVOLVE-BLOCK-START
# The main construct function is defined above
# EVOLVE-BLOCK-END
