"""Optimize 50 equal circle centres in the unit disk via ring-structure + force-directed + L-BFGS-B."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
START_TIME = None
DEADLINE = 140.0  # safety margin before 150s kill


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
        # Only consider pairs closer than 2*current_r (active constraints)
        current_r = _score(points)
        threshold = 2.5 * current_r if current_r > 0.01 else 0.4

        for i in range(n):
            for j in range(i + 1, n):
                if d[i, j] < threshold:
                    # Strong repulsive force (1/d^3 for steepness)
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
        # Ensure radii are positive and ordered
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

    # Phase 1: Force-directed (fast, good for escaping local minima)
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

    # Known optimal structure: 1 + 6 + 12 + 18 + 13 = 50
    # Try multiple ring configurations
    ring_configs = [
        ([1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.95]),
        ([1, 6, 12, 18, 13], [0.0, 0.22, 0.45, 0.68, 0.90]),
        ([1, 6, 12, 18, 13], [0.0, 0.28, 0.55, 0.80, 0.98]),
        ([1, 7, 12, 17, 13], [0.0, 0.25, 0.50, 0.75, 0.95]),
        ([1, 6, 13, 17, 13], [0.0, 0.25, 0.50, 0.75, 0.95]),
        ([1, 6, 12, 18, 13], [0.0, 0.20, 0.42, 0.65, 0.88]),
    ]

    # Also try ring structure optimization
    ring_opt_configs = [
        ([1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.95]),
        ([1, 6, 12, 18, 13], [0.0, 0.22, 0.45, 0.68, 0.90]),
    ]

    for config_idx, (ring_sizes, radii) in enumerate(ring_configs):
        if time.time() - START_TIME > DEADLINE - 20:
            break

        points = _ring_init(ring_sizes, radii)
        # Slight random perturbation to break symmetry
        if config_idx > 0:
            points += np.random.uniform(-0.01, 0.01, size=(N, 2))

        # Full optimization
        pts_opt, score_opt = _full_optimize(points, budget_seconds=min(25.0, DEADLINE - (time.time() - START_TIME) - 15))
        if score_opt > best_score:
            best_score = score_opt
            best_points = pts_opt.copy()

    # Ring structure optimization (optimize radii and offsets directly)
    for config_idx, (ring_sizes, radii) in enumerate(ring_opt_configs):
        if time.time() - START_TIME > DEADLINE - 15:
            break

        pts_ring, score_ring = _optimize_ring_structure(ring_sizes, radii, max_iter=2000)
        if score_ring > best_score:
            best_score = score_ring
            best_points = pts_ring.copy()

        # Refine with L-BFGS-B
        if time.time() - START_TIME > DEADLINE - 10:
            break
        pts_ref, score_ref = _lbfgs_refine(pts_ring, max_iter=1000)
        if score_ref > best_score:
            best_score = score_ref
            best_points = pts_ref.copy()

    # Final polish on best result
    if best_points is not None and time.time() - START_TIME < DEADLINE - 5:
        pts_final, score_final = _lbfgs_refine(best_points, max_iter=500)
        if score_final > best_score:
            best_points = pts_final

    if best_points is None:
        # Fallback
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
