"""Continuous optimization for packing 50 equal circles in a unit circle.
Uses adaptive soft-min surrogate with L-BFGS-B and symmetry-breaking."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
START_TIME = time.time()
TIME_LIMIT = 115.0  # safety margin before 120s kill


def _remaining_time():
    return max(0.0, TIME_LIMIT - (time.time() - START_TIME))


def _ring_init(n_points, n_rings=6, seed=42):
    """Initialize points in concentric rings for geometry-aware starting point."""
    rng = np.random.RandomState(seed)
    points = np.zeros((n_points, 2))
    # Place points on concentric rings
    ring_radii = np.linspace(0.1, 0.95, n_rings)
    pts_per_ring = [0] * n_rings
    # Distribute points: more on outer rings
    weights = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])[:n_rings]
    weights /= weights.sum()
    for i in range(n_rings):
        if i == n_rings - 1:
            pts_per_ring[i] = n_points - sum(pts_per_ring)
        else:
            pts_per_ring[i] = max(1, int(round(n_points * weights[i])))
    # Adjust to ensure exact count
    diff = n_points - sum(pts_per_ring)
    pts_per_ring[-1] += diff

    idx = 0
    for ring_idx in range(n_rings):
        n_in_ring = pts_per_ring[ring_idx]
        if n_in_ring <= 0:
            continue
        r = ring_radii[ring_idx]
        if n_in_ring == 1:
            points[idx, 0] = r
            points[idx, 1] = 0.0
            idx += 1
        else:
            angles = np.linspace(0, 2 * np.pi, n_in_ring, endpoint=False)
            # Add small random angular perturbation for symmetry breaking
            angles += rng.uniform(-0.1, 0.1, n_in_ring)
            points[idx:idx + n_in_ring, 0] = r * np.cos(angles)
            points[idx:idx + n_in_ring, 1] = r * np.sin(angles)
            idx += n_in_ring

    # Small random perturbation to break symmetry
    points += rng.uniform(-0.01, 0.01, points.shape)

    # Clamp to be strictly inside unit circle
    radii = np.linalg.norm(points, axis=1)
    mask = radii >= 0.99
    if np.any(mask):
        points[mask] *= 0.98 / radii[mask, None]

    return points


def _soft_min_score_and_grad(points, beta):
    """Compute soft-min score and gradient.
    Score = min(min_{i<j} |c_i - c_j|/2, min_i (1 - |c_i|))
    We maximize this, so we minimize -soft_min.
    """
    n = len(points)
    # Pairwise distances
    diff = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    # Half distances for pair constraints
    pair_dists = dist / 2.0

    # Container clearance
    radii = np.linalg.norm(points, axis=1)
    clearance = 1.0 - radii

    # Combine all constraints into one array
    all_vals = np.concatenate([pair_dists[np.triu_indices(n, k=1)], clearance])

    # Soft min via log-sum-exp: soft_min = -log(sum(exp(-beta * x))) / beta
    # For numerical stability, subtract max
    min_val = np.min(all_vals)
    # Clip to avoid issues
    all_vals_clipped = np.clip(all_vals, 1e-12, None)
    scaled = -beta * all_vals_clipped
    scaled -= np.max(scaled)  # for stability
    log_sum = np.log(np.sum(np.exp(scaled)))
    soft_min_val = -log_sum / beta

    # Gradient of soft_min w.r.t. points
    # d(soft_min)/dx = (1/beta) * sum(exp(-beta*x_i) * (-beta * dx_i)) / sum(exp(-beta*x_i))
    # = -sum(exp(-beta*x_i) * dx_i) / sum(exp(-beta*x_i))
    exp_vals = np.exp(scaled)
    weights = exp_vals / np.sum(exp_vals)

    grad = np.zeros_like(points)

    # Pair distance gradients
    # d(dist_ij)/dx_i = (x_i - x_j) / dist_ij
    # We need d(pair_dist)/dx_i where pair_dist = dist/2
    # d(pair_dist)/dx_i = (x_i - x_j) / (2 * dist)

    # Use vectorized approach
    # For each pair (i,j) with i<j, contribution to grad[i] and grad[j]
    # weight_ij = weights for that pair
    # grad[i] += weight_ij * (x_i - x_j) / dist_ij
    # grad[j] += weight_ij * (x_j - x_i) / dist_ij

    # Extract upper triangle weights
    iu = np.triu_indices(n, k=1)
    pair_weights = weights[:len(iu[0])]
    pair_dists_flat = pair_dists[iu]

    # Gradient contributions
    diff_u = points[iu[0]] - points[iu[1]]
    # d(pair_dist)/dx_i = diff / (2 * dist) = diff / (4 * pair_dist) since pair_dist = dist/2
    # Actually: dist = 2*pair_dist, so d(pair_dist)/dx_i = diff / (2 * dist) = diff / (4 * pair_dist)
    # Wait: pair_dist = dist/2, d(pair_dist)/dx_i = (1/2) * d(dist)/dx_i = (1/2) * diff/dist = diff/(2*dist)
    # And dist = 2*pair_dist, so d(pair_dist)/dx_i = diff/(2*2*pair_dist) = diff/(4*pair_dist)
    # Let me redo: d(dist)/dx_i = (x_i - x_j)/dist. d(pair_dist)/dx_i = (1/2)*(x_i-x_j)/dist = (x_i-x_j)/(2*dist)
    # Since dist = 2*pair_dist: d(pair_dist)/dx_i = (x_i-x_j)/(4*pair_dist)

    # For safety, use dist directly
    dist_flat = dist[iu]
    grad_pair = diff_u / (2.0 * dist_flat[:, None])  # d(pair_dist)/dx_i for i

    # Accumulate using np.add.at
    np.add.at(grad, iu[0], pair_weights[:, None] * grad_pair)
    np.add.at(grad, iu[1], -pair_weights[:, None] * grad_pair)

    # Container clearance gradients
    # d(1 - |x_i|)/dx_i = -x_i / |x_i|
    cont_weights = weights[len(iu[0]):]
    # Avoid division by zero
    safe_radii = np.where(radii > 1e-12, radii, 1.0)
    grad_cont = -points / safe_radii[:, None]
    grad += cont_weights[:, None] * grad_cont

    # We want to maximize soft_min, so objective = -soft_min
    # Gradient of objective = -gradient of soft_min
    obj = -soft_min_val
    grad_obj = -grad

    return obj, grad_obj


def _compute_score(points):
    """Compute the actual (non-smooth) score."""
    n = len(points)
    diff = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    radii = np.linalg.norm(points, axis=1)
    min_clear = np.min(1.0 - radii)
    return min(min_pair, min_clear)


def _is_valid(points):
    """Check if points are valid."""
    if not np.all(np.isfinite(points)):
        return False
    if points.shape != (N, 2):
        return False
    radii = np.linalg.norm(points, axis=1)
    if np.any(radii > 1.0 + 1e-10):
        return False
    return True


def construct():
    global START_TIME
    START_TIME = time.time()

    best_points = None
    best_score = -np.inf

    # Phase 1: Coarse optimization with low beta
    beta_schedule = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]

    current_points = _ring_init(N, n_rings=6, seed=42)

    for beta in beta_schedule:
        if _remaining_time() < 5.0:
            break

        # Try multiple random initializations for diversity
        n_starts = 3 if beta < 100 else 1

        for start_i in range(n_starts):
            if _remaining_time() < 5.0:
                break

            if start_i == 0:
                pts = current_points.copy()
            else:
                # Perturbed initialization
                seed = 42 + start_i * 100 + int(beta)
                pts = _ring_init(N, n_rings=6, seed=seed)

            def objective_and_grad(x):
                x = x.reshape(N, 2)
                val, g = _soft_min_score_and_grad(x, beta)
                return val, g.reshape(-1)

            try:
                result = minimize(
                    objective_and_grad,
                    current_points.flatten(),
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8,
                             'maxls': 50, 'maxfun': 5000}
                )
                if result.success or _remaining_time() < 3.0:
                    new_points = result.x.reshape(N, 2)
                    if _is_valid(new_points):
                        score = _compute_score(new_points)
                        if score > best_score:
                            best_score = score
                            best_points = new_points.copy()
                            current_points = new_points.copy()
            except Exception:
                pass

        # Update current_points to best found
        if best_points is not None:
            current_points = best_points.copy()

    # Phase 2: Final high-precision polish
    if best_points is not None and _remaining_time() > 3.0:
        for beta in [5000.0, 10000.0, 20000.0]:
            if _remaining_time() < 2.0:
                break

            def objective_and_grad(x):
                x = x.reshape(N, 2)
                val, g = _soft_min_score_and_grad(x, beta)
                return val, g.reshape(-1)

            try:
                result = minimize(
                    objective_and_grad,
                    best_points.flatten(),
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12,
                             'maxls': 100, 'maxfun': 10000}
                )
                if result.success:
                    new_points = result.x.reshape(N, 2)
                    if _is_valid(new_points):
                        score = _compute_score(new_points)
                        if score > best_score:
                            best_score = score
                            best_points = new_points.copy()
            except Exception:
                pass

    # Final validity check
    if best_points is None or not _is_valid(best_points):
        # Fallback: simple hexagonal grid
        best_points = _fallback_hex()

    return best_points


def _fallback_hex():
    """Fallback hexagonal lattice packing."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98
