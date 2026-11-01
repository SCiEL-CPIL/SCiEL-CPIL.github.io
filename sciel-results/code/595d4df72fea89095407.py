"""Packing 50 equal circles in a unit circle: analytic-gradient soft-min annealing + polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
SEED = 42
np.random.seed(SEED)

_START = time.time()
_DEADLINE = 140.0  # seconds; evaluator kills at 150s


def _elapsed():
    return time.time() - _START


def _remaining():
    return _DEADLINE - _elapsed()


def _score(points):
    """Compute the common radius for a set of points."""
    dists_from_origin = np.linalg.norm(points, axis=1)
    boundary_term = 1.0 - dists_from_origin
    if np.any(boundary_term < -1e-12):
        return 0.0
    n = len(points)
    diff = points[:, None, :] - points[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    r = min(min_pair, np.min(boundary_term))
    return max(r, 0.0)


def _soft_min_obj_and_grad(params, n, beta):
    """
    Objective: -soft_min(constraints) with analytic gradient.
    Constraints: d_ij / 2 for pairs, 1 - |c_i| for boundary.
    soft_min(x) = -1/beta * log(sum(exp(-beta * x_i)))
    """
    centers = params.reshape(n, 2)

    # Pair distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]

    # Boundary distances
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    bound_dists = 1.0 - norms

    # Combine all constraints
    vals = np.concatenate([pair_dists / 2.0, bound_dists])

    # Soft min with shift for stability
    m = np.min(vals)
    shifted = vals - m
    exp_terms = np.exp(-beta * shifted)
    S = np.sum(exp_terms)

    obj_val = -(m - (1.0 / beta) * np.log(S))

    # Gradient
    weights = exp_terms / S
    grad_centers = np.zeros_like(centers)

    # Pair contributions
    n_pairs = n * (n - 1) // 2
    pair_weights = weights[:n_pairs]
    mask = pair_weights > 1e-12
    if np.any(mask):
        i_m = i_idx[mask]
        j_m = j_idx[mask]
        w_m = pair_weights[mask]
        d_m = dists[i_m, j_m]
        d_safe = np.where(d_m > 1e-10, d_m, 1e-10)
        vecs = (centers[i_m] - centers[j_m]) / d_safe[:, np.newaxis]
        contrib = w_m[:, np.newaxis] * 0.5 * vecs
        np.add.at(grad_centers, i_m, -contrib)
        np.add.at(grad_centers, j_m, contrib)

    # Boundary contributions
    bound_weights = weights[n_pairs:]
    mask_b = bound_weights > 1e-12
    if np.any(mask_b):
        idx_b = np.where(mask_b)[0]
        w_b = bound_weights[idx_b]
        norm_b = norms[idx_b]
        norm_safe = np.where(norm_b > 1e-10, norm_b, 1e-10)
        vecs_b = centers[idx_b] / norm_safe[:, np.newaxis]
        contrib_b = w_b[:, np.newaxis] * vecs_b
        np.add.at(grad_centers, idx_b, contrib_b)

    return obj_val, grad_centers.flatten()


def _hex_lattice_init(n):
    """Generate n points from hexagonal lattice within unit disk."""
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= n:
            idx = np.argsort(radius[mask], kind='stable')
            return points[idx[:n]]
        spacing *= 0.98


def _ring_init(n):
    """Ring-based initialization approximating optimal distribution."""
    centers = [[0.0, 0.0]]
    rings = [(6, 0.22), (12, 0.44), (12, 0.66), (9, 0.82), (10, 0.92)]
    count = 1
    for count_in_ring, radius in rings:
        if count >= n:
            break
        angles = np.linspace(0, 2 * np.pi, count_in_ring, endpoint=False)
        for angle in angles:
            if count >= n:
                break
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])
            count += 1
    return np.array(centers[:n])


def _random_init(n, seed=42):
    """Random initialization with minimum separation."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(10000):
        if len(centers) >= n:
            break
        x = rng.uniform(-1, 1)
        y = rng.uniform(-1, 1)
        if x * x + y * y > 0.95 ** 2:
            continue
        if centers:
            pts = np.array(centers)
            dists = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
            if np.min(dists) < 0.15:
                continue
        centers.append([x, y])
    while len(centers) < n:
        x = rng.uniform(-0.9, 0.9)
        y = rng.uniform(-0.9, 0.9)
        centers.append([x, y])
    return np.array(centers[:n])


def _force_directed_precondition(points, iters=150):
    """Force-directed relaxation to resolve overlaps."""
    n = len(points)
    points = points.copy()
    r_est = _score(points)
    if r_est <= 0:
        r_est = 0.1

    dt = 0.001
    for it in range(iters):
        if _remaining() < 5.0:
            break
        forces = np.zeros_like(points)
        for i in range(n):
            for j in range(i + 1, n):
                diff = points[i] - points[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    diff = np.array([1e-10, 0.0])
                    dist = 1e-10
                if dist < 2 * r_est:
                    force_mag = (2 * r_est - dist) * 10.0
                    unit = diff / dist
                    forces[i] += force_mag * unit
                    forces[j] -= force_mag * unit
        for i in range(n):
            forces[i] -= 0.1 * points[i]
        points += dt * forces
        norms = np.linalg.norm(points, axis=1)
        for i in range(n):
            if norms[i] > 0.99:
                points[i] *= 0.99 / norms[i]
    return points


def _optimize_from_init(init_points, beta_schedule, max_iter=500):
    """Optimize using L-BFGS-B with analytic gradient and annealing."""
    n = len(init_points)
    points = init_points.copy()
    best_points = points.copy()
    best_r = _score(points)

    for beta in beta_schedule:
        if _remaining() < 10.0:
            break

        def loss_func(x):
            val, grad = _soft_min_obj_and_grad(x, n, beta)
            return val, grad

        bounds = [(-0.999, 0.999)] * (2 * n)

        result = minimize(loss_func, points.flatten(), jac=True,
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': max_iter, 'gtol': 1e-10})

        # Evaluate score at result
        result_pts = result.x.reshape(-1, 2)
        r_result = _score(result_pts)
        if r_result > best_r:
            best_r = r_result
            best_points = result_pts.copy()
            points = result_pts.copy()
        else:
            # Keep previous points if result is worse
            pass

    return best_points, best_r


def construct():
    if _elapsed() > 10:
        return _hex_lattice_init(N)

    inits = []

    # 1. Hexagonal lattice
    hex_pts = _hex_lattice_init(N)
    inits.append(hex_pts)

    # 2. Ring-based
    ring_pts = _ring_init(N)
    inits.append(ring_pts)

    # 3. Random with force-directed preconditioning
    for trial in range(4):
        if _remaining() < 20.0:
            break
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N)) * 0.95
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        pts = _force_directed_precondition(pts, iters=100)
        inits.append(pts)

    # Beta schedule for annealing
    beta_schedule = [50, 100, 500, 2000, 10000, 50000, 100000]

    best_points = None
    best_r = 0.0

    for init in inits:
        if _remaining() < 15.0:
            break
        pts, r = _optimize_from_init(init, beta_schedule, max_iter=300)
        if r > best_r:
            best_r = r
            best_points = pts

    # Final high-precision polish
    if best_points is not None and _remaining() > 5.0:
        final_beta = 100000

        def loss_func(x):
            val, grad = _soft_min_obj_and_grad(x, N, final_beta)
            return val, grad

        bounds = [(-0.999, 0.999)] * (2 * N)
        result = minimize(loss_func, best_points.flatten(), jac=True,
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 2000, 'gtol': 1e-12})

        polished = result.x.reshape(-1, 2)
        r_polished = _score(polished)
        if r_polished > best_r:
            best_points = polished
            best_r = r_polished

    if best_points is None:
        return _hex_lattice_init(N)

    return best_points
