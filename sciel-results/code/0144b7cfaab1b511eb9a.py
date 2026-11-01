"""Packing 50 equal circles in unit circle: soft-min surrogate + gradient optimization + sub-gradient polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # Safety margin for 150s limit

# EVOLVE-BLOCK-START

def _compute_score(centers):
    """Compute the maximum radius r for which circles are disjoint and inside unit circle."""
    n = centers.shape[0]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    min_bound = np.min(1.0 - norms)
    return min(min_pair / 2.0, min_bound)


def _soft_min_obj_and_grad(params, n, beta):
    """
    Vectorized objective and gradient for soft-min surrogate.
    obj = -soft_min(constraints) where constraints are pair distances/2 and boundary distances.
    """
    centers = params.reshape(n, 2)

    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]

    # Boundary distances
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    bound_dists = 1.0 - norms

    # Combined constraint values
    vals = np.concatenate([pair_dists / 2.0, bound_dists])

    # Soft min with shift for numerical stability
    m = np.min(vals)
    shifted = vals - m  # >= 0
    exp_terms = np.exp(-beta * shifted)
    S = np.sum(exp_terms)
    weights = exp_terms / S

    obj_val = -(m - (1.0 / beta) * np.log(S))

    # Gradient computation (vectorized)
    grad = np.zeros_like(centers)

    # Pair contributions: grad_obj[c_i] -= w * 0.5 * (c_i - c_j)/d, grad_obj[c_j] += w * 0.5 * (c_i - c_j)/d
    pair_weights = weights[:n * (n - 1) // 2]
    mask = pair_weights > 1e-12
    if np.any(mask):
        idx = np.where(mask)[0]
        ii = i_idx[idx]
        jj = j_idx[idx]
        dd = pair_dists[idx]
        valid = dd > 1e-8
        if np.any(valid):
            ii_v = ii[valid]
            jj_v = jj[valid]
            dd_v = dd[valid]
            ww_v = pair_weights[idx[valid]]
            vecs = (centers[ii_v] - centers[jj_v]) / dd_v[:, np.newaxis]
            contrib = ww_v[:, np.newaxis] * vecs * 0.5
            np.add.at(grad, ii_v, -contrib)
            np.add.at(grad, jj_v, contrib)

    # Boundary contributions: grad_obj[c_i] += w * c_i / |c_i|
    bound_weights = weights[n * (n - 1) // 2:]
    mask_b = bound_weights > 1e-12
    if np.any(mask_b):
        idx_b = np.where(mask_b)[0]
        valid_b = norms[idx_b] > 1e-8
        if np.any(valid_b):
            idx_v = idx_b[valid_b]
            w_v = bound_weights[idx_v]
            norm_v = norms[idx_v]
            grad[idx_v] += (w_v / norm_v)[:, np.newaxis] * centers[idx_v]

    return obj_val, grad


def _force_directed_relax(centers, n_iter=200, k=0.5):
    """
    Simple force-directed relaxation to resolve overlaps.
    Pushes overlapping circles apart and pulls boundary-violating circles inward.
    """
    centers = centers.copy()
    n = centers.shape[0]
    i_idx, j_idx = np.triu_indices(n, k=1)

    for it in range(n_iter):
        # Compute all pair distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))

        # Desired minimum separation (estimate)
        r_est = 0.125
        min_sep = 2.0 * r_est

        # Forces from pair overlaps
        forces = np.zeros_like(centers)
        for idx in range(len(i_idx)):
            i, j = i_idx[idx], j_idx[idx]
            d = dists[i, j]
            if d < min_sep and d > 1e-10:
                # Repulsive force proportional to overlap
                overlap = min_sep - d
                vec = (centers[i] - centers[j]) / d
                f = k * overlap * vec
                forces[i] += f
                forces[j] -= f
            elif d <= 1e-10:
                # Coincident: random small push
                forces[i] += np.array([0.01, 0.01]) * (it % 2)
                forces[j] += np.array([-0.01, -0.01]) * (it % 2)

        # Boundary forces
        norms = np.sqrt(np.sum(centers * centers, axis=1))
        for i in range(n):
            if norms[i] > 1.0 - r_est:
                vec = centers[i] / norms[i]
                forces[i] -= k * (norms[i] - (1.0 - r_est)) * vec

        # Update with damping
        centers += 0.1 * forces

        # Clamp to container
        norms = np.sqrt(np.sum(centers * centers, axis=1))
        mask = norms > 0.99
        if np.any(mask):
            centers[mask] *= 0.99 / norms[mask]

    return centers


def _subgradient_polish(centers, start_time, max_iters=500, step_decay=0.999):
    """
    Direct sub-gradient ascent on the exact min objective.
    Maximizes r = min(pair_dists/2, boundary_dists) by moving in the direction
    of the active constraint's gradient.
    """
    current = centers.copy()
    n = current.shape[0]
    i_idx, j_idx = np.triu_indices(n, k=1)
    best_score = _compute_score(current)
    step = 0.002

    for it in range(max_iters):
        if time.time() - start_time > DEADLINE - 2:
            break

        # Compute all constraints
        diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        pair_dists = dists[i_idx, j_idx]
        norms = np.sqrt(np.sum(current * current, axis=1))
        bound_dists = 1.0 - norms

        vals = np.concatenate([pair_dists / 2.0, bound_dists])
        min_val = np.min(vals)
        active = np.where(np.abs(vals - min_val) < 1e-8)[0]

        # Compute sub-gradient: average of gradients of active constraints
        grad = np.zeros_like(current)
        n_pairs = n * (n - 1) // 2

        for a in active:
            if a < n_pairs:
                # Pair constraint
                i, j = i_idx[a], j_idx[a]
                d = dists[i, j]
                if d > 1e-10:
                    vec = (current[i] - current[j]) / d
                    grad[i] += 0.5 * vec
                    grad[j] -= 0.5 * vec
            else:
                # Boundary constraint
                i = a - n_pairs
                norm = norms[i]
                if norm > 1e-10:
                    vec = current[i] / norm
                    grad[i] -= vec  # gradient of (1-|c_i|) w.r.t c_i is -c_i/|c_i|

        # Normalize gradient
        gnorm = np.linalg.norm(grad)
        if gnorm > 1e-12:
            grad /= gnorm

        # Update
        current += step * grad

        # Clamp to container
        norms = np.sqrt(np.sum(current * current, axis=1))
        mask = norms > 0.999
        if np.any(mask):
            current[mask] *= 0.999 / norms[mask]

        # Check improvement
        r = _compute_score(current)
        if r > best_score:
            best_score = r
        else:
            step *= step_decay

    return current, best_score


def _get_ring_initialization(n):
    """Ring-based initialization matching known optimal structure: 1+6+12+18+13=50."""
    centers = [[0.0, 0.0]]
    ring_configs = [(6, 0.25), (12, 0.45), (18, 0.65), (13, 0.82)]
    for count, radius in ring_configs:
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        for angle in angles:
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])
    return np.array(centers[:n])


def _get_hex_lattice_initialization(n):
    """Hexagonal lattice initialization."""
    r_est = 0.125
    s = 2 * r_est
    points = []
    max_range = int(2.0 / s) + 4
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3) / 2.0
            if np.hypot(x, y) <= 0.97:
                points.append([x, y])
    if len(points) < n:
        return _get_ring_initialization(n)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    return points[np.argsort(dists)[:n]]


def _get_random_initialization(n, seed=42):
    """Random initialization inside unit circle."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 0.92)
        centers.append([radius * np.cos(angle), radius * np.sin(angle)])
    return np.array(centers)


def _optimize_from_start(centers, beta, maxiter=100, gtol=1e-8):
    """Run L-BFGS-B optimization from a starting point."""
    try:
        res = minimize(
            _soft_min_obj_and_grad,
            centers.flatten(),
            args=(N, beta),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'gtol': gtol}
        )
        if np.all(np.isfinite(res.x)):
            return res.x.reshape(N, 2)
    except Exception:
        pass
    return centers


def _polish_staged(centers, start_time):
    """Progressive beta increase for polishing."""
    current = centers.copy()
    best_score = _compute_score(current)

    betas = [50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]
    for beta in betas:
        if time.time() - start_time > DEADLINE - 15:
            break
        polished = _optimize_from_start(current, beta, maxiter=80, gtol=1e-9)
        if np.all(np.isfinite(polished)):
            r = _compute_score(polished)
            if r > best_score:
                best_score = r
                current = polished
    return current, best_score


def _local_search(centers, start_time, max_iters=15, step_size=0.015):
    """Local search with random perturbations and re-optimization."""
    current = centers.copy()
    current_score = _compute_score(current)
    rng = np.random.default_rng(777)

    for it in range(max_iters):
        if time.time() - start_time > DEADLINE - 20:
            break

        noisy = current + rng.normal(0, step_size, current.shape)
        norms = np.linalg.norm(noisy, axis=1)
        mask = norms > 0.98
        if np.any(mask):
            noisy = noisy.copy()
            noisy[mask] *= 0.98 / norms[mask]

        # Force-directed pre-relaxation
        relaxed = _force_directed_relax(noisy, n_iter=50, k=0.3)
        optimized = _optimize_from_start(relaxed, beta=200.0, maxiter=60, gtol=1e-8)

        if np.all(np.isfinite(optimized)):
            r = _compute_score(optimized)
            if r > current_score:
                current_score = r
                current = optimized.copy()

    return current, current_score


def construct():
    start_time = time.time()
    best_score = 0.0
    best_centers = None

    # Diverse initializations
    initializations = [
        _get_ring_initialization(N),
        _get_hex_lattice_initialization(N),
        _get_random_initialization(N, seed=42),
        _get_random_initialization(N, seed=123),
        _get_random_initialization(N, seed=999),
        _get_random_initialization(N, seed=555),
    ]

    # Phase 1: Force-directed relaxation + moderate-beta optimization for each init
    for init in initializations:
        if time.time() - start_time > DEADLINE - 60:
            break

        # Ensure inside container
        norms = np.linalg.norm(init, axis=1)
        mask = norms > 0.97
        if np.any(mask):
            init = init.copy()
            init[mask] *= 0.97 / norms[mask]

        # Force-directed pre-relaxation
        relaxed = _force_directed_relax(init, n_iter=100, k=0.5)

        # Moderate-beta optimization
        optimized = _optimize_from_start(relaxed, beta=50.0, maxiter=80, gtol=1e-7)
        if np.all(np.isfinite(optimized)):
            r = _compute_score(optimized)
            if r > best_score:
                best_score = r
                best_centers = optimized.copy()

    # Phase 2: Staged polish on best
    if best_centers is not None and time.time() - start_time < DEADLINE - 30:
        best_centers, best_score = _polish_staged(best_centers, start_time)

    # Phase 3: Local search with perturbations
    if best_centers is not None and time.time() - start_time < DEADLINE - 25:
        best_centers, best_score = _local_search(best_centers, start_time, max_iters=12, step_size=0.012)

    # Phase 4: Final high-precision polish with very high beta
    if best_centers is not None and time.time() - start_time < DEADLINE - 10:
        for beta in [10000.0, 20000.0, 50000.0]:
            if time.time() - start_time > DEADLINE - 5:
                break
            polished = _optimize_from_start(best_centers, beta, maxiter=100, gtol=1e-10)
            if np.all(np.isfinite(polished)):
                r = _compute_score(polished)
                if r > best_score:
                    best_score = r
                    best_centers = polished

    # Phase 5: Sub-gradient ascent on exact min objective
    if best_centers is not None and time.time() - start_time < DEADLINE - 3:
        best_centers, best_score = _subgradient_polish(best_centers, start_time, max_iters=300)

    if best_centers is None:
        return initializations[0]

    # Final validation: ensure all points are strictly inside
    norms = np.linalg.norm(best_centers, axis=1)
    mask = norms >= 1.0
    if np.any(mask):
        best_centers = best_centers.copy()
        best_centers[mask] *= 0.9999 / norms[mask]

    return best_centers

# EVOLVE-BLOCK-END
