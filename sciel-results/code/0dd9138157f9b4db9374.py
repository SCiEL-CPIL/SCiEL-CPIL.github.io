"""Rewritten program: particle-equilibrium model with multi-stage optimization."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 16
D = 2


def _pairwise_dists(pts):
    """Compute all pairwise Euclidean distances for an (N,2) array."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Zero out diagonal
    np.fill_diagonal(dists, np.inf)
    return dists


def _score(pts):
    """Squared ratio of max to min pairwise distance. Lower is better."""
    dists = _pairwise_dists(pts)
    min_d = np.min(dists)
    max_d = np.max(dists)
    if min_d < 1e-12:
        return 1e10
    return (max_d / min_d) ** 2


def _soft_objective(params, beta=10.0):
    """
    Soft objective: penalize points that go outside unit circle and use
    a smooth approximation of the ratio.
    params: flat array of length 2*N (coordinates)
    """
    pts = params.reshape(N, 2)
    # Boundary penalty: push points into unit disk
    r = np.sqrt(np.sum(pts ** 2, axis=1))
    boundary_pen = np.sum(np.maximum(r - 1.0, 0.0) ** 2) * 100.0

    dists = _pairwise_dists(pts)
    min_d = np.min(dists)
    max_d = np.max(dists)
    if min_d < 1e-10:
        return 1e10 + boundary_pen
    ratio = max_d / min_d
    # Use log-soft or power for smoothness
    obj = (ratio ** 2) + boundary_pen
    return obj


def _hard_objective(params):
    """Hard objective for final polish."""
    pts = params.reshape(N, 2)
    return _score(pts)


def _ring_init(k, radius):
    """Place k points on a circle of given radius."""
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])


def _hex_init():
    """Hexagonal packing initialization for 16 points."""
    # 4x4 grid with hexagonal offset
    pts = []
    dx = 1.0
    dy = np.sqrt(3) / 2
    for i in range(4):
        for j in range(4):
            x = i * dx + (j % 2) * dx / 2
            y = j * dy
            pts.append([x, y])
    pts = np.array(pts[:16])
    # Center and normalize
    pts -= pts.mean(axis=0)
    max_r = np.max(np.sqrt(np.sum(pts ** 2, axis=1)))
    if max_r > 0:
        pts /= max_r
    return pts


def _random_init():
    """Random initialization within unit disk."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0.2, 1.0, N))
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts


def _optimize_from(pts, time_limit=10.0):
    """Multi-stage optimization starting from initial points."""
    params = pts.flatten()
    start = time.time()

    # Stage 1: L-BFGS with soft objective (boundary-aware)
    try:
        res = minimize(_soft_objective, params, method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12})
        if res.fun < 1e10:
            params = res.x
    except Exception:
        pass

    if time.time() - start > time_limit * 0.5:
        return params.reshape(N, 2)

    # Stage 2: Powell on hard objective
    try:
        res = minimize(_hard_objective, params, method='Powell',
                       options={'maxiter': 300, 'ftol': 1e-12, 'xtol': 1e-12})
        if res.fun < _score(params.reshape(N, 2)):
            params = res.x
    except Exception:
        pass

    if time.time() - start > time_limit * 0.8:
        return params.reshape(N, 2)

    # Stage 3: Final L-BFGS polish
    try:
        res = minimize(_hard_objective, params, method='L-BFGS-B',
                       options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-14})
        if res.fun < _score(params.reshape(N, 2)):
            params = res.x
    except Exception:
        pass

    return params.reshape(N, 2)


def construct():
    """Construct 16 points minimizing (max_dist/min_dist)^2."""
    deadline = time.time() + 140.0  # Safety margin before 150s kill

    best_pts = None
    best_score = np.inf

    # Multiple diverse initializations
    inits = []

    # Ring-based: two concentric rings
    for r1, r2 in [(0.4, 0.9), (0.3, 0.85), (0.5, 0.95), (0.35, 0.8)]:
        inner = _ring_init(8, r1)
        outer = _ring_init(8, r2)
        # Offset outer ring to interleave
        outer_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
        outer = np.column_stack([r2 * np.cos(outer_angles), r2 * np.sin(outer_angles)])
        pts = np.vstack([inner, outer])
        inits.append(pts)

    # Hexagonal
    inits.append(_hex_init())

    # Random starts
    for _ in range(5):
        inits.append(_random_init())

    # Single ring (all 16 on one circle) - baseline
    inits.append(_ring_init(16, 1.0))

    # Two rings with different splits
    for n1, n2 in [(4, 12), (6, 10), (8, 8), (10, 6), (12, 4)]:
        inner = _ring_init(n1, 0.4)
        outer = _ring_init(n2, 0.9)
        # Offset outer
        outer_angles = np.linspace(0, 2 * np.pi, n2, endpoint=False) + np.pi / n2
        outer = np.column_stack([0.9 * np.cos(outer_angles), 0.9 * np.sin(outer_angles)])
        pts = np.vstack([inner, outer])
        inits.append(pts)

    # Optimize each initialization
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 15.0:
            break

        time_per_init = min(12.0, (deadline - time.time()) / max(1, len(inits) - i))
        try:
            optimized = _optimize_from(init_pts, time_limit=time_per_init)
            s = _score(optimized)
            if s < best_score:
                best_score = s
                best_pts = optimized.copy()
        except Exception:
            continue

    # If we still have time, do a final differential evolution on best
    if best_pts is not None and time.time() < deadline - 20.0:
        try:
            bounds = [(-1.0, 1.0)] * (2 * N)
            de_result = differential_evolution(
                _hard_objective, bounds,
                maxiter=50, tol=1e-10, seed=42,
                polish=False, init='sobol'
            )
            de_pts = de_result.x.reshape(N, 2)
            de_score = _score(de_pts)
            if de_score < best_score:
                best_score = de_score
                best_pts = de_pts.copy()
                # Polish the DE result
                if time.time() < deadline - 10.0:
                    polished = _optimize_from(best_pts, time_limit=8.0)
                    p_score = _score(polished)
                    if p_score < best_score:
                        best_score = p_score
                        best_pts = polished.copy()
        except Exception:
            pass

    if best_pts is None:
        # Fallback: simple ring
        best_pts = _ring_init(16, 1.0)

    # Final safety check
    if time.time() > deadline - 5.0:
        # Ensure we return something valid
        pass

    # Verify distinctness
    dists = _pairwise_dists(best_pts)
    if np.min(dists) < 1e-8:
        # Add tiny perturbation to separate any coincident points
        best_pts = best_pts + np.random.normal(0, 1e-6, best_pts.shape)

    return best_pts
