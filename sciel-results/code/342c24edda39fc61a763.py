"""Pack 50 equal circles in the unit disk using ring-based combinatorial search + continuous polish."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 140.0  # safety margin before 150s kill


def _score_from_centers(c):
    """Compute the largest common radius for given centers."""
    n = len(c)
    if n != N:
        return 0.0
    if np.any(~np.isfinite(c)):
        return 0.0
    # Boundary constraint
    dists = np.linalg.norm(c, axis=1)
    r_boundary = np.min(1.0 - dists)
    if r_boundary <= 0:
        return 0.0
    # Pairwise distances
    r_pair = np.inf
    for i in range(n):
        d = np.linalg.norm(c[i + 1:] - c[i], axis=1)
        r_pair = min(r_pair, np.min(d) / 2.0)
    return min(r_boundary, r_pair)


def _neg_score(c):
    return -_score_from_centers(c)


def _ring_score(params, n_rings):
    """
    params: array of length n_rings*2 + 1 (last is center offset? No.)
    Actually: for each ring, we have (radius, angular_offset).
    Number of circles per ring is fixed by the combinatorial assignment.
    params layout: [r0, a0, r1, a1, ..., r_{k-1}, a_{k-1}]
    where ring i has count[i] circles at radius r_i with angles a_i + 2*pi*j/count[i].
    """
    counts = _ring_counts(n_rings)
    n = sum(counts)
    if n != N:
        return 0.0
    centers = []
    for i in range(n_rings):
        r = params[2 * i]
        a = params[2 * i + 1]
        cnt = counts[i]
        if cnt == 0:
            continue
        if r < 0:
            r = 0.0
        angles = a + 2 * np.pi * np.arange(cnt) / cnt
        xs = r * np.cos(angles)
        ys = r * np.sin(angles)
        centers.append(np.column_stack([xs, ys]))
    if not centers:
        return 0.0
    c = np.vstack(centers)
    return _score_from_centers(c)


def _ring_counts(n_rings):
    """
    Heuristic: distribute N circles among n_rings rings.
    Outer rings get more circles.
    Return list of counts.
    """
    # Try several distributions and pick one that sums to N
    # Simple: outer ring gets roughly proportional to its circumference
    # For n_rings rings, ring i (0=innermost) gets count proportional to (i+1)
    weights = np.arange(1, n_rings + 1, dtype=float)
    counts_float = N * weights / weights.sum()
    counts = np.round(counts_float).astype(int)
    # Adjust to sum to N
    diff = N - counts.sum()
    # Add/subtract from the largest ring
    idx = np.argmax(counts)
    counts[idx] += diff
    # Ensure all positive
    if np.any(counts < 1):
        # Fallback: put all in outer ring
        counts = np.zeros(n_rings, dtype=int)
        counts[-1] = N
    return counts.tolist()


def _optimize_ring(n_rings, x0, timeout=5.0):
    """Optimize ring parameters using scipy minimize."""
    counts = _ring_counts(n_rings)
    # Bounds: radius in [0, 1], angle in [0, 2*pi]
    bounds = []
    for i in range(n_rings):
        bounds.append((0.0, 1.0))  # radius
        bounds.append((0.0, 2 * np.pi))  # angle

    def obj(params):
        s = _ring_score(params, n_rings)
        return -s

    result = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
    best_params = x0.copy()
    best_score = _ring_score(x0, n_rings)
    if result.success and -result.fun > best_score:
        best_params = result.x.copy()
        best_score = -result.fun
    # Also try Nelder-Mead for local polish
    try:
        result2 = minimize(obj, best_params, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
        if -result2.fun > best_score:
            best_params = result2.x.copy()
            best_score = -result2.fun
    except Exception:
        pass
    return best_params, best_score


def _global_optimize(centers, timeout=5.0):
    """Polish centers with global L-BFGS-B."""
    def obj(c):
        return -_score_from_centers(c)

    try:
        result = minimize(obj, centers, method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.success and -result.fun > _score_from_centers(centers):
            return result.x
    except Exception:
        pass
    try:
        result2 = minimize(obj, centers, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
        if -result2.fun > _score_from_centers(centers):
            return result2.x
    except Exception:
        pass
    return centers


def construct():
    t_start = time.time()
    best_score = 0.0
    best_centers = None

    # Initial hexagonal lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            init_centers = points[np.argsort(radius, kind="stable")[:N]]
            break
        spacing *= 0.98

    best_score = _score_from_centers(init_centers)
    best_centers = init_centers.copy()

    # Try ring-based configurations
    for n_rings in range(2, 8):
        if time.time() - t_start > DEADLINE - 10:
            break
        counts = _ring_counts(n_rings)
        if sum(counts) != N:
            continue
        # Initial guess: radii spread from 0.1 to 0.9
        radii = np.linspace(0.1, 0.9, n_rings)
        angles = np.zeros(n_rings)
        x0 = np.zeros(2 * n_rings)
        for i in range(n_rings):
            x0[2 * i] = radii[i]
            x0[2 * i + 1] = angles[i]

        params, score = _optimize_ring(n_rings, x0, timeout=5.0)
        if score > best_score:
            # Reconstruct centers
            centers = []
            for i in range(n_rings):
                r = params[2 * i]
                a = params[2 * i + 1]
                cnt = counts[i]
                if cnt == 0:
                    continue
                if r < 0:
                    r = 0.0
                ang = a + 2 * np.pi * np.arange(cnt) / cnt
                xs = r * np.cos(ang)
                ys = r * np.sin(ang)
                centers.append(np.column_stack([xs, ys]))
            c = np.vstack(centers)
            best_score = score
            best_centers = c.copy()

    # Global polish on best centers
    if best_centers is not None and time.time() - t_start < DEADLINE - 5:
        polished = _global_optimize(best_centers, timeout=5.0)
        ps = _score_from_centers(polished)
        if ps > best_score:
            best_score = ps
            best_centers = polished

    # Multi-start global polish
    if time.time() - t_start < DEADLINE - 15:
        np.random.seed(42)
        for trial in range(5):
            if time.time() - t_start > DEADLINE - 10:
                break
            # Random initialization near best
            jitter = np.random.randn(N, 2) * 0.05
            c_try = best_centers + jitter
            # Clip to disk
            norms = np.linalg.norm(c_try, axis=1)
            mask = norms > 0.95
            c_try[mask] *= 0.95 / norms[mask, np.newaxis]
            polished = _global_optimize(c_try, timeout=3.0)
            ps = _score_from_centers(polished)
            if ps > best_score:
                best_score = ps
                best_centers = polished

    # Final high-precision polish
    if best_centers is not None and time.time() - t_start < DEADLINE - 2:
        def obj_final(c):
            return -_score_from_centers(c)
        try:
            res = minimize(obj_final, best_centers, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
            if -res.fun > best_score:
                best_centers = res.x
                best_score = -res.fun
        except Exception:
            pass

    return best_centers
