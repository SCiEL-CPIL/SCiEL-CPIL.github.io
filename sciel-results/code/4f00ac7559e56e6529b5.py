import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

N = 14
_DEADLINE = time.time() + 140.0  # safety margin before the 150 s hard kill

# Precompute the C(14,3) = 364 triangle index triples once for speed.
_IDX = np.array([(i, j, k)
                 for i in range(N)
                 for j in range(i + 1, N)
                 for k in range(j + 1, N)], dtype=np.int64)
_I = _IDX[:, 0]
_J = _IDX[:, 1]
_K = _IDX[:, 2]


def _normalize(points):
    """Center and scale a point set to unit convex-hull area."""
    points = points - np.mean(points, axis=0)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return points
    return points / np.sqrt(hull_area)


def _hull_area(points):
    """Convex hull area (2D)."""
    try:
        return ConvexHull(points).volume
    except Exception:
        return 0.0


def _min_triangle_area(points):
    """Minimum area over all C(N,3) triangles (vectorized)."""
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return np.min(0.5 * np.abs(cross))


def _score(points):
    """min triangle area / convex hull area (the quantity the evaluator maximizes)."""
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    return _min_triangle_area(points) / hull_area


def _objective(x):
    """DE objective: minimize -score (maximize score)."""
    pts = x.reshape(N, 2)
    hull_area = _hull_area(pts)
    if hull_area < 1e-9:
        return 1.0
    return -(_min_triangle_area(pts) / hull_area)


def _local_refine(x, time_budget=8.0):
    """Polish a configuration with adaptive Nelder-Mead; return improved points or None."""
    t0 = time.time()
    try:
        res = minimize(_objective, x, method='Nelder-Mead',
                       options={'maxiter': 2500, 'xatol': 1e-12,
                                'fatol': 1e-13, 'adaptive': True})
        if time.time() - t0 > time_budget:
            return None
        return res.x
    except Exception:
        return None


def construct():
    np.random.seed(12345)
    start = time.time()

    # Diverse seed population for DE (all inside the unit box).
    inits = []

    # Regular 14-gon
    a = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(a), np.sin(a)]))

    # Multi-ring configurations with varied radii / offsets / counts.
    ring_specs = [
        (7, 7, [1.0, 0.6]),
        (7, 7, [1.0, 0.5]),
        (7, 7, [1.0, 0.7]),
        (5, 5, 4, [1.0, 0.65, 0.35]),
        (4, 5, 5, [1.0, 0.7, 0.4]),
        (3, 5, 6, [1.0, 0.7, 0.45]),
        (2, 6, 6, [1.0, 0.8, 0.5]),
        (1, 6, 7, [1.0, 0.85, 0.55]),
    ]
    for spec in ring_specs:
        if len(spec) == 3:
            n1, n2, radii = spec
            n3 = 0
        else:
            n1, n2, n3, radii = spec
        blocks = []
        for cnt, r in zip([n1, n2, n3], radii):
            if cnt <= 0:
                continue
            ang = np.linspace(0, 2 * np.pi, cnt, endpoint=False) + np.random.uniform(0, 2 * np.pi)
            blocks.append(np.column_stack([r * np.cos(ang), r * np.sin(ang)]))
        pts = np.vstack(blocks)
        if len(pts) == N:
            inits.append(pts)

    # Random Gaussian clouds
    for _ in range(8):
        inits.append(np.random.randn(N, 2) * 0.5)

    inits = np.vstack(inits)
    inits = (inits + 1.0) / 2.0  # map into [0, 1]^2 for DE bounds

    best_points = None
    best_val = -np.inf

    def _consider(x):
        nonlocal best_points, best_val
        pts = _normalize(x.reshape(N, 2))
        if _hull_area(pts) < 1e-10:
            return
        v = _score(pts)
        if v > best_val:
            best_val = v
            best_points = pts

    # Global differential evolution over the true objective.
    try:
        de = differential_evolution(
            _objective,
            bounds=[(0.0, 1.0)] * (2 * N),
            seed=12345,
            maxiter=400,
            popsize=40,
            tol=1e-8,
            mutation=(0.3, 1.0),
            recombination=0.9,
            polish=False,
            init='latinhypercube',
            updating='deferred',
            x0=inits,
            workers=1,
            callback=lambda xk, convergence: (
                _consider(xk),
                (time.time() - start > _DEADLINE - 30.0)),
        )
        _consider(de.x)
    except Exception:
        pass

    # Final high-precision local polish of the best configuration found.
    if best_points is not None and time.time() < _DEADLINE - 12.0:
        try:
            refined = _local_refine(best_points.flatten(),
                                    time_budget=_DEADLINE - time.time() - 6.0)
            if refined is not None:
                _consider(refined)
        except Exception:
            pass

    if best_points is None:
        a = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(a), np.sin(a)])
        best_points = _normalize(best_points)

    return best_points


if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_score(pts)}")
    print(f"Points shape: {pts.shape}")
