"""Rewritten program for sphere covering with 50 caps."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
import time


def _covering_radius(points):
    """Compute the covering radius (in radians) for a set of points on the unit sphere.

    The covering radius equals the maximum circumradius of the spherical facets
    of the convex hull.
    """
    points = np.asarray(points, dtype=np.float64)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms

    try:
        hull = ConvexHull(points)
    except Exception:
        tree = cKDTree(points)
        rng = np.random.RandomState(42)
        n_samples = 200000
        gauss = rng.randn(n_samples, 3)
        gauss /= np.linalg.norm(gauss, axis=1, keepdims=True)
        d, idx = tree.query(gauss)
        max_theta = 2 * np.arcsin(np.clip(d.max() / 2.0, 0, 1))
        return max_theta

    simplices = hull.simplices
    p0 = points[simplices[:, 0]]
    p1 = points[simplices[:, 1]]
    p2 = points[simplices[:, 2]]

    e1 = p1 - p0
    e2 = p2 - p0

    n = np.cross(e1, e2)
    n_norm = np.linalg.norm(n, axis=1, keepdims=True)

    valid = n_norm.squeeze() > 1e-12
    if not np.all(valid):
        n_norm[~valid] = 1.0

    C = n / n_norm

    centroid = (p0 + p1 + p2) / 3.0
    centroid_norm = np.linalg.norm(centroid, axis=1, keepdims=True)
    centroid_norm[centroid_norm < 1e-12] = 1.0
    centroid = centroid / centroid_norm

    dot_cc = np.sum(C * centroid, axis=1)
    C = np.where(dot_cc[:, None] < 0, -C, C)

    dots = np.sum(C * p0, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    thetas = np.arccos(dots)

    if not np.all(valid):
        thetas[~valid] = 0.0

    return np.max(thetas)


def _fibonacci_sphere(n, offset=0.0):
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _random_sphere(n, seed=42):
    rng = np.random.RandomState(seed)
    gauss = rng.randn(n, 3)
    norms = np.linalg.norm(gauss, axis=1, keepdims=True)
    return gauss / norms


def _local_search(pts, max_iter=5000, step_size=0.05, seed=42):
    """Local search with simulated annealing to minimize covering radius."""
    rng = np.random.RandomState(seed)
    current_pts = pts.copy()
    current_radius = _covering_radius(current_pts)
    best_pts = current_pts.copy()
    best_radius = current_radius

    T = step_size * 0.5
    T_min = 1e-6
    cooling = 0.9995

    for iteration in range(max_iter):
        if T < T_min:
            break

        idx = rng.randint(N)
        original = current_pts[idx].copy()

        delta = rng.randn(3) * step_size
        new_pt = original + delta
        new_pt /= np.linalg.norm(new_pt)

        current_pts[idx] = new_pt
        new_radius = _covering_radius(current_pts)

        if new_radius < current_radius:
            current_radius = new_radius
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = current_pts.copy()
        else:
            prob = np.exp(-(new_radius - current_radius) / T)
            if rng.rand() < prob:
                current_radius = new_radius
            else:
                current_pts[idx] = original

        T *= cooling

    return best_pts, best_radius


def _refine(pts, max_iter=3000, step_size=0.01, seed=12345):
    """Fine-grained local search for final polish."""
    rng = np.random.RandomState(seed)
    current_pts = pts.copy()
    current_radius = _covering_radius(current_pts)
    best_pts = current_pts.copy()
    best_radius = current_radius

    T = step_size * 0.3
    T_min = 1e-7
    cooling = 0.999

    for iteration in range(max_iter):
        if T < T_min:
            break

        idx = rng.randint(N)
        original = current_pts[idx].copy()

        delta = rng.randn(3) * step_size
        new_pt = original + delta
        new_pt /= np.linalg.norm(new_pt)

        current_pts[idx] = new_pt
        new_radius = _covering_radius(current_pts)

        if new_radius < current_radius:
            current_radius = new_radius
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = current_pts.copy()
        else:
            prob = np.exp(-(new_radius - current_radius) / T)
            if rng.rand() < prob:
                current_radius = new_radius
            else:
                current_pts[idx] = original

        T *= cooling

    return best_pts, best_radius


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()

    best_pts = None
    best_radius = np.inf

    # Multi-start with different initial configurations
    initializations = [
        ("fib0", _fibonacci_sphere(N, 0.0)),
        ("fib025", _fibonacci_sphere(N, 0.25)),
        ("fib05", _fibonacci_sphere(N, 0.5)),
        ("fib075", _fibonacci_sphere(N, 0.75)),
        ("rand42", _random_sphere(N, seed=42)),
        ("rand123", _random_sphere(N, seed=123)),
        ("rand999", _random_sphere(N, seed=999)),
        ("rand2024", _random_sphere(N, seed=2024)),
    ]

    # First pass: coarse local search on each initialization
    for name, init_pts in initializations:
        if time.time() - start_time > 80:
            break

        pts, radius = _local_search(init_pts, max_iter=3000, step_size=0.08, seed=hash(name) % 100000)

        if radius < best_radius:
            best_radius = radius
            best_pts = pts.copy()

    # Second pass: finer local search on the best solution
    if best_pts is not None and time.time() - start_time < 100:
        pts, radius = _local_search(best_pts, max_iter=4000, step_size=0.03, seed=77777)
        if radius < best_radius:
            best_radius = radius
            best_pts = pts

    # Third pass: fine refinement
    if best_pts is not None and time.time() - start_time < 112:
        pts, radius = _refine(best_pts, max_iter=3000, step_size=0.005, seed=12345)
        if radius < best_radius:
            best_radius = radius
            best_pts = pts

    # Final ultra-fine polish
    if best_pts is not None and time.time() - start_time < 117:
        pts, radius = _refine(best_pts, max_iter=2000, step_size=0.002, seed=54321)
        if radius < best_radius:
            best_radius = radius
            best_pts = pts

    # Ensure all points are on the unit sphere
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms

    return best_pts
# EVOLVE-BLOCK-END
