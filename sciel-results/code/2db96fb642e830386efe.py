"""Optimized sphere covering with 130 points — multi-restart Lloyd + smooth-max gradient."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130
_DEADLINE = time.time() + 140  # 10s safety margin

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _time_ok(margin=5.0):
    return time.time() < _DEADLINE - margin

def _covering_radius(pts):
    """Compute the covering radius in radians via convex hull facet circumcenters."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi / 2

    max_r = 0.0
    for simplex in hull.simplices:
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        v1, v2 = b - a, c - a
        cross = np.cross(v1, v2)
        cnorm = np.linalg.norm(cross)
        if cnorm < 1e-12:
            continue
        x = cross / cnorm
        centroid = (a + b + c) / 3.0
        if np.dot(x, centroid) < 0:
            x = -x
        dot_val = np.clip(np.dot(x, a), -1.0, 1.0)
        r = np.arccos(dot_val)
        if r > max_r:
            max_r = r
    return max_r

def _covering_radius_fast(pts):
    """Alias for _covering_radius."""
    return _covering_radius(pts)

def _covering_radius_grid(pts, n_grid=50):
    """Fast approximate covering radius using a grid on the sphere."""
    pts = _normalize(pts)
    # Create a grid of test points
    theta = np.linspace(0, np.pi, n_grid)
    phi = np.linspace(0, 2 * np.pi, n_grid)
    theta, phi = np.meshgrid(theta, phi)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    test_pts = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
    
    # For each test point, find nearest configuration point
    max_dist = 0.0
    for tp in test_pts:
        dots = np.clip(pts @ tp, -1.0, 1.0)
        min_dist = np.arccos(np.max(dots))
        if min_dist > max_dist:
            max_dist = min_dist
    return max_dist

def _initial_config(seed=42):
    """Generate initial configuration using Fibonacci spiral with randomization."""
    rng = np.random.RandomState(seed)
    i = np.arange(N) + 0.5 + rng.randn(N) * 0.1
    i = np.clip(i, 0.5, N - 0.5)
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rng.randn(N) * 0.05
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _fibonacci_spiral(n, jitter=0.0, seed=0):
    """Fibonacci spiral on the unit sphere with optional jitter."""
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    if jitter > 0:
        i = i + rng.randn(n) * jitter
        i = np.clip(i, 0.5, n - 0.5)
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _lloyd_step(pts, lr=0.05, smooth_beta=20.0):
    """
    One Lloyd-type step: for each point, compute the centroid of its Voronoi cell
    (approximated via smooth-max of angular distances to facet circumcenters) and
    move toward it. Uses convex hull to get facets.
    """
    pts = _normalize(pts.copy())
    try:
        hull = ConvexHull(pts)
    except Exception:
        return pts, None

    # For each facet, compute circumcenter
    circumcenters = []
    facet_pts = []  # indices of the 3 points per facet
    for simplex in hull.simplices:
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        v1, v2 = b - a, c - a
        cross = np.cross(v1, v2)
        cnorm = np.linalg.norm(cross)
        if cnorm < 1e-12:
            continue
        x = cross / cnorm
        centroid = (a + b + c) / 3.0
        if np.dot(x, centroid) < 0:
            x = -x
        circumcenters.append(x)
        facet_pts.append(simplex)

    if not circumcenters:
        return pts, None

    CC = np.array(circumcenters)  # (F, 3)
    F = len(CC)
    facet_pts_arr = np.array(facet_pts)

    # For each point i, find the circumcenters of facets that include i
    # and compute a weighted centroid (Lloyd step)
    new_pts = pts.copy()
    for i in range(N):
        # Find facets containing point i
        mask = np.any(facet_pts_arr == i, axis=1)
        if not np.any(mask):
            continue
        cc_i = CC[mask]  # circumcenters adjacent to point i
        # Angular distance from point i to each circumcenter
        dots = np.clip(cc_i @ pts[i], -1.0, 1.0)
        dists = np.arccos(dots)
        # Smooth max weight: emphasize the farthest circumcenter
        # Use softmax with beta for emphasis
        weights = np.exp(smooth_beta * (dists - np.max(dists)))
        weights /= np.sum(weights)
        # Weighted centroid of circumcenters (on the sphere)
        centroid_dir = weights @ cc_i
        cnorm = np.linalg.norm(centroid_dir)
        if cnorm < 1e-12:
            continue
        centroid_dir /= cnorm
        # Move point i toward the centroid
        new_pts[i] = pts[i] + lr * (centroid_dir - pts[i])

    new_pts = _normalize(new_pts)
    return new_pts, None

def _gradient_descent(pts, max_iters=500, lr_init=0.02, lr_decay=0.995, beta=15.0):
    """
    Gradient descent on a smooth approximation of the covering radius.
    For each point, the gradient direction is toward the farthest circumcenter
    of its adjacent facets (smoothed).
    """
    pts = _normalize(pts.copy())
    best_pts = pts.copy()
    best_r = _covering_radius_fast(pts)
    lr = lr_init

    for it in range(max_iters):
        if not _time_ok(5.0):
            break

        try:
            hull = ConvexHull(pts)
        except Exception:
            break

        # Compute all circumcenters
        CC_list = []
        facet_indices = []
        for simplex in hull.simplices:
            a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            v1, v2 = b - a, c - a
            cross = np.cross(v1, v2)
            cnorm = np.linalg.norm(cross)
            if cnorm < 1e-12:
                continue
            x = cross / cnorm
            centroid = (a + b + c) / 3.0
            if np.dot(x, centroid) < 0:
                x = -x
            CC_list.append(x)
            facet_indices.append(simplex)

        if not CC_list:
            break

        CC = np.array(CC_list)
        F = len(CC)
        facet_indices_arr = np.array(facet_indices)

        # For each point, find its adjacent circumcenters and compute gradient
        grad = np.zeros_like(pts)
        for i in range(N):
            # Facets containing point i
            mask = np.any(facet_indices_arr == i, axis=1)
            if not np.any(mask):
                continue
            cc_i = CC[mask]
            dots = np.clip(cc_i @ pts[i], -1.0, 1.0)
            dists = np.arccos(dots)
            # Smooth-max gradient: weight by softmax(beta * dist)
            shifted = dists - np.max(dists)
            w = np.exp(beta * shifted)
            w_sum = np.sum(w)
            if w_sum < 1e-15:
                continue
            w /= w_sum
            # Direction: move toward weighted circumcenter
            target = w @ cc_i
            tn = np.linalg.norm(target)
            if tn < 1e-12:
                continue
            target /= tn
            grad[i] = target - pts[i]

        # Normalize gradient magnitude per point
        gnorms = np.linalg.norm(grad, axis=1, keepdims=True)
        gnorms[gnorms < 1e-12] = 1.0
        grad /= gnorms

        # Take step
        pts = _normalize(pts + lr * grad)

        # Evaluate
        cr = _covering_radius_fast(pts)
        if cr < best_r:
            best_r = cr
            best_pts = pts.copy()

        lr *= lr_decay
        if lr < 0.001:
            break

    return best_pts, best_r

def _simulated_annealing(pts, max_iters=200, t_init=0.1, t_decay=0.95, t_min=0.001):
    """
    Simulated annealing to escape local minima.
    Randomly perturb points and accept worse solutions with probability exp(-delta/T).
    """
    pts = _normalize(pts.copy())
    best_pts = pts.copy()
    best_r = _covering_radius_fast(pts)
    T = t_init
    rng = np.random.RandomState(42)

    for it in range(max_iters):
        if not _time_ok(5.0):
            break
        if T < t_min:
            break

        # Generate perturbation
        perturb = rng.randn(N, 3) * T
        new_pts = _normalize(pts + perturb)

        new_r = _covering_radius_fast(new_pts)
        delta = new_r - best_r

        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or rng.rand() < np.exp(-delta / T):
            pts = new_pts
            if new_r < best_r:
                best_r = new_r
                best_pts = new_pts.copy()

        T *= t_decay

    return best_pts, best_r

def construct():
    """
    Multi-restart strategy:
    1. Generate several diverse initial configs (Fibonacci + jitter, random).
    2. For each, run Lloyd + gradient descent + simulated annealing.
    3. Keep the best.
    """
    start_time = time.time()
    best_pts = None
    best_r = np.pi

    # Generate diverse initial configurations
    inits = []
    # Fibonacci with varying jitter
    for seed in [42, 123, 7, 999, 31415, 271828, 161803]:
        inits.append(_fibonacci_spiral(N, jitter=0.05, seed=seed))
    # Pure Fibonacci
    inits.append(_fibonacci_spiral(N, jitter=0.0, seed=0))
    # Random
    rng = np.random.RandomState(555)
    inits.append(_normalize(rng.randn(N, 3)))
    # More random configs
    for seed in [111, 222, 333]:
        rng = np.random.RandomState(seed)
        inits.append(_normalize(rng.randn(N, 3)))

    # Sort by initial covering radius, try best first
    init_scores = []
    for p in inits:
        try:
            r = _covering_radius_fast(p)
        except Exception:
            r = np.pi
        init_scores.append((r, p))
    init_scores.sort(key=lambda x: x[0])

    for r0, p0 in init_scores:
        if not _time_ok(10.0):
            break

        # Phase 1: Lloyd iterations
        pts = p0.copy()
        for lloyd_iter in range(30):
            if not _time_ok(8.0):
                break
            pts, _ = _lloyd_step(pts, lr=0.08, smooth_beta=15.0)

        cr_lloyd = _covering_radius_fast(pts)
        if cr_lloyd < best_r:
            best_r = cr_lloyd
            best_pts = pts.copy()

        # Phase 2: Gradient descent refinement
        if _time_ok(10.0):
            pts2, cr2 = _gradient_descent(pts, max_iters=200, lr_init=0.015, lr_decay=0.99, beta=12.0)
            if cr2 < best_r:
                best_r = cr2
                best_pts = pts2.copy()

        # Phase 3: Simulated annealing to escape local minima
        if _time_ok(8.0):
            pts3, cr3 = _simulated_annealing(best_pts if best_pts is not None else pts, max_iters=100, t_init=0.05, t_decay=0.97)
            if cr3 < best_r:
                best_r = cr3
                best_pts = pts3.copy()

    # Final polish: a few more Lloyd steps on the best
    if best_pts is not None and _time_ok(5.0):
        for _ in range(15):
            if not _time_ok(4.0):
                break
            best_pts, _ = _lloyd_step(best_pts, lr=0.03, smooth_beta=20.0)
        cr_final = _covering_radius_fast(best_pts)
        if cr_final < best_r:
            best_r = cr_final

    if best_pts is None:
        best_pts = _fibonacci_spiral(N)

    return _normalize(best_pts)

# Test
if __name__ == "__main__":
    t0 = time.time()
    pts = construct()
    print("Shape:", pts.shape)
    print("Covering radius (deg):", np.degrees(_covering_radius_fast(pts)))
    print("Time: %.1f s" % (time.time() - t0))
