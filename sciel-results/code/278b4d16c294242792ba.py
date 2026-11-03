"""
Pack 200 equal circles in a unit square using a combination of
Delaunay-based local search, simulated annealing, and gradient polishing.
"""
import numpy as np
import time
from scipy.spatial import Delaunay, cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 200
_TIME_LIMIT = 145  # Leave margin for 150s limit


def _compute_radius(pts):
    """
    Calculate the maximum possible radius for given centers.
    Returns the min of (wall distances) and (pairwise distances / 2).
    """
    if len(pts) == 0:
        return 0.0

    # Wall distances
    min_wall = min(
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1])
    )

    # Pairwise distances using cKDTree for efficiency
    tree = cKDTree(pts)
    # Query all pairs
    pairs = tree.query_pairs(r=1.0)  # r=1.0 is a safe upper bound
    if len(pairs) == 0:
        return min_wall

    min_pair = np.inf
    for i, j in pairs:
        d = np.linalg.norm(pts[i] - pts[j])
        if d < min_pair:
            min_pair = d

    return min(min_wall, min_pair / 2.0)


def _bottleneck_info(pts):
    """
    Find the bottleneck constraint (either wall or pair) that limits the radius.
    Returns (value, type, indices) where type is 'wall' or 'pair'.
    """
    # Wall distances
    wall_dists = np.array([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1])
    ])
    min_wall_idx = np.argmin(wall_dists)
    min_wall = wall_dists[min_wall_idx]

    # Pairwise distances
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=1.0)
    if len(pairs) == 0:
        return min_wall, 'wall', min_wall_idx

    min_pair = np.inf
    min_pair_idx = None
    for i, j in pairs:
        d = np.linalg.norm(pts[i] - pts[j])
        if d < min_pair:
            min_pair = d
            min_pair_idx = (i, j)

    min_pair_half = min_pair / 2.0

    if min_wall < min_pair_half:
        return min_wall, 'wall', min_wall_idx
    else:
        return min_pair_half, 'pair', min_pair_idx


def _generate_hex_grid(spacing):
    """Generate a hexagonal lattice grid."""
    points = []
    row_h = spacing * np.sqrt(3) / 2
    y = -0.5 + spacing / 2
    row_idx = 0
    while y < 0.5:
        shift = (spacing / 2.0) if row_idx % 2 == 1 else 0.0
        x = -0.5 + spacing / 2 + shift
        while x < 0.5:
            points.append([x, y])
            x += spacing
        y += row_h
        row_idx += 1
    return np.array(points)


def _initial_points():
    """
    Generate initial points using a hexagonal grid, then select N points
    that maximize the minimum distance.
    """
    # Try different spacings to get roughly N points
    best_pts = None
    best_r = 0

    for s in [0.07, 0.075, 0.08, 0.085, 0.09]:
        pts = _generate_hex_grid(s)
        if len(pts) >= N:
            # Select N points that maximize min distance
            # Simple approach: take points sorted by distance from center
            center = np.mean(pts, axis=0)
            dists = np.linalg.norm(pts - center, axis=1)
            sorted_idx = np.argsort(dists)
            selected = pts[sorted_idx[:N]]
            r = _compute_radius(selected)
            if r > best_r:
                best_r = r
                best_pts = selected
        elif len(pts) < N:
            # Pad with random points
            extra = np.random.uniform(-0.45, 0.45, (N - len(pts), 2))
            selected = np.vstack([pts, extra])
            r = _compute_radius(selected)
            if r > best_r:
                best_r = r
                best_pts = selected

    if best_pts is None:
        best_pts = np.random.uniform(-0.45, 0.45, (N, 2))

    return best_pts


def _local_search(pts, max_iters=5000, time_limit=None):
    """
    Local search that tries to improve the minimum radius by moving
    the bottleneck point(s).
    """
    t0 = time.time()
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)

    for it in range(max_iters):
        if time_limit and (time.time() - t0) > time_limit:
            break

        r, btype, bidx = _bottleneck_info(best_pts)

        if btype == 'wall':
            # Move the point closest to the wall away from it
            if bidx == 0:  # left wall
                # Find point with min x
                idx = np.argmin(best_pts[:, 0])
                best_pts[idx, 0] += 0.01
            elif bidx == 1:  # right wall
                idx = np.argmax(best_pts[:, 0])
                best_pts[idx, 0] -= 0.01
            elif bidx == 2:  # bottom wall
                idx = np.argmin(best_pts[:, 1])
                best_pts[idx, 1] += 0.01
            else:  # top wall
                idx = np.argmax(best_pts[:, 1])
                best_pts[idx, 1] -= 0.01
        else:
            # Move one of the two points in the bottleneck pair apart
            i, j = bidx
            dx = best_pts[i, 0] - best_pts[j, 0]
            dy = best_pts[i, 1] - best_pts[j, 1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 1e-10:
                step = 0.005
                best_pts[i, 0] += dx / dist * step
                best_pts[i, 1] += dy / dist * step
                best_pts[j, 0] -= dx / dist * step
                best_pts[j, 1] -= dy / dist * step
            else:
                # Random perturbation
                best_pts[i, 0] += np.random.uniform(-0.01, 0.01)
                best_pts[i, 1] += np.random.uniform(-0.01, 0.01)

        # Clip to square
        best_pts = np.clip(best_pts, -0.5, 0.5)

        r_new = _compute_radius(best_pts)
        if r_new > best_r:
            best_r = r_new
        else:
            # Revert
            best_pts = best_pts.copy()

    return best_pts, best_r


def _simulated_annealing(pts, t0=1.0, t_end=0.01, max_iters=10000, time_limit=None):
    """
    Simulated annealing to escape local minima.
    """
    t_start = time.time()
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    current_pts = pts.copy()
    current_r = best_r

    alpha = 0.9995  # Cooling rate

    for it in range(max_iters):
        if time_limit and (time.time() - t_start) > time_limit:
            break

        # Generate a random perturbation
        candidate = current_pts.copy()
        idx = np.random.randint(N)
        step = t0 * np.exp(-it / max_iters) * 0.02
        candidate[idx, 0] += np.random.uniform(-step, step)
        candidate[idx, 1] += np.random.uniform(-step, step)
        candidate = np.clip(candidate, -0.5, 0.5)

        r_new = _compute_radius(candidate)
        delta = r_new - current_r

        if delta > 0:
            current_pts = candidate
            current_r = r_new
            if r_new > best_r:
                best_r = r_new
                best_pts = candidate.copy()
        else:
            # Accept with probability exp(delta / t)
            t = t0 * (t_end / t0) ** (it / max_iters)
            if np.random.random() < np.exp(delta / max(t, 1e-10)):
                current_pts = candidate
                current_r = r_new

    return best_pts, best_r


def _gradient_polish(pts, time_limit=None):
    """
    Use gradient-based optimization to polish the solution.
    """
    t_start = time.time()

    def objective(x):
        pts_local = x.reshape((N, 2))
        return -_compute_radius(pts_local)

    x0 = pts.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)

    try:
        res = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        final_pts = res.x.reshape((N, 2))
        final_pts = np.clip(final_pts, -0.5, 0.5)
        r = _compute_radius(final_pts)
        if r > _compute_radius(pts):
            return final_pts, r
    except:
        pass

    return pts, _compute_radius(pts)


def construct():
    t0 = time.time()
    np.random.seed(42)

    # Initial points
    pts = _initial_points()
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)

    # Phase 1: Local search
    remaining = _TIME_LIMIT - (time.time() - t0) - 30
    if remaining > 5:
        pts, r = _local_search(pts, max_iters=3000, time_limit=remaining)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    # Phase 2: Simulated annealing
    remaining = _TIME_LIMIT - (time.time() - t0) - 20
    if remaining > 5:
        pts, r = _simulated_annealing(pts, max_iters=5000, time_limit=remaining)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    # Phase 3: Another local search
    remaining = _TIME_LIMIT - (time.time() - t0) - 10
    if remaining > 3:
        pts, r = _local_search(best_pts, max_iters=2000, time_limit=remaining)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    # Phase 4: Gradient polish
    remaining = _TIME_LIMIT - (time.time() - t0)
    if remaining > 2:
        pts, r = _gradient_polish(best_pts, time_limit=remaining - 1)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    # Final check
    final_r = _compute_radius(best_pts)
    if final_r > best_r:
        best_r = final_r

    return best_pts


if __name__ == "__main__":
    pts = construct()
    r = _compute_radius(pts)
    print(f"Best Radius: {r}")
