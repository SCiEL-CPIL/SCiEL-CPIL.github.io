"""Circle packing in unit square: pack 26 circles, maximize sum of radii.
Strategy: Voronoi tessellation dual representation with gradient ascent on sum of inradii."""

import numpy as np
import time
from scipy.spatial import Voronoi
from scipy.optimize import minimize

N = 26


def _eval_packing(xy, r):
    """Evaluate a packing: returns (total_radius, is_valid)."""
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True


def _voronoi_inradii(centers):
    """
    Compute inradii of Voronoi cells for points in [0,1]^2.
    Each circle is inscribed in its Voronoi cell, guaranteeing non-overlap.
    Returns array of inradii (one per center).
    """
    n = len(centers)
    # Add boundary points to ensure Voronoi cells are bounded
    # Use a large set of boundary points
    boundary_pts = []
    # Corners and edge midpoints, plus extra points
    for i in range(-5, 8):
        t = i / 5.0
        boundary_pts.append([t, -0.5])
        boundary_pts.append([t, 1.5])
        boundary_pts.append([-0.5, t])
        boundary_pts.append([1.5, t])
    # Additional boundary points for better cell shapes
    for i in range(20):
        t = (i + 0.5) / 20.0
        boundary_pts.append([t, -0.1])
        boundary_pts.append([t, 1.1])
        boundary_pts.append([-0.1, t])
        boundary_pts.append([1.1, t])

    all_pts = np.vstack([centers, np.array(boundary_pts)])

    try:
        vor = Voronoi(all_pts)
    except Exception:
        # Fallback: use minimum distance to other centers and walls
        radii = np.zeros(n)
        for i in range(n):
            min_dist = min(centers[i, 0], 1.0 - centers[i, 0],
                           centers[i, 1], 1.0 - centers[i, 1])
            for j in range(n):
                if i != j:
                    d = np.sqrt((centers[i, 0] - centers[j, 0])**2 +
                                (centers[i, 1] - centers[j, 1])**2)
                    min_dist = min(min_dist, d / 2.0)
            radii[i] = max(min_dist * 0.99, 0.001)
        return radii

    radii = np.zeros(n)
    for i in range(n):
        # Get Voronoi region for point i
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]

        if len(region) == 0 or -1 in region:
            # Unbounded region - use distance to nearest neighbor/wall
            min_dist = min(centers[i, 0], 1.0 - centers[i, 0],
                           centers[i, 1], 1.0 - centers[i, 1])
            for j in range(n):
                if i != j:
                    d = np.sqrt((centers[i, 0] - centers[j, 0])**2 +
                                (centers[i, 1] - centers[j, 1])**2)
                    min_dist = min(min_dist, d / 2.0)
            radii[i] = max(min_dist * 0.99, 0.001)
            continue

        # Get vertices of the Voronoi cell
        vertices = vor.vertices[region]

        if len(vertices) == 0:
            radii[i] = 0.001
            continue

        # Inradius = minimum distance from center to each edge of the cell
        # Edge from vertices[k] to vertices[(k+1)%len]
        min_dist = np.inf
        nv = len(vertices)
        for k in range(nv):
            v1 = vertices[k]
            v2 = vertices[(k + 1) % nv]
            # Distance from point to line segment
            d = _point_to_segment_distance(centers[i], v1, v2)
            min_dist = min(min_dist, d)

        radii[i] = max(min_dist, 0.001)

    return radii


def _point_to_segment_distance(p, a, b):
    """Distance from point p to line segment ab."""
    ab = b - a
    ap = p - a
    ab_sq = np.dot(ab, ab)
    if ab_sq < 1e-20:
        return np.sqrt(np.dot(ap, ap))
    t = np.dot(ap, ab) / ab_sq
    t = np.clip(t, 0.0, 1.0)
    closest = a + t * ab
    return np.sqrt(np.dot(p - closest, p - closest))


def _voronoi_objective(v, n):
    """
    Objective: minimize -sum(inradii).
    v is the flattened center positions (2n values).
    """
    centers = v.reshape(n, 2)
    # Clip to [0,1]
    centers = np.clip(centers, 0.001, 0.999)
    radii = _voronoi_inradii(centers)
    return -np.sum(radii)


def _voronoi_gradient(v, n, eps=1e-5):
    """
    Numerical gradient of the Voronoi objective.
    """
    grad = np.zeros(2 * n)
    for i in range(2 * n):
        v_plus = v.copy()
        v_plus[i] += eps
        f_plus = _voronoi_objective(v_plus, n)

        v_minus = v.copy()
        v_minus[i] -= eps
        f_minus = _voronoi_objective(v_minus, n)

        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad


def _voronoi_optimize(centers0, deadline, method='gradient_ascent', max_iter=200):
    """
    Optimize center positions to maximize sum of Voronoi inradii.
    Uses gradient ascent (or L-BFGS-B on negative objective).
    """
    n = len(centers0)
    best_centers = centers0.copy()
    best_radii = _voronoi_inradii(best_centers)
    best_sum = np.sum(best_radii)

    if method == 'gradient_ascent':
        # Simple gradient ascent with adaptive step size
        step_size = 0.05
        for it in range(max_iter):
            if time.time() > deadline:
                break

            centers = np.clip(best_centers, 0.001, 0.999)
            grad = _voronoi_gradient(centers.ravel(), n)
            grad = grad.reshape(n, 2)

            # Normalize gradient
            grad_norm = np.sqrt(np.sum(grad**2))
            if grad_norm < 1e-10:
                break
            grad = grad / grad_norm

            # Try a few step sizes
            best_step_sum = best_sum
            best_new_centers = best_centers.copy()
            for ss in [step_size, step_size * 0.5, step_size * 2.0]:
                new_centers = np.clip(best_centers + ss * grad, 0.001, 0.999)
                new_radii = _voronoi_inradii(new_centers)
                new_sum = np.sum(new_radii)
                if new_sum > best_step_sum:
                    best_step_sum = new_sum
                    best_new_centers = new_centers

            if best_sum >= best_step_sum - 1e-10:
                # No improvement, reduce step size
                step_size *= 0.7
                if step_size < 1e-6:
                    break
            else:
                best_centers = best_new_centers
                best_radii = _voronoi_inradii(best_centers)
                best_sum = np.sum(best_radii)
                step_size = min(step_size * 1.1, 0.1)

    elif method == 'lbfgs':
        v0 = best_centers.ravel()
        bounds = [(0.001, 0.999)] * (2 * n)

        def obj(v):
            if time.time() > deadline:
                return float('inf')
            return _voronoi_objective(v, n)

        def grad(v):
            if time.time() > deadline:
                return np.zeros(2 * n)
            return _voronoi_gradient(v, n)

        try:
            result = minimize(obj, v0, jac=grad, method='L-BFGS-B',
                              bounds=bounds,
                              options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8,
                                       'maxfun': 5000})
            new_centers = np.clip(result.x.reshape(n, 2), 0.001, 0.999)
            new_radii = _voronoi_inradii(new_centers)
            new_sum = np.sum(new_radii)
            if new_sum > best_sum:
                best_centers = new_centers
                best_radii = new_radii
                best_sum = new_sum
        except Exception:
            pass

    return best_centers, best_radii, best_sum


def _grid_init(side):
    """Create a grid of up to N circle centers."""
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < N:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                positions.append([x, y])
    while len(positions) < N:
        positions.append([0.5, 0.5])
    return np.array(positions[:N])


def _random_init(n, seed=0):
    """Random initialization of centers."""
    rng = np.random.RandomState(seed)
    return rng.uniform(0.1, 0.9, size=(n, 2))


def _local_search_voronoi(centers, deadline, grid_size=50):
    """
    Local search: for each center, try moving it to a nearby grid position
    that maximizes the sum of Voronoi inradii.
    """
    n = len(centers)
    best_centers = centers.copy()
    best_radii = _voronoi_inradii(best_centers)
    best_sum = np.sum(best_radii)

    # Create a local grid around each center
    for sweep in range(5):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            # Local grid around current position
            cx, cy = best_centers[i]
            local_range = 0.15
            gx = np.linspace(max(0.001, cx - local_range), min(0.999, cx + local_range), grid_size)
            gy = np.linspace(max(0.001, cy - local_range), min(0.999, cy + local_range), grid_size)
            GX, GY = np.meshgrid(gx, gy, indexing='ij')
            candidates = np.column_stack([GX.ravel(), GY.ravel()])

            best_local_sum = best_sum
            best_local_center = best_centers[i].copy()
            for cand in candidates:
                new_centers = best_centers.copy()
                new_centers[i] = cand
                radii = _voronoi_inradii(new_centers)
                s = np.sum(radii)
                if s > best_local_sum:
                    best_local_sum = s
                    best_local_center = cand.copy()

            if np.sum(best_local_center**2) != np.sum(best_centers[i]**2) or best_local_sum > best_sum + 1e-10:
                best_centers[i] = best_local_center
                best_radii = _voronoi_inradii(best_centers)
                best_sum = np.sum(best_radii)
                improved = True

        if not improved:
            break

    return best_centers, best_radii, best_sum


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    best_centers = None
    best_radii = None
    best_sum = 0.0

    # Phase 1: Multiple grid initializations with Voronoi optimization
    for side in [5, 6, 7]:
        if time.time() > deadline:
            break
        centers = _grid_init(side)

        # Gradient ascent optimization
        out_centers, out_radii, out_sum = _voronoi_optimize(
            centers.copy(), deadline, method='gradient_ascent', max_iter=100)

        if out_sum > best_sum:
            best_sum = out_sum
            best_centers = out_centers
            best_radii = out_radii

        # Local search refinement
        if time.time() < deadline - 20:
            ls_centers, ls_radii, ls_sum = _local_search_voronoi(
                best_centers.copy(), deadline, grid_size=30)
            if ls_sum > best_sum:
                best_sum = ls_sum
                best_centers = ls_centers
                best_radii = ls_radii

    # Phase 2: Random initializations
    for seed in range(10):
        if time.time() > deadline:
            break
        centers = _random_init(N, seed=seed)

        # Gradient ascent
        out_centers, out_radii, out_sum = _voronoi_optimize(
            centers.copy(), deadline, method='gradient_ascent', max_iter=80)

        if out_sum > best_sum:
            best_sum = out_sum
            best_centers = out_centers
            best_radii = out_radii

    # Phase 3: L-BFGS-B polish on best result
    if best_centers is not None and time.time() < deadline - 30:
        out_centers, out_radii, out_sum = _voronoi_optimize(
            best_centers.copy(), deadline, method='lbfgs')
        if out_sum > best_sum:
            best_sum = out_sum
            best_centers = out_centers
            best_radii = out_radii

    # Phase 4: Final local search
    if best_centers is not None and time.time() < deadline - 10:
        ls_centers, ls_radii, ls_sum = _local_search_voronoi(
            best_centers.copy(), deadline, grid_size=40)
        if ls_sum > best_sum:
            best_sum = ls_sum
            best_centers = ls_centers
            best_radii = ls_radii

    # Fallback
    if best_centers is None:
        centers = _grid_init(6)
        radii = _voronoi_inradii(centers)
        best_centers = centers
        best_radii = radii
        best_sum = np.sum(radii)

    # Build final result: centers and inradii
    n = len(best_centers)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y = best_centers[i]
        ri = best_radii[i]
        # Ensure circle is inside the unit square
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
        result[i, 2] = ri

    # Final validation
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)

    if not valid:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
