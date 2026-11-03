import numpy as np
from scipy.optimize import minimize
import time

N = 50


def _score(points):
    """Compute the packing radius for a set of points."""
    if len(points) != N:
        return -1.0
    if not np.all(np.isfinite(points)):
        return -1.0
    # Distance from origin
    dists = np.sqrt(np.sum(points ** 2, axis=1))
    min_boundary = 1.0 - np.max(dists)
    if min_boundary <= 0:
        return -1.0
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2)) / 2.0
    return min(min_boundary, min_pair)


def _build_ring_params(radii, angles):
    """Build point array from ring parameters.
    
    radii: array of ring radii
    angles: list of angle arrays, one per ring
    """
    pts = [np.zeros((0, 2))]
    for i, r in enumerate(radii):
        if i == 0:
            # Center
            pts.append(np.array([[0.0, 0.0]]))
        else:
            n = len(angles[i])
            a = np.array(angles[i])
            pts.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))
    return np.vstack(pts)


def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit

    # Best packing found so far
    best_points = None
    best_score = -1.0

    def update_best(pts):
        nonlocal best_points, best_score
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
        return s

    # Try known good initial packing: concentric rings
    # Known packing has rings: 1, 6, 12, 18, 13 (total 50)
    # Let's start with a reasonable guess

    # Strategy: Use a parametric ring-based approach
    # Ring structure: 1 center + 6 + 12 + 18 + 13
    ring_sizes = [1, 6, 12, 18, 13]

    # Initial radii guess (will be optimized)
    # The unit circle has radius 1, so max radius for outer ring is about 1 - r_circle
    # With 50 circles of radius ~0.126, outer ring radius ~0.75-0.85
    init_radii = np.array([0.0, 0.25, 0.50, 0.70, 0.85])

    # Build initial angle arrays
    angles_list = []
    for i, n in enumerate(ring_sizes):
        if i == 0:
            angles_list.append(np.array([0.0]))
        else:
            angles_list.append(np.linspace(0, 2 * np.pi, n, endpoint=False))

    def flatten_params(radii, angles_list):
        """Flatten ring parameters into a single vector."""
        vec = list(radii)
        for i in range(1, len(ring_sizes)):
            vec.extend(list(angles_list[i]))
        return np.array(vec)

    def unflatten_params(vec):
        """Unflatten parameter vector back to radii and angles."""
        radii = vec[:5].copy()
        angles_list = [np.array([0.0])]
        idx = 5
        for i in range(1, len(ring_sizes)):
            n = ring_sizes[i]
            angles_list.append(vec[idx:idx + n].copy())
            idx += n
        return radii, angles_list

    def objective(vec):
        """Maximize the packing radius = minimize negative radius."""
        if time.time() - start_time > deadline:
            return 1e10
        radii, angles_list = unflatten_params(vec)
        # Ensure radii are in reasonable range
        radii = np.clip(radii, 0.0, 0.95)
        pts = _build_ring_params(radii, angles_list)
        if len(pts) != N:
            return 1e10
        s = _score(pts)
        return -s

    # Optimize ring parameters
    x0 = flatten_params(init_radii, angles_list)
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
        if time.time() - start_time < deadline - 20:
            x_opt = result.x
            radii_opt, angles_opt = unflatten_params(x_opt)
            radii_opt = np.clip(radii_opt, 0.0, 0.95)
            pts_opt = _build_ring_params(radii_opt, angles_opt)
            update_best(pts_opt)
    except Exception:
        pass

    # Local refinement: gradient-free optimization on individual points
    if time.time() - start_time < deadline - 10 and best_points is not None:
        def objective_points(vec):
            if time.time() - start_time > deadline:
                return 1e10
            pts = vec.reshape(N, 2)
            s = _score(pts)
            return -s

        x0 = best_points.flatten()
        try:
            result = minimize(objective_points, x0, method='Nelder-Mead',
                              options={'maxiter': 50000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if time.time() - start_time < deadline - 5:
                x_opt = result.x
                pts_opt = x_opt.reshape(N, 2)
                update_best(pts_opt)
        except Exception:
            pass

    # Additional local refinement with tighter tolerances
    if time.time() - start_time < deadline - 5 and best_points is not None:
        def objective_points_tight(vec):
            if time.time() - start_time > deadline:
                return 1e10
            pts = vec.reshape(N, 2)
            s = _score(pts)
            return -s

        x0 = best_points.flatten()
        try:
            result = minimize(objective_points_tight, x0, method='Nelder-Mead',
                              options={'maxiter': 100000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if time.time() - start_time < deadline - 2:
                x_opt = result.x
                pts_opt = x_opt.reshape(N, 2)
                update_best(pts_opt)
        except Exception:
            pass

    # Final check and return
    if best_points is None:
        # Fallback: hexagonal grid
        spacing = 0.24
        points = []
        for j in range(-10, 11):
            for i in range(-10, 11):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.sqrt(x * x + y * y) <= 1.0 - spacing / 2.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) > N:
            points = points[:N]
        best_points = points

    return best_points
