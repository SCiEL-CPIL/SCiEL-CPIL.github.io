import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _score(pts):
    """Compute packing radius: min(pairwise_dist/2, 1 - max_radius)."""
    if len(pts) != N:
        return -1.0
    if not np.all(np.isfinite(pts)):
        return -1.0
    # Distance from origin
    dists = np.sqrt(np.sum(pts ** 2, axis=1))
    min_boundary = 1.0 - np.max(dists)
    if min_boundary <= 0:
        return -1.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2)) / 2.0
    return min(min_boundary, min_pair)

def _ring_to_points(radii, angles_list):
    """Convert ring parameters to point array."""
    pts = [np.array([[0.0, 0.0]])]
    for i in range(1, len(radii)):
        r = radii[i]
        a = angles_list[i]
        pts.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))
    return np.vstack(pts)

def _flatten_ring(radii, angles_list):
    """Flatten ring params to a 1D vector."""
    vec = list(radii)
    for i in range(1, len(angles_list)):
        vec.extend(list(angles_list[i]))
    return np.array(vec)

def _unflatten_ring(vec, ring_sizes):
    """Unflatten 1D vector to radii and angles_list."""
    radii = vec[:len(ring_sizes)].copy()
    angles_list = [np.array([0.0])]
    idx = len(ring_sizes)
    for i in range(1, len(ring_sizes)):
        n = ring_sizes[i]
        angles_list.append(vec[idx:idx + n].copy())
        idx += n
    return radii, angles_list

def _soft_violation(C, r):
    """Soft violation cost for a fixed radius r. Lower is better (0 = feasible)."""
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_viol = np.maximum(0, 2.0 * r - dist)
    pair_cost = np.sum(pair_viol ** 2)
    rad = np.linalg.norm(C, axis=1)
    bound_viol = np.maximum(0, rad - (1.0 - r))
    bound_cost = np.sum(bound_viol ** 2)
    return pair_cost + bound_cost

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill

    ring_sizes = [1, 6, 12, 18, 13]  # total 50
    n_params = len(ring_sizes) + sum(ring_sizes[1:])  # 5 + 6+12+18+13 = 54

    # Initial ring radii estimates
    init_radii = np.array([0.0, 0.252, 0.504, 0.756, 0.875])
    angles_list = []
    for i, n in enumerate(ring_sizes):
        if i == 0:
            angles_list.append(np.array([0.0]))
        else:
            offset = np.pi / n if i % 2 == 0 else 0.0
            angles_list.append(np.linspace(0, 2 * np.pi, n, endpoint=False) + offset)

    x0 = _flatten_ring(init_radii, angles_list)
    best_points = None
    best_score = -1.0

    def update_best(pts):
        nonlocal best_points, best_score
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
        return s

    # --- Stage 1: Differential Evolution on ring parameters ---
    # This is the structural change: global search over the low-dimensional ring space
    if time.time() < deadline - 50:
        # Bounds: radii in [0, 0.95], angles in [0, 2*pi]
        bounds = [(0.0, 0.95)] * len(ring_sizes)
        for i in range(1, len(ring_sizes)):
            bounds.extend([(0, 2 * np.pi)] * ring_sizes[i])

        def ring_objective(vec):
            if time.time() > deadline - 50:
                return 1e10
            radii, angles_list = _unflatten_ring(vec, ring_sizes)
            radii = np.clip(radii, 0.0, 0.95)
            pts = _ring_to_points(radii, angles_list)
            return -_score(pts)

        try:
            result_de = differential_evolution(
                ring_objective, bounds,
                maxiter=100, popsize=15, tol=1e-10,
                mutation=(0.3, 1.0), recombination=0.8,
                seed=42, polish=False,
                callback=lambda xk, convergence: None
            )
            if time.time() < deadline - 40:
                x_opt = result_de.x
                radii_opt, angles_opt = _unflatten_ring(x_opt, ring_sizes)
                radii_opt = np.clip(radii_opt, 0.0, 0.95)
                pts_opt = _ring_to_points(radii_opt, angles_opt)
                update_best(pts_opt)
        except Exception:
            pass

    # --- Stage 2: L-BFGS-B refinement on ring parameters ---
    if time.time() < deadline - 30 and best_points is not None:
        bounds = [(0.0, 0.95)] * len(ring_sizes)
        for i in range(1, len(ring_sizes)):
            bounds.extend([(0, 2 * np.pi)] * ring_sizes[i])

        def ring_objective(vec):
            if time.time() > deadline - 30:
                return 1e10
            radii, angles_list = _unflatten_ring(vec, ring_sizes)
            radii = np.clip(radii, 0.0, 0.95)
            pts = _ring_to_points(radii, angles_list)
            return -_score(pts)

        # Use the best DE result as starting point if available
        x_start = x0
        if 'result_de' in dir() and result_de is not None:
            x_start = result_de.x

        try:
            result_lbfgs = minimize(ring_objective, x_start, method='L-BFGS-B', bounds=bounds,
                                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if time.time() < deadline - 25:
                x_opt = result_lbfgs.x
                radii_opt, angles_opt = _unflatten_ring(x_opt, ring_sizes)
                radii_opt = np.clip(radii_opt, 0.0, 0.95)
                pts_opt = _ring_to_points(radii_opt, angles_opt)
                update_best(pts_opt)
        except Exception:
            pass

    # --- Stage 3: Binary search for max radius using soft violation ---
    # This is the structural idea from inspiration 1: find the largest r for which a feasible packing exists
    if time.time() < deadline - 20 and best_points is not None:
        # Start from current best score as lower bound
        r_low = max(0.1, best_score)
        r_high = 0.135  # Slightly above known best
        C_best = best_points.copy()
        
        num_steps = 12
        for i in range(num_steps):
            if time.time() > deadline - 15:
                break
            r_mid = (r_low + r_high) / 2.0
            
            def soft_obj(x):
                C = x.reshape(N, 2)
                return _soft_violation(C, r_mid)
            
            limit = 1.0 - r_mid
            bounds = [(-limit, limit)] * (2 * N)
            
            try:
                res = minimize(soft_obj, C_best.flatten(), method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
                if time.time() > deadline - 12:
                    break
                C_opt = res.x.reshape(N, 2)
                val = res.fun
                if val < 1e-8:
                    r_low = r_mid
                    C_best = C_opt
                else:
                    r_high = r_mid
            except Exception:
                r_high = r_mid
        
        # Update best with the result of binary search
        if best_points is not None:
            update_best(C_best)

    # --- Stage 4: Full point-based Nelder-Mead refinement ---
    if time.time() < deadline - 10 and best_points is not None:
        def point_objective(vec):
            if time.time() > deadline - 10:
                return 1e10
            pts = vec.reshape(N, 2)
            s = _score(pts)
            return -s

        x0_pts = best_points.flatten()
        try:
            result3 = minimize(point_objective, x0_pts, method='Nelder-Mead',
                               options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if time.time() < deadline - 5:
                pts_opt3 = result3.x.reshape(N, 2)
                update_best(pts_opt3)
        except Exception:
            pass

    # --- Stage 5: Final tight polish ---
    if time.time() < deadline - 3 and best_points is not None:
        def point_objective_tight(vec):
            if time.time() > deadline - 3:
                return 1e10
            pts = vec.reshape(N, 2)
            s = _score(pts)
            return -s

        x0_final = best_points.flatten()
        try:
            result4 = minimize(point_objective_tight, x0_final, method='Nelder-Mead',
                               options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            if time.time() < deadline - 1:
                pts_opt4 = result4.x.reshape(N, 2)
                update_best(pts_opt4)
        except Exception:
            pass

    # Fallback: hexagonal lattice
    if best_points is None or best_score < 0.05:
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
            d = np.linalg.norm(points, axis=1)
            points = points[np.argsort(d, kind='stable')[:N]]
        best_points = points

    return best_points
# EVOLVE-BLOCK-END
