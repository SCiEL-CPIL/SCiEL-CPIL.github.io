"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 26

_start_time = time.time()
_TIME_LIMIT = 145  # seconds, safety margin before 150s kill


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _TIME_LIMIT - _elapsed()


def _validate(xyz):
    """Check if configuration is valid: circles inside unit square and non-overlapping."""
    n = xyz.shape[0]
    x = xyz[:, 0]
    y = xyz[:, 1]
    r = xyz[:, 2]
    if np.any(r < -1e-9):
        return False
    if np.any(x - r < -1e-7) or np.any(x + r > 1 + 1e-7):
        return False
    if np.any(y - r < -1e-7) or np.any(y + r > 1 + 1e-7):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-7:
                return False
    return True


def _enforce_validity(params):
    """
    Enforce all constraints by shrinking radii and clipping positions.
    Returns a valid configuration.
    """
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)

    # Clip radii to fit within walls
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    # Iteratively resolve overlaps
    iu = np.triu_indices(N, k=1)
    for _ in range(300):
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 0
        if not np.any(active):
            break
        for k in range(len(active)):
            if not active[k]:
                continue
            i = iu[0][k]
            j = iu[1][k]
            ov = overlap[k]
            if p[i, 2] <= p[j, 2]:
                shrink = min(ov, p[i, 2] - 1e-6)
                p[i, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[j, 2] - 1e-6)
                    p[j, 2] -= shrink2
            else:
                shrink = min(ov, p[j, 2] - 1e-6)
                p[j, 2] -= shrink
                rem = ov - shrink
                if rem > 0:
                    shrink2 = min(rem, p[i, 2] - 1e-6)
                    p[i, 2] -= shrink2
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))

    return p


def _voronoi_radii(centers, tol=1e-8):
    """
    Compute the maximum radius for each circle such that circles don't overlap
    and stay within the unit square, using Voronoi cell distances.
    This inherently ensures non-overlap.
    """
    n = len(centers)
    x = centers[:, 0]
    y = centers[:, 1]
    radii = np.zeros(n)

    # Distance to walls
    wall_dist = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))

    # Compute Voronoi diagram
    try:
        vor = Voronoi(centers)
    except Exception:
        # Fallback: just use wall distances
        radii = np.maximum(wall_dist, 1e-6)
        return radii

    # For each circle, find the minimum distance to its Voronoi cell edges
    for i in range(n):
        # Find the region for point i
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]

        if len(region) == 0 or -1 in region:
            # Unbounded region, use wall distance
            radii[i] = wall_dist[i]
            continue

        min_dist = wall_dist[i]

        # For each edge in the region, compute distance to the edge line
        for edge_idx in region:
            if edge_idx < 0:
                continue
            edge = vor.ridge_points[edge_idx]
            v1 = vor.vertices[edge[0]]
            v2 = vor.vertices[edge[1]]

            # Distance from center i to the line through v1, v2
            dx = v2[0] - v1[0]
            dy = v2[1] - v1[1]
            length = np.sqrt(dx * dx + dy * dy)
            if length < 1e-10:
                continue

            # Vector from v1 to center i
            cx = x[i] - v1[0]
            cy = y[i] - v1[1]

            # Distance to the infinite line
            dist_to_line = abs(dx * cy - dy * cx) / length

            # Check if the closest point is within the segment
            t = (cx * dx + cy * dy) / (length * length)
            if 0 <= t <= 1:
                min_dist = min(min_dist, dist_to_line)
            else:
                # Closest point is an endpoint
                d1 = np.sqrt((x[i] - v1[0])**2 + (y[i] - v1[1])**2)
                d2 = np.sqrt((x[i] - v2[0])**2 + (y[i] - v2[1])**2)
                min_dist = min(min_dist, d1, d2)

        radii[i] = max(min_dist, 1e-6)

    return radii


def _voronoi_objective(centers, penalty=0.0):
    """
    Objective: maximize sum of Voronoi-constrained radii.
    Returns (-sum_radii, gradient) for minimization.
    """
    n = len(centers)
    x = centers[:, 0]
    y = centers[:, 1]

    radii = _voronoi_radii(centers)
    val = -np.sum(radii)

    # Numerical gradient
    grad = np.zeros_like(centers)
    eps = 1e-6
    for i in range(n):
        for j in range(2):
            orig = centers[i, j]
            centers[i, j] = orig + eps
            r_plus = _voronoi_radii(centers)
            centers[i, j] = orig - eps
            r_minus = _voronoi_radii(centers)
            centers[i, j] = orig
            grad[i, j] = (-np.sum(r_plus) + np.sum(r_minus)) / (2 * eps)

    return val, grad


def _optimize_voronoi(centers, time_budget=5.0):
    """
    Optimize center positions to maximize sum of Voronoi-constrained radii.
    """
    start_time = time.time()
    best_centers = centers.copy()
    best_val, _ = _voronoi_objective(best_centers)
    best_radii = _voronoi_radii(best_centers)
    best_score = np.sum(best_radii)

    # Use L-BFGS-B with numerical gradient
    for trial in range(3):
        if time.time() - start_time > time_budget:
            break

        def obj(c):
            v, _ = _voronoi_objective(c)
            return v

        def grad(c):
            _, g = _voronoi_objective(c)
            return g

        try:
            bounds = [(0, 1), (0, 1)] * N
            result = minimize(obj, best_centers, jac=grad, method='L-BFGS-B',
                             bounds=bounds,
                             options={'maxiter': 2000, 'ftol': 1e-14,
                                      'gtol': 1e-12, 'maxfun': 5000})
            if result.success or result.fun < best_val:
                new_centers = result.x
                new_radii = _voronoi_radii(new_centers)
                new_score = np.sum(new_radii)
                if new_score > best_score:
                    best_score = new_score
                    best_centers = new_centers
                    best_radii = new_radii
                    best_val = result.fun
        except Exception:
            pass

    return best_centers, best_radii, best_score


def _physics_relax(xyz, time_limit=5.0, grow_rate=0.0005):
    """
    Force-directed physics relaxation that also grows radii over time.
    """
    start_time = time.time()
    xyz = xyz.copy().astype(float)
    x = xyz[:, 0].copy()
    y = xyz[:, 1].copy()
    r = xyz[:, 2].copy()

    best_score = np.sum(r)
    best_xyz = xyz.copy()
    last_improvement = 0.0

    dt = 0.005
    current_grow = grow_rate

    for iteration in range(8000):
        if time.time() - start_time > time_limit:
            break

        if iteration % 100 == 0:
            if (time.time() - start_time) - last_improvement > 2.0:
                current_grow *= 0.7
            else:
                current_grow = grow_rate

        r *= (1.0 + current_grow)

        wall_max = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        r = np.minimum(r, wall_max)
        r = np.maximum(r, 1e-6)

        fx = np.zeros(N)
        fy = np.zeros(N)

        idx_i, idx_j = np.triu_indices(N, k=1)
        dx = x[idx_i] - x[idx_j]
        dy = y[idx_i] - y[idx_j]
        dist_sq = dx * dx + dy * dy
        dist = np.sqrt(np.maximum(dist_sq, 1e-12))
        min_dist = r[idx_i] + r[idx_j]
        overlap = min_dist - dist
        active = overlap > 0

        if np.any(active):
            force = np.zeros(len(active))
            force[active] = overlap[active] * 80.0
            inv_dist = np.where(dist > 1e-10, 1.0 / dist, 0.0)

            np.add.at(fx, idx_i, force * dx * inv_dist)
            np.add.at(fx, idx_j, -force * dx * inv_dist)
            np.add.at(fy, idx_i, force * dy * inv_dist)
            np.add.at(fy, idx_j, -force * dy * inv_dist)

        for i in range(N):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 150.0
            if x[i] > 1 - r[i]:
                fx[i] -= (x[i] - (1 - r[i])) * 150.0
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 150.0
            if y[i] > 1 - r[i]:
                fy[i] -= (y[i] - (1 - r[i])) * 150.0

        x += dt * fx
        y += dt * fy

        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])

        xyz[:, 0] = x
        xyz[:, 1] = y
        xyz[:, 2] = r

        if _validate(xyz):
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_xyz = xyz.copy()
                last_improvement = time.time() - start_time

    return best_xyz


def _softplus(z, eps=1e-4):
    return np.logaddexp(0, z / eps) * eps


def _sigmoid(z, eps=1e-4):
    return 1.0 / (1.0 + np.exp(-z / eps))


def _objective_grad(params, penalty_weight, smooth_eps=1e-4):
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]

    val = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0

    lx = r - x
    ux = x + r - 1.0
    ly = r - y
    uy = y + r - 1.0

    sp_lx = _softplus(lx, smooth_eps)
    sg_lx = _sigmoid(lx, smooth_eps)
    grad[0::3] += -2.0 * penalty_weight * sp_lx * sg_lx
    grad[2::3] += 2.0 * penalty_weight * sp_lx * sg_lx

    sp_ux = _softplus(ux, smooth_eps)
    sg_ux = _sigmoid(ux, smooth_eps)
    grad[0::3] += 2.0 * penalty_weight * sp_ux * sg_ux
    grad[2::3] += 2.0 * penalty_weight * sp_ux * sg_ux

    sp_ly = _softplus(ly, smooth_eps)
    sg_ly = _sigmoid(ly, smooth_eps)
    grad[1::3] += -2.0 * penalty_weight * sp_ly * sg_ly
    grad[2::3] += 2.0 * penalty_weight * sp_ly * sg_ly

    sp_uy = _softplus(uy, smooth_eps)
    sg_uy = _sigmoid(uy, smooth_eps)
    grad[1::3] += 2.0 * penalty_weight * sp_uy * sg_uy
    grad[2::3] += 2.0 * penalty_weight * sp_uy * sg_uy

    sp_nr = _softplus(-r, smooth_eps)
    sg_nr = _sigmoid(-r, smooth_eps)
    grad[2::3] += -2.0 * penalty_weight * sp_nr * sg_nr

    val += penalty_weight * (np.sum(sp_lx**2) + np.sum(sp_ux**2) +
                             np.sum(sp_ly**2) + np.sum(sp_uy**2) +
                             np.sum(sp_nr**2))

    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    sp_ov = _softplus(overlap, smooth_eps)
    sg_ov = _sigmoid(overlap, smooth_eps)

    factor = 2.0 * penalty_weight * sp_ov * sg_ov
    inv_dist = 1.0 / dist

    np.add.at(grad[0::3], idx_i, -factor * dx * inv_dist)
    np.add.at(grad[0::3], idx_j, factor * dx * inv_dist)
    np.add.at(grad[1::3], idx_i, -factor * dy * inv_dist)
    np.add.at(grad[1::3], idx_j, factor * dy * inv_dist)
    np.add.at(grad[2::3], idx_i, factor)
    np.add.at(grad[2::3], idx_j, factor)

    val += penalty_weight * np.sum(sp_ov**2)

    return val, grad


def _polish_analytic(xyz, time_budget=5.0):
    start_time = time.time()
    params = xyz.flatten().copy()
    best_params = params.copy()
    best_val, _ = _objective_grad(best_params, 1000.0)

    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((1e-6, 0.5))

    for penalty_weight in [500.0, 2000.0, 10000.0, 50000.0, 200000.0]:
        if time.time() - start_time > time_budget:
            break

        def obj(p):
            v, _ = _objective_grad(p, penalty_weight)
            return v

        def grad(p):
            _, g = _objective_grad(p, penalty_weight)
            return g

        try:
            result = minimize(obj, best_params, jac=grad, method='L-BFGS-B',
                             bounds=bounds,
                             options={'maxiter': 5000, 'ftol': 1e-16,
                                      'gtol': 1e-14, 'maxfun': 30000})
            result_xyz = result.x.reshape(N, 3)
            if _validate(result_xyz):
                obj_val, _ = _objective_grad(result.x, penalty_weight)
                if obj_val < best_val:
                    best_params = result.x
                    best_val = obj_val
        except Exception:
            pass

    return best_params.reshape(N, 3)


def _initial_config_grid():
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.9
            cells.append([x, y, r])
    return np.array(cells[:N])


def _initial_config_hex():
    side = 5
    positions = []
    for i in range(side):
        for j in range(side):
            if j % 2 == 1:
                x = (i + 0.5) / side + 0.1 / side
            else:
                x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.95
            positions.append([x, y, r])
    positions.append([0.5, 0.5, 0.5 / side * 0.5])
    return np.array(positions[:N])


def _initial_config_corner():
    rng = np.random.RandomState(123)
    circles = np.zeros((N, 3))

    corners = [[0.15, 0.15], [0.85, 0.15], [0.15, 0.85], [0.85, 0.85]]
    for i, (cx, cy) in enumerate(corners):
        circles[i, 0] = cx
        circles[i, 1] = cy
        circles[i, 2] = 0.14

    edges = [[0.5, 0.12], [0.5, 0.88], [0.12, 0.5], [0.88, 0.5]]
    for i, (cx, cy) in enumerate(edges):
        idx = 4 + i
        circles[idx, 0] = cx
        circles[idx, 1] = cy
        circles[idx, 2] = 0.12

    remaining = 12
    for i in range(remaining):
        angle = 2 * np.pi * i / remaining
        cx = 0.5 + 0.25 * np.cos(angle)
        cy = 0.5 + 0.25 * np.sin(angle)
        idx = 8 + i
        circles[idx, 0] = cx
        circles[idx, 1] = cy
        circles[idx, 2] = 0.10

    circles[:, 0] += 0.005 * rng.randn(N)
    circles[:, 1] += 0.005 * rng.randn(N)
    circles[:, 0] = np.clip(circles[:, 0], 0.05, 0.95)
    circles[:, 1] = np.clip(circles[:, 1], 0.05, 0.95)

    return circles


def construct():
    global_start = time.time()
    deadline = 140.0

    best_xyz = _initial_config_grid()
    best_score = np.sum(best_xyz[:, 2])

    initial_configs = []
    initial_configs.append(_initial_config_grid())
    initial_configs.append(_initial_config_hex())
    initial_configs.append(_initial_config_corner())

    np.random.seed(42)
    for trial in range(4):
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.05, 0.15, N)
        for i in range(N):
            x[i] = np.clip(x[i], r[i], 1 - r[i])
            y[i] = np.clip(y[i], r[i], 1 - r[i])
        initial_configs.append(np.column_stack([x, y, r]))

    positions = []
    positions.append([0.5, 0.5, 0.10])
    for k in range(8):
        angle = 2 * np.pi * k / 8
        x = 0.5 + 0.12 * 1.8 * np.cos(angle)
        y = 0.5 + 0.12 * 1.8 * np.sin(angle)
        positions.append([x, y, 0.12 * 0.9])
    for k in range(10):
        angle = 2 * np.pi * k / 10 + np.pi / 10
        x = 0.5 + 0.10 * 3.5 * np.cos(angle)
        y = 0.5 + 0.10 * 3.5 * np.sin(angle)
        positions.append([x, y, 0.10 * 0.9])
    for k in range(8):
        angle = 2 * np.pi * k / 8
        x = 0.5 + 0.08 * 4.8 * np.cos(angle)
        y = 0.5 + 0.08 * 4.8 * np.sin(angle)
        positions.append([x, y, 0.08 * 0.9])

    initial_configs.append(np.array(positions[:N]))

    for idx, xyz0 in enumerate(initial_configs):
        elapsed = time.time() - global_start
        remaining = deadline - elapsed
        if remaining < 12.0:
            break

        # Phase 1: Physics relaxation
        phys_time = min(4.0, remaining * 0.30)
        if phys_time < 1.5:
            break

        try:
            xyz_phys = _physics_relax(xyz0, time_limit=phys_time)

            # Phase 2: Voronoi-based optimization
            elapsed = time.time() - global_start
            remaining = deadline - elapsed
            vor_time = min(5.0, remaining * 0.35)
            if vor_time > 1.0:
                centers = xyz_phys[:, :2].copy()
                vor_centers, vor_radii, vor_score = _optimize_voronoi(centers, time_budget=vor_time)
                vor_xyz = np.column_stack([vor_centers, vor_radii])
                if _validate(vor_xyz) and vor_score > np.sum(xyz_phys[:, 2]):
                    xyz_phys = vor_xyz

            # Phase 3: Analytic gradient polish
            elapsed = time.time() - global_start
            remaining = deadline - elapsed
            polish_time = min(4.0, remaining * 0.30)
            if polish_time > 1.0:
                xyz_opt = _polish_analytic(xyz_phys, time_budget=polish_time)
                if _validate(xyz_opt):
                    xyz_phys = xyz_opt

            # Validate and fix
            if not _validate(xyz_phys):
                xyz_phys = _enforce_validity(xyz_phys)

            score = np.sum(xyz_phys[:, 2])
            if score > best_score:
                best_score = score
                best_xyz = xyz_phys
        except Exception:
            pass

    # Final high-precision polish
    elapsed = time.time() - global_start
    remaining = deadline - elapsed
    if remaining > 3.0:
        try:
            # Try Voronoi optimization on best result
            vor_time = min(3.0, remaining * 0.4)
            if vor_time > 1.0:
                centers = best_xyz[:, :2].copy()
                vor_centers, vor_radii, vor_score = _optimize_voronoi(centers, time_budget=vor_time)
                vor_xyz = np.column_stack([vor_centers, vor_radii])
                if _validate(vor_xyz) and vor_score > best_score:
                    best_xyz = vor_xyz
                    best_score = vor_score

            # Final analytic polish
            elapsed = time.time() - global_start
            remaining = deadline - elapsed
            if remaining > 2.0:
                xyz_final = _polish_analytic(best_xyz, time_budget=min(3.0, remaining * 0.5))
                if _validate(xyz_final) and np.sum(xyz_final[:, 2]) > best_score:
                    best_xyz = xyz_final
                    best_score = np.sum(best_xyz[:, 2])
        except Exception:
            pass

    if not _validate(best_xyz):
        best_xyz = _enforce_validity(best_xyz)

    if not _validate(best_xyz):
        best_xyz = _initial_config_grid()

    return best_xyz
