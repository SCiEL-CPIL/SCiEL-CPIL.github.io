"""
Heilbronn triangles: maximize the smallest triangle area among 11 points.
Strategy: Direct coordinate optimization with hard-min Nelder-Mead,
multiple diverse starts, and a final global polish.
"""
import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of the equilateral triangle with side 1

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
N_TRIPLETS = len(_TRIPLETS)

DEADLINE = 145  # Safety margin from 150s limit


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)


def _all_triangle_areas(pts):
    """Compute all triangle areas (vectorized)."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    return np.abs(cross) * 0.5


def _soft_min_area(pts, beta):
    """Soft minimum of triangle areas using log-sum-exp (differentiable)."""
    areas = _all_triangle_areas(pts)
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, subtract max
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min


def _project_to_triangle(pts):
    """
    Project points onto the equilateral triangle with vertices
    V0=(0,0), V1=(1,0), V2=(0.5, S3H).
    Uses barycentric coordinate clipping.
    """
    # Convert to barycentric:
    # x = v1_x * b1 + v2_x * b2  where b0+b1+b2=1
    # Actually: P = b0*V0 + b1*V1 + b2*V2
    # x = b1*1 + b2*0.5
    # y = b2*S3H
    # So b2 = y/S3H, b1 = x - 0.5*b2 = x - 0.5*y/S3H, b0 = 1-b1-b2
    b2 = pts[:, 1] / S3H
    b1 = pts[:, 0] - 0.5 * b2
    b0 = 1.0 - b1 - b2

    # Clip to valid barycentric region: b0,b1,b2 >= 0
    b0 = np.maximum(b0, 0)
    b1 = np.maximum(b1, 0)
    b2 = np.maximum(b2, 0)

    # Normalize to sum to 1
    total = b0 + b1 + b2
    total = np.maximum(total, 1e-12)
    b0 /= total
    b1 /= total
    b2 /= total

    # Convert back to Cartesian
    x = b1 * V1[0] + b2 * V2[0]
    y = b1 * V1[1] + b2 * V2[1]
    return np.column_stack([x, y])


def construct():
    start_time = time.time()
    rng = np.random.default_rng(12345)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin=5):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # Generate diverse initial configurations
    init_configs = []

    # 1. Symmetric configuration: corners + midpoints + center + extras
    # Corners in Cartesian
    corners = np.array([V0, V1, V2])
    midpoints = np.array([(V0 + V1) / 2, (V1 + V2) / 2, (V2 + V0) / 2])
    center = (V0 + V1 + V2) / 3

    # Start with 3 corners, 3 midpoints, 1 center = 7 points
    # Need 4 more points
    # Place them at 1/3 and 2/3 along edges
    edge1_13 = V0 + (V1 - V0) / 3
    edge1_23 = V0 + 2 * (V1 - V0) / 3
    edge2_13 = V1 + (V2 - V1) / 3
    edge2_23 = V1 + 2 * (V2 - V1) / 3

    sym1 = np.vstack([corners, midpoints, [center, edge1_13, edge1_23, edge2_13]])
    init_configs.append(sym1)

    # Alternative symmetric: use 2/3 points on edges
    sym2 = np.vstack([corners, midpoints, [center, edge1_23, edge2_23, V2 + (V0 - V2) / 3]])
    init_configs.append(sym2)

    # 2. Grid-like configurations
    # Place points on a coarse grid within the triangle
    for grid_n in [3, 4]:
        u_vals = np.linspace(0.05, 0.95, grid_n)
        v_vals = np.linspace(0.05, 0.95, grid_n)
        grid_pts = []
        for u in u_vals:
            for v in v_vals:
                if u + v <= 0.95:
                    # Convert barycentric (u, v, 1-u-v) to Cartesian
                    x = v * 1.0 + (1 - u - v) * 0.5
                    y = (1 - u - v) * S3H
                    grid_pts.append([x, y])
        # Take up to 11 points
        if len(grid_pts) >= N:
            # Select 11 well-spread points
            grid_pts = np.array(grid_pts)
            # Simple selection: take evenly spaced indices
            idx = np.linspace(0, len(grid_pts) - 1, N, dtype=int)
            init_configs.append(grid_pts[idx])

    # 3. Random configurations
    for _ in range(40):
        # Random points in triangle
        r1 = rng.random(N)
        r2 = rng.random(N)
        # Rejection sampling or direct transformation
        # Use: if r1+r2 > 1, reflect
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        # Barycentric to Cartesian
        x = r2 * 1.0 + (1 - r1 - r2) * 0.5
        y = (1 - r1 - r2) * S3H
        pts = np.column_stack([x, y])
        init_configs.append(pts)

    # 4. Near-boundary configurations
    for _ in range(20):
        pts = np.zeros((N, 2))
        for i in range(N):
            edge = rng.integers(0, 3)
            t = rng.random()
            if edge == 0:  # V0-V1
                pts[i] = V0 + t * (V1 - V0)
            elif edge == 1:  # V1-V2
                pts[i] = V1 + t * (V2 - V1)
            else:  # V2-V0
                pts[i] = V2 + t * (V0 - V2)
        init_configs.append(pts)

    # Optimization loop
    max_starts = min(len(init_configs), 80)
    current_start = 0

    for pts0 in init_configs:
        if not _elapsed_ok(30):
            break
        if current_start >= max_starts:
            break
        current_start += 1

        current_pts = _project_to_triangle(pts0.copy())
        _update_best(current_pts)

        # Phase 1: L-BFGS-B with soft min (increasing beta)
        for beta in [20, 50, 100, 200, 500]:
            if not _elapsed_ok(15):
                break

            def objective_soft(x, beta=beta):
                pts = x.reshape(N, 2)
                pts = _project_to_triangle(pts)
                return -_soft_min_area(pts, beta)

            try:
                res = minimize(objective_soft, current_pts.flatten(), method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13})
                current_pts = _project_to_triangle(res.x.reshape(N, 2))
                _update_best(current_pts)
            except Exception:
                pass

        # Phase 2: Nelder-Mead on hard minimum
        if _elapsed_ok(10):
            def objective_hard(x):
                pts = x.reshape(N, 2)
                pts = _project_to_triangle(pts)
                return -_min_triangle_area(pts)

            try:
                res_nm = minimize(objective_hard, current_pts.flatten(), method='Nelder-Mead',
                                  options={'maxiter': 800, 'xatol': 1e-12, 'fatol': 1e-14,
                                           'adaptive': True})
                current_pts = _project_to_triangle(res_nm.x.reshape(N, 2))
                _update_best(current_pts)
            except Exception:
                pass

        # Phase 3: Random perturbation + quick re-optimization
        if _elapsed_ok(12) and current_start <= max_starts // 2:
            for _ in range(3):
                if not _elapsed_ok(8):
                    break
                perturbed = current_pts + rng.normal(0, 0.03, size=(N, 2))
                perturbed = _project_to_triangle(perturbed)

                def quick_obj(x):
                    pts = x.reshape(N, 2)
                    pts = _project_to_triangle(pts)
                    return -_soft_min_area(pts, 200)

                try:
                    res_q = minimize(quick_obj, perturbed.flatten(), method='L-BFGS-B',
                                     options={'maxiter': 100, 'ftol': 1e-14})
                    new_pts = _project_to_triangle(res_q.x.reshape(N, 2))
                    _update_best(new_pts)
                    if _min_triangle_area(new_pts) > _min_triangle_area(current_pts):
                        current_pts = new_pts
                except Exception:
                    pass

    # Final global polish on the best found
    if best_pts is not None and _elapsed_ok(8):
        def final_obj(x):
            pts = x.reshape(N, 2)
            pts = _project_to_triangle(pts)
            return -_min_triangle_area(pts)

        try:
            res_final = minimize(final_obj, best_pts.flatten(), method='Nelder-Mead',
                                 options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-16,
                                          'adaptive': True})
            final_pts = _project_to_triangle(res_final.x.reshape(N, 2))
            _update_best(final_pts)
        except Exception:
            pass

    # Additional epigraph-style polish: try to push the minimum area up
    if best_pts is not None and _elapsed_ok(5):
        # Identify the critical triangle(s)
        areas = _all_triangle_areas(best_pts)
        min_idx = np.where(np.isclose(areas, areas.min(), atol=1e-10))[0]
        # Try small adjustments to points in the critical triangles
        for idx in min_idx[:5]:
            if not _elapsed_ok(3):
                break
            i, j, k = _TRIPLETS[idx]
            # Try moving each of the 3 points slightly
            for p_idx in [i, j, k]:
                for direction in range(4):
                    delta = np.zeros((N, 2))
                    if direction == 0:
                        delta[p_idx] = [0.01, 0]
                    elif direction == 1:
                        delta[p_idx] = [-0.01, 0]
                    elif direction == 2:
                        delta[p_idx] = [0, 0.01]
                    else:
                        delta[p_idx] = [0, -0.01]
                    trial = _project_to_triangle(best_pts + delta)
                    _update_best(trial)

    if best_pts is None:
        # Fallback: random points
        r1 = rng.random(N)
        r2 = rng.random(N)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        x = r2 * 1.0 + (1 - r1 - r2) * 0.5
        y = (1 - r1 - r2) * S3H
        best_pts = np.column_stack([x, y])

    return best_pts
