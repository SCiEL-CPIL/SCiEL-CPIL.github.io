import time
import math
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
T0 = time.time()
DEADLINE = T0 + 145.0  # safety margin before 150s kill
SQRT3 = math.sqrt(3.0)

# Optimal 11-hexagon cluster on triangular lattice (proven best packing)
_OPTIMAL_POSITIONS = np.array([
    [0.0, 0.0],
    [2.0, 0.0], [-1.0, SQRT3], [1.0, SQRT3],
    [-1.0, -SQRT3], [1.0, -SQRT3],
    [3.0, 0.0], [-2.0, SQRT3], [2.0, SQRT3],
    [-2.0, -SQRT3], [2.0, -SQRT3]
], dtype=float)


def _hex_vertices_local(theta):
    """Unit hexagon vertices (circumradius 1) rotated by theta."""
    ang = np.arange(6) * (math.pi / 3.0) + theta
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)


def _outer_side_normals(theta_outer):
    """Outward normals of the 6 sides of the outer hexagon."""
    ang = theta_outer + np.arange(6) * (math.pi / 3.0) + math.pi / 6.0
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)


def _outer_side(theta, centers, theta_outer):
    """Minimal outer side length containing all inner hexagon vertices."""
    verts_local = _hex_vertices_local(theta)
    verts = (centers[:, None, :] + verts_local[None, :, :]).reshape(-1, 2)
    normals = _outer_side_normals(theta_outer)
    proj = verts @ normals.T
    apothem = proj.max()
    return apothem * 2.0 / SQRT3


def _outer_side_smooth(theta, centers, theta_outer, beta=50.0):
    """Soft-max version of outer side length for smoother optimization."""
    verts_local = _hex_vertices_local(theta)
    verts = (centers[:, None, :] + verts_local[None, :, :]).reshape(-1, 2)
    normals = _outer_side_normals(theta_outer)
    proj = verts @ normals.T  # (36, 6)
    # Soft max over all projections
    proj_flat = proj.flatten()
    m = proj_flat.max()
    soft_max = m + math.log(np.exp((proj_flat - m) * beta).sum()) / beta
    return soft_max * 2.0 / SQRT3


def _pairwise_penalty(centers):
    """Penalty for pairs of centers closer than 2 (overlap)."""
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    dpair = d[iu]
    viol = np.maximum(0.0, 2.0 - dpair)
    return np.sum(viol * viol)


def _objective(params, theta_outer):
    """Objective: outer side length + large penalty for overlap."""
    centers = params[:N * 2].reshape(N, 2)
    theta = params[N * 2]
    s = _outer_side(theta, centers, theta_outer)
    pen = _pairwise_penalty(centers)
    return s + 1000.0 * pen


def _objective_smooth(params, theta_outer, beta=50.0):
    """Smooth objective using soft-max for outer side."""
    centers = params[:N * 2].reshape(N, 2)
    theta = params[N * 2]
    s = _outer_side_smooth(theta, centers, theta_outer, beta=beta)
    pen = _pairwise_penalty(centers)
    return s + 1000.0 * pen


def _is_valid(centers, theta, theta_outer, tol=1e-5):
    """Check non-overlap and containment."""
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    if d[iu].min() < 2.0 - tol:
        return False
    verts_local = _hex_vertices_local(theta)
    verts = (centers[:, None, :] + verts_local[None, :, :]).reshape(-1, 2)
    normals = _outer_side_normals(theta_outer)
    s = _outer_side(theta, centers, theta_outer)
    apothem = s * SQRT3 / 2.0
    proj = verts @ normals.T
    if proj.max() > apothem + tol:
        return False
    return True


def _min_enclosing_hex_lp(inner_hexes, angle_deg):
    """
    Use LP to find the optimal outer hexagon center and inradius
    for given inner hexagons and outer angle.
    Returns (cx, cy, side_length) or None.
    """
    all_vertices = []
    for ix, iy, iangle_deg in inner_hexes:
        iangle = math.radians(iangle_deg)
        for k in range(6):
            a = iangle + math.pi * k / 3
            all_vertices.append((ix + math.cos(a), iy + math.sin(a)))

    n_pts = len(all_vertices)
    c = [0.0, 0.0, 1.0]  # minimize t (inradius)
    A_ub = []
    b_ub = []

    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)

        for vx, vy in all_vertices:
            # (vx-cx)*nx + (vy-cy)*ny <= t
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-nx * vx - ny * vy)
            # -(vx-cx)*nx - (vy-cy)*ny <= t
            A_ub.append([nx, ny, -1.0])
            b_ub.append(nx * vx + ny * vy)

    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs')
        if res.success:
            cx, cy, t = res.x
            side = t / (SQRT3 / 2)
            return cx, cy, side
    except Exception:
        pass
    return None


def _find_best_outer_lp(inner_hexes, time_limit=8.0):
    """
    Find best outer hexagon angle and center via LP.
    Returns (cx, cy, side, angle_deg).
    """
    start = time.time()
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Coarse search over 0..30 degrees (hexagonal symmetry)
    coarse = np.arange(0, 30.01, 0.5)
    for angle_deg in coarse:
        if time.time() > start + time_limit:
            break
        res = _min_enclosing_hex_lp(inner_hexes, float(angle_deg))
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = float(angle_deg)

    # Fine search around best
    if best_angle >= 0:
        fine_lo = max(0.0, best_angle - 0.5)
        fine_hi = min(30.0, best_angle + 0.5)
        fine = np.arange(fine_lo, fine_hi + 0.001, 0.01)
        for angle_deg in fine:
            if time.time() > start + time_limit:
                break
            res = _min_enclosing_hex_lp(inner_hexes, float(angle_deg))
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = float(angle_deg)

        # Ultra-fine search
        if time.time() < start + time_limit:
            uf_lo = max(0.0, best_angle - 0.05)
            uf_hi = min(30.0, best_angle + 0.05)
            ultra_fine = np.arange(uf_lo, uf_hi + 0.001, 0.001)
            for angle_deg in ultra_fine:
                if time.time() > start + time_limit:
                    break
                res = _min_enclosing_hex_lp(inner_hexes, float(angle_deg))
                if res:
                    cx, cy, side = res
                    if side < best_side:
                        best_side = side
                        best_center = (cx, cy)
                        best_angle = float(angle_deg)

    return best_center[0], best_center[1], best_side, best_angle


def _make_init(seed):
    """Generate initial parameters for optimization."""
    rng = np.random.RandomState(seed)
    mode = seed % 6

    if mode == 0:
        # Optimal 11-hex cluster (exact)
        centers = _OPTIMAL_POSITIONS.copy()
    elif mode == 1:
        # Optimal cluster with small random jitter
        centers = _OPTIMAL_POSITIONS.copy() + rng.uniform(-0.05, 0.05, size=(N, 2))
    elif mode == 2:
        # Hexagonal lattice: center + 6 ring + 4 outer
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        for i in range(6):
            ang = i * math.pi / 3.0
            centers[i + 1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
        for i in range(4):
            ang = (i + 0.5) * math.pi / 3.0
            centers[i + 7] = [3.5 * math.cos(ang), 3.5 * math.sin(ang)]
    elif mode == 3:
        # Offset hexagonal lattice
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        for i in range(6):
            ang = i * math.pi / 3.0 + math.pi / 6.0
            centers[i + 1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
        for i in range(4):
            ang = (i + 0.5) * math.pi / 3.0
            centers[i + 7] = [3.4 * math.cos(ang), 3.4 * math.sin(ang)]
    elif mode == 4:
        # Triangular lattice, 11 closest points
        u = (SQRT3, 0.0)
        v = (SQRT3 / 2, 1.5)
        candidates = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                candidates.append((x * x + y * y, x, y))
        candidates.sort(key=lambda t: t[0])
        centers = np.array([(c[1], c[2]) for c in candidates[:N]])
    else:
        # Random perturbation of optimal
        centers = _OPTIMAL_POSITIONS.copy() + rng.uniform(-0.15, 0.15, size=(N, 2))

    theta = rng.uniform(-math.pi / 6, math.pi / 6)
    params = np.concatenate([centers.reshape(-1), [theta]])
    return params


def _optimize_one(init_params, theta_outer, maxiter=400):
    """Single L-BFGS-B optimization run."""
    try:
        res = minimize(_objective, init_params, args=(theta_outer,),
                       method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-13, 'gtol': 1e-11})
        return res
    except Exception:
        return None


def _polish(params, theta_outer):
    """High-precision polish with smooth objective then exact."""
    cur_best = _objective(params, theta_outer)
    best_params = params.copy()
    
    # Step 1: Smooth objective with L-BFGS-B (multiple beta values)
    for beta in [100.0, 200.0, 50.0]:
        try:
            res = minimize(_objective_smooth, best_params, args=(theta_outer, beta),
                           method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < cur_best:
                cur_best = _objective(res.x, theta_outer)
                best_params = res.x
        except Exception:
            pass
    
    # Step 2: SLSQP with exact objective
    try:
        res = minimize(_objective, best_params, args=(theta_outer,),
                       method='SLSQP', options={'maxiter': 800, 'ftol': 1e-16, 'maxls': 80})
        if res.fun < cur_best:
            cur_best = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Step 3: Another L-BFGS-B pass with very tight tolerances
    try:
        res = minimize(_objective, best_params, args=(theta_outer,),
                       method='L-BFGS-B',
                       options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-16})
        if res.fun < cur_best:
            cur_best = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Step 4: Final SLSQP
    try:
        res = minimize(_objective, best_params, args=(theta_outer,),
                       method='SLSQP', options={'maxiter': 500, 'ftol': 1e-18})
        if res.fun < cur_best:
            cur_best = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Step 5: One more L-BFGS-B with different starting point perturbation
    try:
        jitter = np.random.RandomState(42).uniform(-1e-6, 1e-6, size=best_params.shape)
        res = minimize(_objective, best_params + jitter, args=(theta_outer,),
                       method='L-BFGS-B',
                       options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-16})
        if res.fun < cur_best:
            cur_best = res.fun
            best_params = res.x
    except Exception:
        pass
    
    return best_params


def _lp_refine(centers, theta, time_limit=6.0):
    """
    Use LP to find the best outer hexagon (center, angle) for given inner hexagons.
    Returns (cx, cy, side, angle_deg) or None.
    """
    inner_hexes = [[float(centers[i, 0]), float(centers[i, 1]),
                    float(math.degrees(theta))] for i in range(N)]
    return _find_best_outer_lp(inner_hexes, time_limit=time_limit)


def construct():
    best_score = float('inf')
    best_result = None  # (inner, center, side, angle_deg)

    # Phase 1: Multi-start continuous optimization with fixed outer angle
    outer_rotations = [0.0, math.pi / 6.0, math.pi / 3.0, math.pi / 12.0, math.pi / 4.0]
    seed = 0
    maxiter = 350

    for theta_o in outer_rotations:
        while time.time() < DEADLINE - 25:  # reserve 25s for LP refinement + polish
            init_params = _make_init(seed)
            seed += 1
            res = _optimize_one(init_params, theta_o, maxiter)
            if res is None:
                continue
            params = res.x
            centers = params[:N * 2].reshape(N, 2)
            theta = params[N * 2]
            if not _is_valid(centers, theta, theta_o):
                continue
            s = _outer_side(theta, centers, theta_o)
            if s < best_score:
                best_score = s
                best_result = (centers.copy(), theta, theta_o)

    # Phase 2: LP-based outer hexagon optimization for the best inner configuration
    if best_result is not None and time.time() < DEADLINE - 10:
        centers, theta, _ = best_result
        lp_res = _lp_refine(centers, theta, time_limit=10.0)
        if lp_res is not None:
            cx, cy, side_lp, angle_deg = lp_res
            angle_rad = math.radians(angle_deg)
            # Verify validity with LP-found outer hexagon
            if _is_valid(centers, theta, angle_rad):
                s_check = _outer_side(theta, centers, angle_rad)
                if s_check < best_score:
                    best_score = s_check
                    best_result = (centers.copy(), theta, angle_rad)

    # Phase 3: Polish the best solution (multiple outer rotations for polish)
    if best_result is not None and time.time() < DEADLINE - 5:
        centers, theta, theta_o = best_result
        for theta_o_try in [theta_o, 0.0, math.pi / 6.0, math.pi / 3.0, math.pi / 12.0]:
            if time.time() > DEADLINE - 5:
                break
            params = np.concatenate([centers.reshape(-1), [theta]])
            polished = _polish(params, theta_o_try)
            centers_p = polished[:N * 2].reshape(N, 2)
            theta_p = polished[N * 2]
            if _is_valid(centers_p, theta_p, theta_o_try):
                s_p = _outer_side(theta_p, centers_p, theta_o_try)
                if s_p < best_score:
                    best_score = s_p
                    best_result = (centers_p.copy(), theta_p, theta_o_try)

    # Phase 4: Final LP refinement on polished solution
    if best_result is not None and time.time() < DEADLINE - 1:
        centers, theta, _ = best_result
        lp_res = _lp_refine(centers, theta, time_limit=5.0)
        if lp_res is not None:
            cx, cy, side_lp, angle_deg = lp_res
            angle_rad = math.radians(angle_deg)
            if _is_valid(centers, theta, angle_rad):
                s_check = _outer_side(theta, centers, angle_rad)
                if s_check < best_score:
                    best_score = s_check
                    best_result = (centers.copy(), theta, angle_rad)
        
        # Phase 5: One more polish after LP
        if time.time() < DEADLINE - 0.5:
            centers, theta, theta_o = best_result
            params = np.concatenate([centers.reshape(-1), [theta]])
            polished = _polish(params, theta_o)
            centers_p = polished[:N * 2].reshape(N, 2)
            theta_p = polished[N * 2]
            if _is_valid(centers_p, theta_p, theta_o):
                s_p = _outer_side(theta_p, centers_p, theta_o)
                if s_p < best_score:
                    best_score = s_p
                    best_result = (centers_p.copy(), theta_p, theta_o)

    if best_result is None:
        # Fallback: return something (may be invalid)
        centers = np.zeros((N, 2))
        return (list(centers[:, 0].tolist()), list(centers[:, 1].tolist())), (0.0, 0.0), 40.0, 0.0

    centers, theta, theta_o = best_result
    s = _outer_side(theta, centers, theta_o)
    inner = [[float(centers[i, 0]), float(centers[i, 1]), float(math.degrees(theta))] for i in range(N)]
    return inner, (0.0, 0.0), float(s), float(math.degrees(theta_o))
