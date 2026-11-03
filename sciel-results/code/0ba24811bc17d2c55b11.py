import time
import math
import numpy as np

N = 11
T0 = time.time()
DEADLINE = T0 + 140.0  # safety margin before 150s kill
SQRT3 = math.sqrt(3.0)
PI = math.pi


def _hex_vertices_local(theta):
    """Unit hexagon vertices (circumradius 1) rotated by theta. (6,2)"""
    ang = np.arange(6) * (PI / 3.0) + theta
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)


def _outer_side_normals(theta_outer):
    """Outward normals of the 6 sides of the outer hexagon. (6,2)"""
    ang = theta_outer + np.arange(6) * (PI / 3.0) + PI / 6.0
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)


def _compute_side(centers, theta, theta_outer):
    """Compute the minimal outer side length for given centers and rotations."""
    verts_local = _hex_vertices_local(theta)  # (6,2)
    verts = centers[:, None, :] + verts_local[None, :, :]  # (N,6,2)
    verts = verts.reshape(-1, 2)  # (N*6, 2)
    normals = _outer_side_normals(theta_outer)  # (6,2)
    proj = verts @ normals.T  # (N*6, 6)
    apothem = proj.max()
    return apothem * 2.0 / SQRT3


def _pairwise_penalty(centers, weight=1000.0):
    """Penalty for pairs closer than 2 (minimum distance for non-overlap)."""
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    dpair = d[iu]
    viol = np.maximum(0.0, 2.0 - dpair)
    return weight * np.sum(viol * viol)


def _objective(params, theta_outer):
    """
    Objective: minimize outer side length + penalty for overlaps.
    params: (N*2 + 1) = [x0,y0,...,x_{N-1},y_{N-1}, theta]
    All inner hexagons share the same rotation theta.
    """
    centers = params[:N * 2].reshape(N, 2)
    theta = params[N * 2]
    s = _compute_side(centers, theta, theta_outer)
    pen = _pairwise_penalty(centers)
    return s + pen


def _is_valid(centers, theta, theta_outer, tol=1e-5):
    """Check non-overlap and containment."""
    # Non-overlap
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    if d[iu].min() < 2.0 - tol:
        return False
    # Containment
    s = _compute_side(centers, theta, theta_outer)
    apothem = s * SQRT3 / 2.0
    verts_local = _hex_vertices_local(theta)
    verts = (centers[:, None, :] + verts_local[None, :, :]).reshape(-1, 2)
    normals = _outer_side_normals(theta_outer)
    proj = verts @ normals.T
    if proj.max() > apothem + tol:
        return False
    return True


def _make_init(seed):
    """Generate diverse initial configurations."""
    rng = np.random.RandomState(seed)
    mode = seed % 6

    if mode == 0:
        # Hexagonal lattice: center + ring of 6 + 4 more
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        for i in range(6):
            ang = i * PI / 3.0
            centers[i + 1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
        for i in range(4):
            ang = (i + 0.5) * PI / 3.0
            centers[i + 7] = [3.5 * math.cos(ang), 3.5 * math.sin(ang)]
    elif mode == 1:
        # Offset hexagonal lattice
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        for i in range(6):
            ang = i * PI / 3.0 + PI / 6.0
            centers[i + 1] = [2.0 * math.cos(ang), 2.0 * math.sin(ang)]
        for i in range(4):
            ang = (i + 0.5) * PI / 3.0
            centers[i + 7] = [3.4 * math.cos(ang), 3.4 * math.sin(ang)]
    elif mode == 2:
        # Compact cluster (known good packing)
        centers = np.zeros((N, 2))
        positions = [
            (0, 0), (2, 0), (-1, 1.732), (1, 1.732),
            (-1, -1.732), (1, -1.732), (3, 0),
            (-2, 1.732), (2, 1.732), (-2, -1.732), (2, -1.732)
        ]
        for i in range(N):
            centers[i] = positions[i]
    elif mode == 3:
        # Slightly perturbed compact cluster
        centers = np.zeros((N, 2))
        positions = [
            (0, 0), (2, 0), (-1, 1.732), (1, 1.732),
            (-1, -1.732), (1, -1.732), (3, 0),
            (-2, 1.732), (2, 1.732), (-2, -1.732), (2, -1.732)
        ]
        for i in range(N):
            centers[i] = [positions[i][0] + rng.normal(0, 0.1),
                          positions[i][1] + rng.normal(0, 0.1)]
    elif mode == 4:
        # Ring-based: 1 center + 6 inner ring + 4 outer ring
        centers = np.zeros((N, 2))
        centers[0] = [0, 0]
        for i in range(6):
            ang = i * PI / 3.0
            r = 2.0
            centers[i + 1] = [r * math.cos(ang), r * math.sin(ang)]
        for i in range(4):
            ang = (i + 0.5) * PI / 3.0
            r = 3.2
            centers[i + 7] = [r * math.cos(ang), r * math.sin(ang)]
    else:
        # Random with moderate spread
        centers = rng.uniform(-3, 3, size=(N, 2))

    # Small random rotation
    theta = rng.uniform(-PI / 6, PI / 6)
    params = np.concatenate([centers.reshape(-1), [theta]])
    return params


def _optimize_one(init_params, theta_outer, maxiter=300):
    """Run L-BFGS-B optimization."""
    from scipy.optimize import minimize
    try:
        res = minimize(_objective, init_params, args=(theta_outer,),
                       method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
        return res.x
    except Exception:
        return init_params


def _polish(params, theta_outer):
    """High-precision polish using SLSQP and tighter L-BFGS-B."""
    from scipy.optimize import minimize
    best = params
    best_val = _objective(params, theta_outer)

    # SLSQP with tight tolerance
    try:
        res = minimize(_objective, params, args=(theta_outer,),
                       method='SLSQP',
                       options={'maxiter': 300, 'ftol': 1e-14, 'maxls': 50})
        val = res.fun
        if val < best_val:
            best = res.x
            best_val = val
    except Exception:
        pass

    # Tight L-BFGS-B
    try:
        res = minimize(_objective, best, args=(theta_outer,),
                       method='L-BFGS-B',
                       options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13})
        val = res.fun
        if val < best_val:
            best = res.x
            best_val = val
    except Exception:
        pass

    return best


def construct():
    from scipy.optimize import minimize

    best = None
    best_score = float('inf')
    best_params = None

    # Try different outer rotations
    outer_rotations = [0.0, PI / 6.0, PI / 3.0, PI / 12.0]

    seed = 0
    maxiter = 300

    for theta_o in outer_rotations:
        while time.time() < DEADLINE - 15:  # reserve 15s for polish
            init_params = _make_init(seed)
            seed += 1
            try:
                params = _optimize_one(init_params, theta_o, maxiter)
            except Exception:
                continue

            centers = params[:N * 2].reshape(N, 2)
            theta = params[N * 2]

            if not _is_valid(centers, theta, theta_o):
                continue

            s = _compute_side(centers, theta, theta_o)
            if s < best_score:
                best_score = s
                best = (centers.copy(), theta, theta_o)
                best_params = params.copy()

            # Early exit if we're running low on time
            if time.time() > DEADLINE - 20:
                break

    # Polish the best solution
    if best_params is not None and time.time() < DEADLINE:
        theta_o = best[2]
        polished = _polish(best_params, theta_o)
        centers_p = polished[:N * 2].reshape(N, 2)
        theta_p = polished[N * 2]

        if _is_valid(centers_p, theta_p, theta_o):
            s_p = _compute_side(centers_p, theta_p, theta_o)
            if s_p < best_score:
                best_score = s_p
                best = (centers_p.copy(), theta_p, theta_o)

    if best is None:
        # Fallback: return a valid (but suboptimal) configuration
        centers = np.zeros((N, 2))
        positions = [
            (0, 0), (2, 0), (-1, 1.732), (1, 1.732),
            (-1, -1.732), (1, -1.732), (3, 0),
            (-2, 1.732), (2, 1.732), (-2, -1.732), (2, -1.732)
        ]
        for i in range(N):
            centers[i] = positions[i]
        s = _compute_side(centers, 0.0, 0.0)
        inner = [[float(centers[i, 0]), float(centers[i, 1]), 0.0] for i in range(N)]
        return inner, (0.0, 0.0), float(s), 0.0

    centers, theta, theta_o = best
    s = _compute_side(centers, theta, theta_o)
    inner = [[float(centers[i, 0]), float(centers[i, 1]), float(math.degrees(theta))] for i in range(N)]
    return inner, (0.0, 0.0), float(s), float(math.degrees(theta_o))
