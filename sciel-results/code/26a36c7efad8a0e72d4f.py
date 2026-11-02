import time
import math
import numpy as np

N = 11
T0 = time.time()
DEADLINE = 115.0  # safety margin before 120s kill
TOL = 1e-7


def _hex_vertices_local(theta):
    # unit hexagon vertices (circumradius 1) rotated by theta
    c = math.cos(theta)
    s = math.sin(theta)
    ang = np.arange(6) * (math.pi / 3.0) + theta
    vx = np.cos(ang)
    vy = np.sin(ang)
    return np.stack([vx, vy], axis=1)  # (6,2)


def _outer_side_normals(theta_outer):
    # outward normals of the 6 sides of the outer hexagon
    # side i has outward normal at angle theta_outer + i*pi/3 + pi/6
    ang = theta_outer + np.arange(6) * (math.pi / 3.0) + math.pi / 6.0
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)  # (6,2)


def _outer_side(theta, centers, theta_outer):
    # centers: (N,2); returns minimal outer side length containing all vertices
    verts_local = _hex_vertices_local(theta)  # (6,2)
    # all world vertices: (N*6, 2)
    verts = centers[:, None, :] + verts_local[None, :, :]  # (N,6,2)
    verts = verts.reshape(-1, 2)  # (N*6,2)
    normals = _outer_side_normals(theta_outer)  # (6,2)
    # for each side normal n, required half-width = max_v (n . (v - c_outer))
    # c_outer = (0,0)
    # distance from center along normal n = n . v
    # side length s relates to apothem a = s * sqrt(3)/2, and a = max over sides of (n . v)
    # actually for a regular hexagon, the distance from center to each side (apothem) = s*sqrt(3)/2
    # and that must be >= max over vertices of (n . v) for each side normal n
    # So apothem_needed = max over (n, v) of (n . v)
    # s = apothem_needed * 2 / sqrt(3)
    proj = verts @ normals.T  # (N*6, 6)
    apothem = proj.max()
    s = apothem * 2.0 / math.sqrt(3.0)
    return s


def _pairwise_penalty(centers):
    # centers (N,2); penalty for pairs closer than 2
    # d_ij = |c_i - c_j|
    diff = centers[:, None, :] - centers[None, :, :]  # (N,N,2)
    d = np.linalg.norm(diff, axis=2)
    # for i<j, penalty = max(0, 2 - d)^2
    iu = np.triu_indices(N, k=1)
    dpair = d[iu]
    viol = np.maximum(0.0, 2.0 - dpair)
    return np.sum(viol * viol)


def _objective(params, theta_outer):
    # params: (N*2 + 1) = [x0,y0,...,xN-1,yN-1, theta]
    centers = params[:N * 2].reshape(N, 2)
    theta = params[N * 2]
    s = _outer_side(theta, centers, theta_outer)
    pen = _pairwise_penalty(centers)
    return s + 1000.0 * pen


def _is_valid(centers, theta, theta_outer):
    # check non-overlap
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    if d[iu].min() < 2.0 - 1e-5:
        return False
    # check containment: all vertices inside outer hexagon
    s = _outer_side(theta, centers, theta_outer)
    # recompute with a slightly larger side to be safe; actually check each vertex
    verts_local = _hex_vertices_local(theta)
    verts = (centers[:, None, :] + verts_local[None, :, :]).reshape(-1, 2)
    normals = _outer_side_normals(theta_outer)
    apothem = s * math.sqrt(3) / 2.0
    proj = verts @ normals.T  # (N*6, 6)
    if proj.max() > apothem + 1e-5:
        return False
    return True


def _make_init(seed):
    rng = np.random.RandomState(seed)
    # random feasible-ish start: place on a coarse grid region, small theta
    centers = rng.uniform(-3, 3, size=(N, 2))
    theta = rng.uniform(-math.pi / 6, math.pi / 6)
    params = np.concatenate([centers.reshape(-1), [theta]])
    return params


def _optimize_one(init_params, theta_outer, maxiter):
    from scipy.optimize import minimize
    res = minimize(_objective, init_params, args=(theta_outer,),
                   method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
    return res


def construct():
    global T0
    from scipy.optimize import minimize

    best = None
    best_score = float('inf')
    theta_outer = 0.0

    # Try a few outer rotations? The outer hexagon orientation is free.
    # We can fix theta_outer = 0 by symmetry (rotating everything). Keep 0.
    outer_rotations = [0.0]

    seed = 0
    maxiter = 400

    for theta_o in outer_rotations:
        while time.time() - T0 < DEADLINE:
            init_params = _make_init(seed)
            seed += 1
            try:
                res = _optimize_one(init_params, theta_o, maxiter)
            except Exception:
                continue
            params = res.x
            centers = params[:N * 2].reshape(N, 2)
            theta = params[N * 2]
            if not _is_valid(centers, theta, theta_o):
                continue
            s = _outer_side(theta, centers, theta_o)
            if s < best_score:
                best_score = s
                best = (centers.copy(), theta, theta_o)
                # tighten maxiter as we get better? keep
            if time.time() - T0 > DEADLINE - 5:
                break

    if best is None:
        # fallback: return something (may be invalid -> score 0)
        centers = np.zeros((N, 2))
        return (list(centers[:, 0].tolist()), list(centers[:, 1].tolist())), (0.0, 0.0), 40.0, 0.0

    centers, theta, theta_o = best
    s = _outer_side(theta, centers, theta_o)
    inner = [[float(centers[i, 0]), float(centers[i, 1]), float(math.degrees(theta))] for i in range(N)]
    return inner, (0.0, 0.0), float(s), float(math.degrees(theta_o))
