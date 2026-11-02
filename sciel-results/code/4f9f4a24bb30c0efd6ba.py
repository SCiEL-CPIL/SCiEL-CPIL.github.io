import time
import math
import numpy as np

N = 11
T0 = time.time()
DEADLINE = 145.0  # safety margin before 150s kill
TOL = 1e-7


def _hex_vertices_local(theta):
    # unit hexagon vertices (circumradius 1) rotated by theta
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
    proj = verts @ normals.T  # (N*6, 6)
    apothem = proj.max()
    s = apothem * 2.0 / math.sqrt(3.0)
    return s


def _pairwise_penalty(centers):
    # centers (N,2); penalty for pairs closer than 2
    diff = centers[:, None, :] - centers[None, :, :]  # (N,N,2)
    d = np.linalg.norm(diff, axis=2)
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
    # Use a structured initial guess: hexagonal lattice packing
    # Place hexagons in a hexagonal arrangement around center
    # For N=11: 1 center + 6 in first ring + 4 in second ring (partial)
    # Or just use a grid that's likely to be feasible
    centers = np.zeros((N, 2))
    
    # Try a few different structured patterns based on seed
    pattern = seed % 3
    
    if pattern == 0:
        # Hexagonal lattice: center at origin, rings of 6, 12, ...
        # For 11 hexagons: 1 + 6 + 4 (from second ring)
        # Hex lattice spacing: 2*sqrt(3) for non-overlapping (distance between centers >= 2)
        # Actually for hexagons of side 1, distance between centers must be >= 2
        # Hex lattice with spacing 2: neighbors at distance 2
        positions = []
        # Center
        positions.append((0, 0))
        # First ring: 6 hexagons at distance 2, angles 0, 60, 120, ...
        for k in range(6):
            ang = k * math.pi / 3
            positions.append((2 * math.cos(ang), 2 * math.sin(ang)))
        # Second ring: pick 4 from the 12 available
        # Second ring positions at distance 4 and 2*sqrt(3)
        # Let's just pick 4 that are far apart
        second_ring = []
        for k in range(12):
            ang = k * math.pi / 6
            # Alternate between distance 4 (aligned with first ring) and 2*sqrt(3)
            if k % 2 == 0:
                r = 4.0
            else:
                r = 2 * math.sqrt(3)
            second_ring.append((r * math.cos(ang), r * math.sin(ang)))
        # Pick 4 evenly spaced from second ring
        for k in range(0, 12, 3):  # 0, 3, 6, 9
            positions.append(second_ring[k])
        
        centers = np.array(positions[:N])
        
    elif pattern == 1:
        # Grid pattern: 3x3 + 2 extra
        # Hexagonal grid offset
        for i in range(3):
            for j in range(3):
                x = j * 3.5  # horizontal spacing
                y = i * 2 * math.sqrt(3)  # vertical spacing
                if j % 2 == 1:
                    y += math.sqrt(3)  # offset odd rows
                centers[i * 3 + j] = (x - 3.5, y - math.sqrt(3))
        # Add 2 more
        centers[9] = (0, 0)
        centers[10] = (3.5, 2 * math.sqrt(3) + math.sqrt(3))
        
    else:
        # Random but with some structure
        centers = rng.uniform(-3, 3, size=(N, 2))
    
    theta = rng.uniform(-math.pi / 6, math.pi / 6)
    params = np.concatenate([centers.reshape(-1), [theta]])
    return params


def _optimize_one(init_params, theta_outer, maxiter):
    from scipy.optimize import minimize
    res = minimize(_objective, init_params, args=(theta_outer,),
                   method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12})
    return res


def _polish(params, theta_outer, maxiter):
    """High-precision polish pass with tighter tolerances"""
    from scipy.optimize import minimize
    res = minimize(_objective, params, args=(theta_outer,),
                   method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-16, 'gtol': 1e-14})
    return res


def construct():
    global T0
    from scipy.optimize import minimize

    best = None
    best_score = float('inf')
    theta_outer = 0.0

    outer_rotations = [0.0]

    seed = 0
    maxiter = 600

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
                best = (centers.copy(), theta, theta_o, params.copy())
                # If we have time, do a polish pass
                if time.time() - T0 < DEADLINE - 10:
                    try:
                        res_polish = _polish(params, theta_o, 200)
                        params_p = res_polish.x
                        centers_p = params_p[:N * 2].reshape(N, 2)
                        theta_p = params_p[N * 2]
                        if _is_valid(centers_p, theta_p, theta_o):
                            s_p = _outer_side(theta_p, centers_p, theta_o)
                            if s_p < best_score:
                                best_score = s_p
                                best = (centers_p.copy(), theta_p, theta_o, params_p.copy())
                    except Exception:
                        pass
            if time.time() - T0 > DEADLINE - 5:
                break

    if best is None:
        # fallback: return something (may be invalid -> score 0)
        centers = np.zeros((N, 2))
        return (list(centers[:, 0].tolist()), list(centers[:, 1].tolist())), (0.0, 0.0), 40.0, 0.0

    centers, theta, theta_o, _ = best
    s = _outer_side(theta, centers, theta_o)
    inner = [[float(centers[i, 0]), float(centers[i, 1]), float(math.degrees(theta))] for i in range(N)]
    return inner, (0.0, 0.0), float(s), float(math.degrees(theta_o))
