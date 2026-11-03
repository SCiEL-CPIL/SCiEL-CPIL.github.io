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


def _outer_side_smooth(theta, centers, theta_outer, beta=50.0):
    # Smooth approximation using log-sum-exp
    verts_local = _hex_vertices_local(theta)  # (6,2)
    verts = centers[:, None, :] + verts_local[None, :, :]  # (N,6,2)
    verts = verts.reshape(-1, 2)  # (N*6,2)
    normals = _outer_side_normals(theta_outer)  # (6,2)
    proj = verts @ normals.T  # (N*6, 6)
    # log-sum-exp for smooth max
    m = proj.max(axis=0)  # max per column (per normal)
    logsumexp = m + np.log(np.sum(np.exp(proj - m), axis=0))  # (6,)
    apothem = logsumexp.max()
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


def _objective(params, theta_outer, use_smooth=False, beta=50.0):
    # params: (N*2 + 1) = [x0,y0,...,xN-1,yN-1, theta]
    centers = params[:N * 2].reshape(N, 2)
    theta = params[N * 2]
    if use_smooth:
        s = _outer_side_smooth(theta, centers, theta_outer, beta)
    else:
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


def _make_init_hex_lattice(mode, seed):
    """Generate structured hexagonal lattice initializations."""
    rng = np.random.RandomState(seed)
    
    if mode == 0:
        # Central hexagonal cluster: 1 center + 6 ring + 4 more
        # Standard 11-hex packing in hexagonal lattice
        centers = []
        # Center
        centers.append([0.0, 0.0])
        # First ring (6 hexagons at distance 2)
        for i in range(6):
            ang = i * math.pi / 3
            x = 2.0 * math.cos(ang)
            y = 2.0 * math.sin(ang)
            centers.append([x, y])
        # Second ring partial (4 more to make 11)
        # Place at positions that fit in the gaps
        for i in [0, 1, 3, 4]:
            ang = i * math.pi / 3 + math.pi / 6
            r = 2.0 * math.sqrt(3)
            x = r * math.cos(ang)
            y = r * math.sin(ang)
            centers.append([x, y])
        
        centers = np.array(centers[:N])
        # Add small random perturbation
        centers += rng.normal(0, 0.01, size=centers.shape)
        theta = rng.uniform(-math.pi / 12, math.pi / 12)
        
    elif mode == 1:
        # Linear arrangement with slight zigzag
        centers = []
        x_pos = np.linspace(-4, 4, N)
        for i in range(N):
            x = x_pos[i]
            y = 0.5 * math.sin(i * math.pi / 3)  # slight zigzag
            centers.append([x, y])
        centers = np.array(centers)
        centers += rng.normal(0, 0.05, size=centers.shape)
        theta = rng.uniform(-math.pi / 6, math.pi / 6)
        
    elif mode == 2:
        # Two-row arrangement
        centers = []
        # Row 1: 6 hexagons
        for i in range(6):
            x = i * 2.0 - 5.0
            y = 0.0
            centers.append([x, y])
        # Row 2: 5 hexagons offset
        for i in range(5):
            x = i * 2.0 - 4.0
            y = math.sqrt(3)
            centers.append([x, y])
        centers = np.array(centers[:N])
        centers += rng.normal(0, 0.05, size=centers.shape)
        theta = rng.uniform(-math.pi / 6, math.pi / 6)
        
    else:
        # Random fallback
        centers = rng.uniform(-4, 4, size=(N, 2))
        theta = rng.uniform(-math.pi / 6, math.pi / 6)
    
    params = np.concatenate([centers.reshape(-1), [theta]])
    return params


def _make_init(seed):
    """Random feasible-ish start with diverse modes."""
    rng = np.random.RandomState(seed)
    mode = seed % 3  # Cycle through lattice modes
    return _make_init_hex_lattice(mode, seed)


def _optimize_one(init_params, theta_outer, maxiter, use_smooth=True, beta=50.0):
    from scipy.optimize import minimize
    
    def obj(params):
        return _objective(params, theta_outer, use_smooth=use_smooth, beta=beta)
    
    res = minimize(obj, init_params,
                   method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
    return res


def _polish(centers, theta, theta_outer):
    """High-precision polish using SLSQP with tighter tolerances."""
    from scipy.optimize import minimize
    
    def obj(params):
        return _objective(params, theta_outer, use_smooth=False)
    
    def constraint_nonoverlap(params):
        c = params[:N * 2].reshape(N, 2)
        diff = c[:, None, :] - c[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        return d[iu] - 2.0  # Should be >= 0
    
    params = np.concatenate([centers.reshape(-1), [theta]])
    
    # Define constraints for each pair
    constraints = []
    iu = np.triu_indices(N, k=1)
    for (i, j) in zip(*iu):
        def make_constraint(i, j):
            def constraint(params):
                c = params[:N * 2].reshape(N, 2)
                d = np.linalg.norm(c[i] - c[j])
                return d - 2.0
            return constraint
        constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
    
    res = minimize(obj, params, method='SLSQP', constraints=constraints,
                   options={'maxiter': 200, 'ftol': 1e-14})
    
    if res.success:
        new_centers = res.x[:N * 2].reshape(N, 2)
        new_theta = res.x[N * 2]
        return new_centers, new_theta
    
    return centers, theta


def construct():
    global T0
    from scipy.optimize import minimize

    best = None
    best_score = float('inf')
    
    # Try multiple outer rotations to exploit symmetry
    outer_rotations = [0.0, math.pi / 6, math.pi / 3]

    seed = 0
    maxiter = 300
    
    for theta_o in outer_rotations:
        while time.time() - T0 < DEADLINE:
            init_params = _make_init(seed)
            seed += 1
            try:
                # First pass: smooth objective with L-BFGS-B
                res = _optimize_one(init_params, theta_o, maxiter, use_smooth=True, beta=50.0)
                params = res.x
                centers = params[:N * 2].reshape(N, 2)
                theta = params[N * 2]
                
                # Check validity before polish
                if not _is_valid(centers, theta, theta_o):
                    # Try to fix with a second L-BFGS-B pass with non-smooth objective
                    res2 = _optimize_one(params, theta_o, maxiter // 2, use_smooth=False)
                    params = res2.x
                    centers = params[:N * 2].reshape(N, 2)
                    theta = params[N * 2]
                    if not _is_valid(centers, theta, theta_o):
                        continue
                
                # Polish phase with SLSQP
                if time.time() - T0 < DEADLINE - 3:
                    centers, theta = _polish(centers, theta, theta_o)
                    if not _is_valid(centers, theta, theta_o):
                        continue
                
                s = _outer_side(theta, centers, theta_o)
                if s < best_score:
                    best_score = s
                    best = (centers.copy(), theta, theta_o)
                    
            except Exception:
                continue
            
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
