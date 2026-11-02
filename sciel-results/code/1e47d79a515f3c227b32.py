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
    ang = theta_outer + np.arange(6) * (math.pi / 3.0) + math.pi / 6.0
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)  # (6,2)


def _compute_cluster_centers(t, tx, ty, theta):
    """
    Compute world positions of 11 hexagons in a compact hexagonal cluster.
    t: cluster scale factor (controls spacing between hex centers)
    tx, ty: global translation
    theta: global rotation of the cluster
    Returns: (N, 2) array of centers
    """
    # Local positions in hexagonal lattice (center + ring of 6 + 4 more)
    local = np.zeros((N, 2))
    # Center
    local[0] = [0.0, 0.0]
    # Ring of 6 at distance 2*t
    for i in range(6):
        ang = i * math.pi / 3.0
        local[i + 1] = [2.0 * t * math.cos(ang), 2.0 * t * math.sin(ang)]
    # 4 more at distance ~3.5*t, offset by 30 degrees
    for i in range(4):
        ang = (i + 0.5) * math.pi / 3.0
        local[i + 7] = [3.5 * t * math.cos(ang), 3.5 * t * math.sin(ang)]
    
    # Rotate cluster
    c = math.cos(theta)
    s = math.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    world = local @ rot.T + np.array([tx, ty])
    return world


def _cluster_objective(params):
    """
    params: [t, tx, ty, theta, theta_outer]
    t: cluster scale
    tx, ty: translation
    theta: cluster rotation
    theta_outer: outer hexagon orientation
    
    Returns: outer side length (with penalty if hexagons overlap)
    """
    t, tx, ty, theta, theta_outer = params
    if t < 0.95:  # ensure no overlap (min center distance should be >= 2)
        t = 0.95
    centers = _compute_cluster_centers(t, tx, ty, theta)
    
    # Check pairwise distances for overlap
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    min_d = d[iu].min()
    
    # Compute outer side length
    verts_local = _hex_vertices_local(0.0)  # hexagons are axis-aligned in local frame
    # Each hexagon's vertices in world: center + rotated local vertices
    # Since cluster is rotated by theta, and hexagons are aligned with cluster,
    # each hexagon has rotation theta
    c = math.cos(theta)
    s = math.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    verts_rot = verts_local @ rot.T  # (6,2)
    
    verts = centers[:, None, :] + verts_rot[None, :, :]  # (N, 6, 2)
    verts = verts.reshape(-1, 2)  # (N*6, 2)
    
    normals = _outer_side_normals(theta_outer)
    proj = verts @ normals.T  # (N*6, 6)
    apothem = proj.max()
    s_outer = apothem * 2.0 / math.sqrt(3.0)
    
    # Penalty for overlap
    if min_d < 2.0:
        penalty = 1000.0 * (2.0 - min_d) ** 2
    else:
        penalty = 0.0
    
    return s_outer + penalty


def _is_valid_cluster(params):
    """Check if cluster configuration is valid (no overlap, contained in outer hex)"""
    t, tx, ty, theta, theta_outer = params
    if t < 0.95:
        return False
    centers = _compute_cluster_centers(t, tx, ty, theta)
    
    # Check pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    if d[iu].min() < 2.0 - 1e-5:
        return False
    
    # Check containment
    verts_local = _hex_vertices_local(0.0)
    c = math.cos(theta)
    s = math.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    verts_rot = verts_local @ rot.T
    verts = centers[:, None, :] + verts_rot[None, :, :]
    verts = verts.reshape(-1, 2)
    
    normals = _outer_side_normals(theta_outer)
    proj = verts @ normals.T
    apothem_needed = proj.max()
    
    # Outer hexagon must contain all vertices
    # For a regular hexagon with side s, apothem = s*sqrt(3)/2
    # We need apothem >= apothem_needed
    # This is automatically satisfied if we compute s from apothem
    # But let's verify the outer hexagon actually contains all points
    # The outer hexagon has vertices at distance s from center, with sides at distance s*sqrt(3)/2
    # A point is inside if its projection on each outward normal <= apothem
    # Since we set apothem = proj.max(), this is satisfied by construction
    return True


def _make_cluster_init(seed):
    """Create initial parameters for cluster optimization"""
    rng = np.random.RandomState(seed)
    # Start with t close to 1.0 (minimum non-overlapping)
    t = 1.0 + rng.uniform(0, 0.3)
    tx = rng.uniform(-1, 1)
    ty = rng.uniform(-1, 1)
    theta = rng.uniform(0, math.pi / 3)
    theta_outer = rng.uniform(0, math.pi / 3)
    return np.array([t, tx, ty, theta, theta_outer])


def _optimize_cluster(init_params, maxiter):
    """Optimize cluster parameters using L-BFGS-B"""
    from scipy.optimize import minimize
    res = minimize(_cluster_objective, init_params,
                   method='L-BFGS-B',
                   options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
    return res


def _polish_cluster(params, maxiter):
    """High-precision polish pass for cluster parameters"""
    from scipy.optimize import minimize
    res = minimize(_cluster_objective, params,
                   method='L-BFGS-B',
                   options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12})
    return res


def construct():
    global T0
    from scipy.optimize import minimize

    best_params = None
    best_score = float('inf')

    seed = 0
    maxiter = 300

    while time.time() - T0 < DEADLINE:
        init_params = _make_cluster_init(seed)
        seed += 1
        try:
            res = _optimize_cluster(init_params, maxiter)
        except Exception:
            continue
        
        params = res.x
        if not _is_valid_cluster(params):
            continue
        
        score = _cluster_objective(params)
        if score < best_score:
            best_score = score
            best_params = params.copy()
            
            # Polish with tighter tolerances
            if time.time() - T0 < DEADLINE - 10:
                try:
                    res_polish = _polish_cluster(params, 200)
                    params_polish = res_polish.x
                    if _is_valid_cluster(params_polish):
                        score_polish = _cluster_objective(params_polish)
                        if score_polish < best_score:
                            best_score = score_polish
                            best_params = params_polish.copy()
                except Exception:
                    pass

        if time.time() - T0 > DEADLINE - 5:
            break

    if best_params is None:
        # Fallback: return a valid configuration
        t, tx, ty, theta, theta_outer = 1.0, 0.0, 0.0, 0.0, 0.0
        params = np.array([t, tx, ty, theta, theta_outer])
    else:
        params = best_params

    t, tx, ty, theta, theta_outer = params
    centers = _compute_cluster_centers(t, tx, ty, theta)
    s = _cluster_objective(params)
    
    # Build return values
    inner = []
    for i in range(N):
        inner.append([float(centers[i, 0]), float(centers[i, 1]), float(math.degrees(theta))])
    
    return inner, (0.0, 0.0), float(s), float(math.degrees(theta_outer))
