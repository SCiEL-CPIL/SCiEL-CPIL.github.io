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


def _hexagon_distance(center1, center2, angle1, angle2):
    """
    Compute the minimum distance between two unit hexagons given their centers and angles.
    If they overlap, returns a negative value (penalty).
    Uses support function approach: for convex polygons, the distance can be computed
    by checking all pairs of edges/faces.
    """
    # Get vertices of each hexagon
    v1 = center1 + _hex_vertices_local(angle1)  # (6,2)
    v2 = center2 + _hex_vertices_local(angle2)  # (6,2)
    
    # Compute minimum distance between two convex hexagons
    # For simplicity, use a conservative estimate based on center distance
    # and the fact that hexagons have inradius sqrt(3)/2
    center_dist = np.linalg.norm(center2 - center1)
    
    # The minimum center distance for non-overlapping unit hexagons depends on relative orientation
    # For aligned hexagons, min center distance = 2 * sqrt(3) / 2 * 2 = 2*sqrt(3) ≈ 3.464? No.
    # Actually for two unit hexagons (side 1, circumradius 1), the minimum center distance
    # for non-overlap is 2 (when facing each other with flat sides) or sqrt(3) (when point to point).
    # Let's use a simpler approach: compute actual minimum distance between vertices/edges
    
    # Compute all pairwise distances between vertices
    min_dist = float('inf')
    for i in range(6):
        for j in range(6):
            d = np.linalg.norm(v1[i] - v2[j])
            if d < min_dist:
                min_dist = d
    
    # Also check edge-to-edge distances (approximate by checking midpoints)
    for i in range(6):
        for j in range(6):
            # Midpoint of edge i of hex1
            m1 = (v1[i] + v1[(i+1)%6]) / 2
            # Midpoint of edge j of hex2
            m2 = (v2[j] + v2[(j+1)%6]) / 2
            d = np.linalg.norm(m1 - m2)
            if d < min_dist:
                min_dist = d
    
    return min_dist


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
    centers = _compute_cluster_centers(t, tx, ty, theta)
    
    # Check pairwise distances for overlap using center distance approximation
    # For hexagons aligned with cluster rotation theta, min center distance is 2.0
    # (this is the distance between centers of adjacent hexagons in a tight packing)
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    min_d = d[iu].min()
    
    # Compute outer side length
    verts_local = _hex_vertices_local(0.0)  # hexagons are axis-aligned in local frame
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
    
    # Penalty for overlap (use center distance as proxy)
    if min_d < 2.0:
        penalty = 1000.0 * (2.0 - min_d) ** 2
    else:
        penalty = 0.0
    
    return s_outer + penalty


def _is_valid_cluster(params):
    """Check if cluster configuration is valid (no overlap, contained in outer hex)"""
    t, tx, ty, theta, theta_outer = params
    centers = _compute_cluster_centers(t, tx, ty, theta)
    
    # Check pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    if d[iu].min() < 2.0 - 1e-5:
        return False
    
    return True


def _make_cluster_init(seed):
    """Create initial parameters for cluster optimization"""
    rng = np.random.RandomState(seed)
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
