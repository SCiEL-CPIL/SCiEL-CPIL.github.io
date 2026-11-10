import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

_START_TIME = time.time()
_DEADLINE = 630.0
_SAFETY_MARGIN = 30.0


def _hex_vertices(cx, cy, d, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _min_containing_hex_side(centers, outer_angle_deg=0.0):
    """
    Compute the minimum side length s of a regular hexagon centered at (0,0)
    with the given angle that contains all unit hexagons with given centers.
    """
    base_angles = np.deg2rad(60.0 * np.arange(6))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)

    n = len(centers)
    # All vertices for all hexagons: (n, 6, 2)
    verts = np.zeros((n, 6, 2))
    for i in range(n):
        verts[i, :, 0] = centers[i, 0] + base_cos
        verts[i, :, 1] = centers[i, 1] + base_sin

    # Outward normals at angles 30, 90, 150, 210, 270, 330 (relative to outer hex angle)
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330])) + np.deg2rad(outer_angle_deg)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6, 2)

    verts_flat = verts.reshape(-1, 2)  # (36, 2)
    projections = normals @ verts_flat.T  # (6, 36)
    max_proj_per_normal = np.max(projections, axis=1)  # (6,)
    max_proj = np.max(max_proj_per_normal)  # scalar

    s = (2.0 / np.sqrt(3.0)) * max_proj
    return s


def _check_separation(centers, tol=1e-6):
    """
    Check if all pairs of unit hexagons are separated using SAT.
    """
    n = len(centers)
    base_angles = np.deg2rad(60.0 * np.arange(6))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)

    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])

    all_verts = np.zeros((n, 6, 2))
    for i in range(n):
        all_verts[i, :, 0] = centers[i, 0] + base_cos
        all_verts[i, :, 1] = centers[i, 1] + base_sin

    for i in range(n):
        for j in range(i + 1, n):
            vi = all_verts[i]
            vj = all_verts[j]
            proj_i = vi @ normals.T
            proj_j = vj @ normals.T
            min_i = np.min(proj_i, axis=1)
            max_i = np.max(proj_i, axis=1)
            min_j = np.min(proj_j, axis=1)
            max_j = np.max(proj_j, axis=1)
            separated = np.any(max_i < min_j - tol) or np.any(max_j < min_i - tol)
            if not separated:
                return False
    return True


def _make_rigid_cluster():
    """
    Create a compact 12-hexagon cluster in the hexagonal lattice.
    Using the standard hex lattice: neighbors at distance sqrt(3) in 6 directions.
    For flat-top hexagons (angle 0), the 6 neighbor offsets are:
    (sqrt(3), 0), (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5),
    (-sqrt(3), 0), (-sqrt(3)/2, -1.5), (sqrt(3)/2, -1.5)
    
    We use a 12-hex cluster: center + 6 ring-1 + 5 from ring-2.
    """
    sqrt3 = np.sqrt(3)
    # Ring 1: 6 hexagons around center
    # Ring 2: we pick 5 specific positions to make a compact shape
    
    # Center
    pts = [[0.0, 0.0]]
    
    # Ring 1 (6 hexagons)
    ring1 = [
        [sqrt3, 0.0],
        [sqrt3/2, 1.5],
        [-sqrt3/2, 1.5],
        [-sqrt3, 0.0],
        [-sqrt3/2, -1.5],
        [sqrt3/2, -1.5],
    ]
    pts.extend(ring1)
    
    # Ring 2: pick 5 that form a compact shape
    # Ring 2 positions (distance 2*sqrt(3) from center in various directions):
    # Let's use: (2*sqrt(3), 0), (3*sqrt(3)/2, 3.0), (0, 3.0), (-3*sqrt(3)/2, 3.0), (-2*sqrt(3), 0)
    # Actually let me compute proper ring-2 positions:
    # Ring 2 = neighbors of ring-1 that aren't already in ring-1 or center
    # (sqrt(3), 0) + (sqrt(3), 0) = (2*sqrt(3), 0)
    # (sqrt(3), 0) + (sqrt(3)/2, 1.5) = (3*sqrt(3)/2, 1.5)
    # (sqrt3/2, 1.5) + (sqrt3/2, 1.5) = (sqrt3, 3.0)
    # (sqrt3/2, 1.5) + (0, 3.0)... wait, let me just use known positions
    
    # For a flat-top hex grid, ring 2 positions are at:
    # (2*sqrt(3), 0), (3*sqrt(3)/2, 1.5), (sqrt(3), 3.0), (0, 3.0),
    # (-sqrt(3), 3.0), (-3*sqrt(3)/2, 1.5), (-2*sqrt(3), 0),
    # (-3*sqrt(3)/2, -1.5), (-sqrt(3), -3.0), (0, -3.0),
    # (sqrt(3), -3.0), (3*sqrt(3)/2, -1.5)
    
    # Pick 5 that give a compact (roughly hexagonal) shape:
    ring2_selected = [
        [2*sqrt3, 0.0],
        [3*sqrt3/2, 1.5],
        [0.0, 3.0],
        [-3*sqrt3/2, 1.5],
        [-2*sqrt3, 0.0],
    ]
    pts.extend(ring2_selected)
    
    return np.array(pts)


def _make_rigid_cluster_v2():
    """
    Alternative compact 12-hex cluster: 2 rows of 6.
    For flat-top hexagons, adjacent in same row: dx = sqrt(3)
    Adjacent in next row: dx = sqrt(3)/2, dy = 1.5
    """
    sqrt3 = np.sqrt(3)
    pts = []
    # Row 0: 6 hexagons
    for i in range(6):
        pts.append([i * sqrt3, 0.0])
    # Row 1: 6 hexagons, shifted by sqrt(3)/2 in x
    for i in range(6):
        pts.append([i * sqrt3 + sqrt3/2, 1.5])
    
    return np.array(pts)


def _make_rigid_cluster_v3():
    """
    3 rows: 4 + 4 + 4
    """
    sqrt3 = np.sqrt(3)
    pts = []
    for i in range(4):
        pts.append([i * sqrt3, 0.0])
    for i in range(4):
        pts.append([i * sqrt3 + sqrt3/2, 1.5])
    for i in range(4):
        pts.append([i * sqrt3, 3.0])
    
    return np.array(pts)


def _make_rigid_cluster_v4():
    """
    2 rows: 5 + 7 (or 7 + 5)
    """
    sqrt3 = np.sqrt(3)
    pts = []
    # Row 0: 5 hexagons centered
    for i in range(5):
        pts.append([(i - 2) * sqrt3, 0.0])
    # Row 1: 7 hexagons
    for i in range(7):
        pts.append([(i - 3) * sqrt3 + sqrt3/2, 1.5])
    
    return np.array(pts)


def _make_rigid_cluster_v5():
    """
    Compact: center + 6 ring1 + 5 ring2 but different ring2 selection
    """
    sqrt3 = np.sqrt(3)
    pts = [[0.0, 0.0]]
    ring1 = [
        [sqrt3, 0.0],
        [sqrt3/2, 1.5],
        [-sqrt3/2, 1.5],
        [-sqrt3, 0.0],
        [-sqrt3/2, -1.5],
        [sqrt3/2, -1.5],
    ]
    pts.extend(ring1)
    # Different ring-2 selection for more compact shape
    ring2_selected = [
        [3*sqrt3/2, 1.5],
        [sqrt3, 3.0],
        [-sqrt3, 3.0],
        [-3*sqrt3/2, 1.5],
        [0.0, -3.0],
    ]
    pts.extend(ring2_selected)
    
    return np.array(pts)


def _transform_cluster(base_pts, tx, ty, angle_deg):
    """Apply translation and rotation to the base cluster points."""
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s], [s, c]])
    rotated = base_pts @ R.T
    return rotated + np.array([tx, ty])


def _objective_rigid(params, base_pts):
    """
    Objective: minimize the containing hexagon side length for a rigid cluster.
    params: (tx, ty, angle_deg)
    """
    tx, ty, angle_deg = params
    centers = _transform_cluster(base_pts, tx, ty, angle_deg)
    s = _min_containing_hex_side(centers, outer_angle_deg=0.0)
    return s


def _objective_rigid_with_center(params, base_pts):
    """
    Objective where we also optimize the outer hex center and angle.
    params: (tx, ty, angle_deg)
    The outer hex is centered at the centroid of the cluster.
    """
    tx, ty, angle_deg = params
    centers = _transform_cluster(base_pts, tx, ty, angle_deg)
    
    # The outer hexagon should be centered at the centroid of the inner hexagons
    centroid = np.mean(centers, axis=0)
    
    # Shift so centroid is at origin for computing min containing hex
    shifted = centers - centroid
    s = _min_containing_hex_side(shifted, outer_angle_deg=0.0)
    return s


def _objective_full(params, base_pts):
    """
    Full objective: params = (tx, ty, angle_inner_deg, angle_outer_deg)
    tx, ty: translation of the rigid cluster
    angle_inner_deg: rotation of the inner hexagons
    angle_outer_deg: rotation of the outer hexagon
    """
    tx, ty, angle_inner, angle_outer = params
    a = np.deg2rad(angle_inner)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s], [s, c]])
    rotated = base_pts @ R.T
    centers = rotated + np.array([tx, ty])
    
    # Compute min containing hex side with outer angle
    s = _min_containing_hex_side(centers, outer_angle_deg=angle_outer)
    return s


def _check_validity(centers, outer_center, outer_angle, outer_side):
    """Check all validity constraints."""
    # Check separation
    if not _check_separation(centers, tol=1e-6):
        return False
    
    # Check containment: all inner vertices inside outer hex
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    inner_verts_all = np.zeros((12, 6, 2))
    for i in range(12):
        inner_verts_all[i] = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, 0.0)
    
    for i in range(12):
        for v in inner_verts_all[i]:
            # Check if v is inside outer hex
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                cross_val = np.cross(p2 - p1, v - p1)
                if cross_val < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
    
    # Check s >= sqrt(12)
    if outer_side < np.sqrt(12) - 1e-8:
        return False
    
    return True


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon side length.
    Uses rigid cluster approach: optimize only global rotation and translation.
    """
    # Multiple rigid cluster topologies to try
    clusters = [
        _make_rigid_cluster(),
        _make_rigid_cluster_v2(),
        _make_rigid_cluster_v3(),
        _make_rigid_cluster_v4(),
        _make_rigid_cluster_v5(),
    ]
    
    best_s = np.inf
    best_result = None
    
    rng = np.random.default_rng(42)
    
    for cluster_idx, base_pts in enumerate(clusters):
        elapsed = time.time() - _START_TIME
        if elapsed > _DEADLINE - _SAFETY_MARGIN:
            break
        
        # Center the cluster at origin first
        centroid = np.mean(base_pts, axis=0)
        base_pts_c = base_pts - centroid
        
        # Try multiple initial angles
        for angle_init in range(0, 60, 5):
            elapsed = time.time() - _START_TIME
            if elapsed > _DEADLINE - _SAFETY_MARGIN:
                break
            
            # Initial params: tx=0, ty=0, angle=angle_init
            x0 = np.array([0.0, 0.0, float(angle_init)])
            
            # Nelder-Mead optimization
            try:
                result = minimize(
                    _objective_rigid,
                    x0,
                    args=(base_pts_c,),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 5000,
                        'maxfev': 50000,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                    }
                )
            except Exception:
                continue
            
            if result is not None:
                tx, ty, angle_deg = result.x
                centers = _transform_cluster(base_pts_c, tx, ty, angle_deg)
                
                # The outer hex is centered at origin (or centroid of centers)
                outer_center = np.array([0.0, 0.0])
                s = _min_containing_hex_side(centers, outer_angle_deg=0.0)
                
                if _check_validity(centers, outer_center, 0.0, s):
                    if s < best_s:
                        best_s = s
                        best_result = (centers.copy(), outer_center.copy(), 0.0, s)
        
        # Also try with outer angle optimization (4 params)
        for angle_init in range(0, 60, 10):
            elapsed = time.time() - _START_TIME
            if elapsed > _DEADLINE - _SAFETY_MARGIN:
                break
            
            x0 = np.array([0.0, 0.0, float(angle_init), 0.0])
            
            try:
                result = minimize(
                    _objective_full,
                    x0,
                    args=(base_pts_c,),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 5000,
                        'maxfev': 50000,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                    }
                )
            except Exception:
                continue
            
            if result is not None:
                tx, ty, angle_inner, angle_outer = result.x
                centers = _transform_cluster(base_pts_c, tx, ty, angle_inner)
                outer_center = np.array([0.0, 0.0])
                s = _min_containing_hex_side(centers, outer_angle_deg=angle_outer)
                
                if _check_validity(centers, outer_center, angle_outer, s):
                    if s < best_s:
                        best_s = s
                        best_result = (centers.copy(), outer_center.copy(), angle_outer, s)
        
        # Differential evolution for global search (3 params)
        elapsed = time.time() - _START_TIME
        if elapsed < _DEADLINE - _SAFETY_MARGIN:
            try:
                bounds = [(-5, 5), (-5, 5), (0, 60)]
                de_result = differential_evolution(
                    _objective_rigid,
                    bounds,
                    args=(base_pts_c,),
                    seed=42,
                    maxiter=200,
                    tol=1e-10,
                    polish=True,
                )
            except Exception:
                de_result = None
            
            if de_result is not None and de_result.success:
                tx, ty, angle_deg = de_result.x
                centers = _transform_cluster(base_pts_c, tx, ty, angle_deg)
                outer_center = np.array([0.0, 0.0])
                s = _min_containing_hex_side(centers, outer_angle_deg=0.0)
                
                if _check_validity(centers, outer_center, 0.0, s):
                    if s < best_s:
                        best_s = s
                        best_result = (centers.copy(), outer_center.copy(), 0.0, s)
    
    # If no valid result found, use a fallback (free optimization)
    if best_result is None:
        # Fallback: use the first cluster as-is
        base_pts = _make_rigid_cluster()
        centroid = np.mean(base_pts, axis=0)
        centers = base_pts - centroid
        outer_center = np.array([0.0, 0.0])
        s = _min_containing_hex_side(centers, outer_angle_deg=0.0)
        best_result = (centers, outer_center, 0.0, s)
    
    centers, outer_center, outer_angle, outer_side = best_result
    
    # Build output
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = centers[i, 0]
        inner_hex_data[i, 1] = centers[i, 1]
        inner_hex_data[i, 2] = 0.0
    
    outer_hex_data = np.array([outer_center[0], outer_center[1], outer_angle])
    outer_hex_side_length = outer_side
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
