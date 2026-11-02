"""Place 12 unit regular hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

def _hex_vertices(cx, cy, s, ang_deg):
    """Return (6,2) array of vertices for a regular hexagon."""
    a = np.radians(ang_deg)
    # Flat-top hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees
    angles = a + np.deg2rad(np.arange(0, 360, 60))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _min_outer_side(vertices, outer_center, outer_angle_deg):
    """
    Compute the minimum side length S of a regular hexagon with given center and rotation
    that contains all given vertices (points).
    
    A regular hexagon with center (ox, oy), side S, rotation theta (flat-top at theta=0)
    can be described by 6 half-planes. For each vertex (px, py) of the outer hexagon
    (at distance S from center, at angles theta, theta+60, ...), the supporting line
    is perpendicular to that vertex direction.
    
    The constraint for a point (px, py) to be inside:
    For each face direction d_i (unit vector from center to face midpoint),
    (p - c) . d_i <= S (since the apothem = S * cos(30°) = S * sqrt(3)/2)
    
    Wait, let me reconsider. For a regular hexagon with side length S:
    - The distance from center to each vertex is S.
    - The distance from center to each face (apothem) is S * sqrt(3)/2.
    
    The 6 face normals point at angles theta + 30°, theta + 90°, ... (for flat-top, faces are at 30, 90, 150, 210, 270, 330)
    Actually for flat-top (vertex at 0°), the faces are centered at 30°, 90°, 150°, 210°, 270°, 330°.
    
    For a point to be inside: (p - c) . n_i <= apothem for all 6 face normals n_i.
    So: S >= max_i ( (p - c) . n_i ) / (sqrt(3)/2) for all points p.
    
    Thus S_min = (2/sqrt(3)) * max over all points p and all 6 face normals n_i of ((p-c).n_i).
    """
    ox, oy = outer_center
    theta = np.radians(outer_angle_deg)
    # Face normals at theta + 30, theta + 90, ..., theta + 330
    face_angles = theta + np.deg2rad(np.arange(30, 360, 60))
    normals = np.column_stack([np.cos(face_angles), np.sin(face_angles)])
    
    # For each vertex, compute max dot product with any face normal
    # vertices: (M, 2), normals: (6, 2)
    diffs = vertices - np.array([ox, oy])  # (M, 2)
    dots = diffs @ normals.T  # (M, 6)
    max_dot = np.max(dots)
    
    return (2.0 / np.sqrt(3.0)) * max_dot

def _objective(params):
    """
    params: flattened array of [x1,y1,a1, x2,y2,a2, ..., x12,y2,a12, ox,oy, S, oang]
    where ai is angle in radians, oang is outer angle in radians.
    We minimize S + penalty for containment violations.
    """
    n_inner = 12
    # Unpack
    positions = params[:n_inner * 3].reshape(n_inner, 3)
    ox, oy, S, oang = params[n_inner * 3:n_inner * 3 + 4]
    
    # Collect all vertices of inner hexagons
    all_verts = []
    for i in range(n_inner):
        x, y, ang = positions[i]
        v = _hex_vertices(x, y, 1.0, np.degrees(ang))
        all_verts.append(v)
    all_verts = np.vstack(all_verts)  # (72, 2)
    
    # Compute required S for containment
    required_S = _min_outer_side(all_verts, (ox, oy), np.degrees(oang))
    
    # Penalty: if S < required_S, penalize
    if S < required_S:
        penalty = 100.0 * (required_S - S) ** 2
    else:
        penalty = 0.0
    
    return S + penalty

def _construct_initial():
    """Create a compact honeycomb arrangement of 12 hexagons."""
    # Axial coordinates for 12 hexagons in a compact cluster
    # Center + 6 neighbors + 5 more in a ring
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2), (1, 1)
    ]
    
    # Convert axial to cartesian (flat-top, side=1)
    positions = []
    for q, r in axial:
        x = 1.5 * q
        y = np.sqrt(3) / 2 * (q + 2 * r)
        positions.append([x, y, 0.0])  # angle 0 (flat-top)
    
    positions = np.array(positions)
    
    # Estimate outer hexagon: center at origin, find bounding
    all_verts = np.vstack([_hex_vertices(p[0], p[1], 1.0, 0.0) for p in positions])
    
    # Try a few outer angles and pick best
    best_params = None
    best_score = float('inf')
    for oang_deg in [0, 30, 15, 45]:
        req_S = _min_outer_side(all_verts, (0.0, 0.0), oang_deg)
        params = np.concatenate([positions.flatten(), [0.0, 0.0, req_S, np.radians(oang_deg)]])
        score = _objective(params)
        if score < best_score:
            best_score = score
            best_params = params.copy()
    
    return best_params

def construct():
    """Construct 12 unit hexagons inside a regular hexagon, minimizing outer side."""
    start_time = time.time()
    deadline = 120.0  # safety margin
    
    # Initial solution
    params = _construct_initial()
    best_params = params.copy()
    best_score = _objective(params)
    
    # Local optimization with Nelder-Mead
    try:
        from scipy.optimize import minimize
        result = minimize(_objective, params, method='Nelder-Mead',
                         options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x.copy()
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        pass  # return best found so far
    
    # Extract final parameters
    n_inner = 12
    positions = best_params[:n_inner * 3].reshape(n_inner, 3)
    ox, oy, S, oang = best_params[n_inner * 3:n_inner * 3 + 4]
    
    # Convert angles to degrees
    inner = [[float(p[0]), float(p[1]), float(np.degrees(p[2]))] for p in positions]
    center = (float(ox), float(oy))
    side = float(S)
    angle = float(np.degrees(oang))
    
    return inner, center, side, angle
