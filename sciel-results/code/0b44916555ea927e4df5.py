import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0
SQRT3 = math.sqrt(3)

_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i + 1) % 6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i + 1) % 6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, ex
    _norm = math.sqrt(_nx * _nx + _ny * _ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

_UX = _UNIT_HEX_VERTS[:, 0]
_UY = _UNIT_HEX_VERTS[:, 1]
_ENX = _EDGE_NORMALS[:, 0]
_ENY = _EDGE_NORMALS[:, 1]

def _hex_disjoint_fast(h1, h2):
    dx = h1[0] - h2[0]
    dy = h1[1] - h2[1]
    dist_sq = dx * dx + dy * dy
    if dist_sq > 4.5:
        return True
    a1, a2 = h1[2], h2[2]
    c1, s1 = math.cos(a1), math.sin(a1)
    c2, s2 = math.cos(a2), math.sin(a2)
    v1x = h1[0] + _UX * c1 - _UY * s1
    v1y = h1[1] + _UX * s1 + _UY * c1
    v2x = h2[0] + _UX * c2 - _UY * s2
    v2y = h2[1] + _UX * s2 + _UY * c2
    en1x = _ENX * c1 - _ENY * s1
    en1y = _ENX * s1 + _ENY * c1
    en2x = _ENX * c2 - _ENY * s2
    en2y = _ENX * s2 + _ENY * c2
    for i in range(6):
        nx, ny = en1x[i], en1y[i]
        p1 = v1x * nx + v1y * ny
        p2 = v2x * nx + v2y * ny
        if p1.max() < p2.min() or p2.max() < p1.min():
            return True
    for i in range(6):
        nx, ny = en2x[i], en2y[i]
        p1 = v1x * nx + v1y * ny
        p2 = v2x * nx + v2y * ny
        if p1.max() < p2.min() or p2.max() < p1.min():
            return True
    return False

def _is_valid(positions, angles_deg):
    angles_rad = np.radians(angles_deg)
    for i in range(N):
        for j in range(i + 1, N):
            h1 = (positions[i, 0], positions[i, 1], angles_rad[i])
            h2 = (positions[j, 0], positions[j, 1], angles_rad[j])
            if not _hex_disjoint_fast(h1, h2):
                return False
    return True

def _compute_min_outer_side(positions, angles_deg):
    angles_rad = np.radians(angles_deg)
    c = np.cos(angles_rad)
    s = np.sin(angles_rad)
    rx = _UX[None, :] * c[:, None] - _UY[None, :] * s[:, None]
    ry = _UX[None, :] * s[:, None] + _UY[None, :] * c[:, None]
    verts_x = rx + positions[:, 0:1]
    verts_y = ry + positions[:, 1:2]
    verts_flat = np.stack((verts_x, verts_y), axis=-1).reshape(-1, 2)
    center = np.mean(verts_flat, axis=0)
    offsets = verts_flat - center

    def eval_phi(phi_deg):
        phi = np.radians(phi_deg)
        normal_angles = phi + np.pi / 6 + np.arange(6) * np.pi / 3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / SQRT3) * max_projs
        return float(np.max(side_vals))

    phi_vals = np.radians(np.arange(0, 60, 0.5))
    normal_angles = phi_vals[:, None] + np.pi / 6 + np.arange(6) * np.pi / 3
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
    abs_proj = np.abs(proj)
    max_projs = np.max(abs_proj, axis=0)
    side_vals = (2.0 / SQRT3) * max_projs
    best_side = float(np.max(side_vals).min())
    best_idx = int(np.argmin(np.max(side_vals, axis=1)))
    best_phi = float(best_idx * 0.5)

    lo = max(0.0, best_phi - 0.5)
    hi = min(60.0, best_phi + 0.5)
    gr = (math.sqrt(5) - 1) / 2
    x1 = hi - gr * (hi - lo)
    x2 = lo + gr * (hi - lo)
    f1 = eval_phi(x1)
    f2 = eval_phi(x2)
    for _ in range(50):
        if f1 < f2:
            hi = x2
            x2 = x1
            f2 = f1
            x1 = hi - gr * (hi - lo)
            f1 = eval_phi(x1)
        else:
            lo = x1
            x1 = x2
            f1 = f2
            x2 = lo + gr * (hi - lo)
            f2 = eval_phi(x2)
    best_phi = (lo + hi) / 2.0
    best_side = min(best_side, eval_phi(best_phi))

    return best_side, best_phi, float(center[0]), float(center[1])

def _make_rigid_cluster():
    """
    Creates a rigid cluster of 12 unit hexagons.
    Lattice: a1 = (sqrt(3), 0), a2 = (sqrt(3)/2, 1.5)
    We select 12 points that form a compact shape.
    """
    a1 = np.array([SQRT3, 0.0])
    a2 = np.array([SQRT3 / 2.0, 1.5])
    
    # Coordinates for a 12-hexagon cluster
    # Center (0,0)
    # Ring 1: 6 neighbors
    # Ring 2: 5 more to make 12
    coords = [
        (0, 0), 
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, -1), (-2, 1), (2, 0), (-2, 0), (0, 2)
    ]
    
    pos = np.zeros((N, 2))
    for idx, (i, j) in enumerate(coords):
        pos[idx] = i * a1 + j * a2
        
    # Center the cluster at origin
    center = np.mean(pos, axis=0)
    pos -= center
    
    return pos

def _objective_rigid(params):
    """
    params: [x, y, rot, shear_x, shear_y]
    x, y: translation
    rot: rotation of the whole cluster
    shear_x, shear_y: deformation of the lattice to allow non-regular packing
    """
    x, y, rot, sx, sy = params
    
    # Base positions
    base_pos = _make_rigid_cluster()
    
    # Apply shear/deformation
    # This allows the "super-hexagon" to stretch/compress
    pos = base_pos.copy()
    pos[:, 0] *= (1.0 + sx)
    pos[:, 1] *= (1.0 + sy)
    
    # Apply rotation
    c, s = math.cos(rot), math.sin(rot)
    R = np.array([[c, -s], [s, c]])
    pos = pos @ R.T
    
    # Apply translation
    pos[:, 0] += x
    pos[:, 1] += y
    
    # All angles are 0 relative to the cluster rotation? 
    # Wait, the task allows each hexagon to rotate independently.
    # But the strategy says "fixed internal relative positions and angles".
    # So all hexagons have the same angle (rot).
    angles = np.full(N, math.degrees(rot))
    
    # Check validity
    if not _is_valid(pos, angles):
        # Penalty for overlap
        # Calculate minimum distance between centers vs min distance for disjoint
        # For unit hexagons, min distance is sqrt(3) for parallel sides, or 2 for vertices?
        # Actually, just return a large penalty if invalid.
        # To make it smooth, we could calculate penetration depth, but for Nelder-Mead,
        # a large constant penalty works if we start from valid points.
        return 100.0
    
    side, _, _, _ = _compute_min_outer_side(pos, angles)
    return side

def construct():
    t0 = time.time()
    
    # Initial rigid cluster
    base_pos = _make_rigid_cluster()
    
    # Initial parameters: [x, y, rot, shear_x, shear_y]
    # Start with identity transform
    x0 = 0.0
    y0 = 0.0
    rot0 = 0.0
    sx0 = 0.0
    sy0 = 0.0
    
    params0 = np.array([x0, y0, rot0, sx0, sy0])
    
    # Nelder-Mead options
    options = {
        'maxiter': 500,
        'maxfev': 500,
        'xatol': 1e-6,
        'fatol': 1e-6
    }
    
    best_params = params0.copy()
    best_side = float('inf')
    
    # Try a few different initial shears/rotations to escape local minima
    # Since the search space is small (5D), we can try multiple starts
    starts = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0.05, -0.05],
        [0, 0, 0, -0.05, 0.05],
        [0, 0, 0, 0.1, 0.1],
        [0, 0, 0, -0.1, -0.1],
        [0, 0, math.pi/6, 0, 0],
        [0, 0, math.pi/3, 0, 0],
    ]
    
    for start in starts:
        if time.time() - t0 > DEADLINE - 10.0:
            break
            
        p0 = np.array(start)
        try:
            res = minimize(_objective_rigid, p0, method='Nelder-Mead', options=options)
            if res.fun < best_side:
                best_side = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Final verification and extraction
    x, y, rot, sx, sy = best_params
    
    # Reconstruct positions
    pos = _make_rigid_cluster()
    pos[:, 0] *= (1.0 + sx)
    pos[:, 1] *= (1.0 + sy)
    c, s = math.cos(rot), math.sin(rot)
    R = np.array([[c, -s], [s, c]])
    pos = pos @ R.T
    pos[:, 0] += x
    pos[:, 1] += y
    
    angles = np.full(N, math.degrees(rot))
    
    if not _is_valid(pos, angles):
        # Fallback to a known valid configuration if optimization failed
        # Use the honeycomb from the original program
        a1 = np.array([SQRT3, 0.0])
        a2 = np.array([SQRT3 / 2.0, 1.5])
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        pos = np.zeros((N, 2))
        for idx, (i, j) in enumerate(lattice_coords):
            pos[idx] = i * a1 + j * a2
        angles = np.zeros(N)
        
    best_side, best_phi, best_cx, best_cy = _compute_min_outer_side(pos, angles)
    
    inner = []
    for i in range(N):
        inner.append([float(pos[i, 0]), float(pos[i, 1]), float(angles[i])])

    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)

    return inner, center, side, angle
