import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 12
DEADLINE = 145.0

_SQRT3 = math.sqrt(3)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))

_UX = _UNIT_HEX_VERTS[:, 0]
_UY = _UNIT_HEX_VERTS[:, 1]

# Honeycomb lattice coordinates for 12 hexagons
# This configuration is a "flower" of 7 + a ring of 5? 
# Standard honeycomb cluster of 19 is too big.
# Let's use a compact cluster.
# A hexagon of side 2 has 19 cells. We need 12.
# A hexagon of side 1 has 7 cells.
# Let's use a specific set of lattice vectors that form a compact shape.
# Coordinates in basis (a1, a2) where a1=(sqrt(3), 0), a2=(sqrt(3)/2, 1.5)
# (0,0)
# (1,0), (-1,0)
# (0,1), (0,-1)
# (1,-1), (-1,1)
# (2,-1), (-2,1)
# (1,1), (-1,-1)
# (2,0)
# Total 12. This is a hexagon of side 2 missing the 7 "corner" cells?
# Side 2 hexagon has 1+6+12 = 19 cells.
# Removing the 7 outermost corners (distance 2 from center) leaves 12?
# Distance 0: 1
# Distance 1: 6
# Distance 2: 12
# Total 19.
# If we take all distance 0 and 1 (7 cells) and 5 from distance 2, we get 12.
# The 5 from distance 2 can be chosen to maintain symmetry or compactness.
# Let's just use the 12 coordinates listed above.
_LATTICE_COORDS = [
    (0, 0),
    (1, 0), (-1, 0),
    (0, 1), (0, -1),
    (1, -1), (-1, 1),
    (2, -1), (-2, 1),
    (1, 1), (-1, -1),
    (2, 0)
]

def _get_positions_and_angles(params):
    """
    params: [a1x, a1y, a2x, a2y, ox, oy, angle_offset]
    a1, a2: lattice basis vectors
    ox, oy: origin offset
    angle_offset: rotation of the entire lattice (0 or 30 deg effectively handled by basis)
    """
    a1 = np.array([params[0], params[1]])
    a2 = np.array([params[2], params[3]])
    origin = np.array([params[4], params[5]])
    
    # We fix the rotation of individual hexagons to 0 degrees relative to the lattice?
    # Actually, in a honeycomb, all hexagons have the same orientation.
    # We can allow a global rotation of the whole packing.
    # But the basis vectors a1, a2 define the orientation.
    # Standard honeycomb: a1 = (sqrt(3)*s, 0), a2 = (sqrt(3)/2*s, 1.5*s) where s is scale.
    # If we vary a1, a2 freely, we might distort the hexagons or create gaps/overlaps.
    # To ensure valid hexagons, the distance between centers must be at least sqrt(3) * scale.
    # Here scale is 1. So min distance is sqrt(3).
    
    pos = np.zeros((N, 2))
    for i, (i, j) in enumerate(_LATTICE_COORDS):
        pos[i] = origin + i * a1 + j * a2
        
    # Angles are all 0 (aligned with lattice)
    ang = np.zeros(N)
    return pos, ang

def _compute_min_outer_side_smooth(pos, ang, beta=10.0):
    """
    Compute a smooth approximation of the minimum enclosing regular hexagon side length.
    """
    angles_rad = np.radians(ang)
    c = np.cos(angles_rad)
    s = np.sin(angles_rad)
    rx = _UX[None, :] * c[:, None] - _UY[None, :] * s[:, None]
    ry = _UX[None, :] * s[:, None] + _UY[None, :] * c[:, None]
    verts_x = rx + pos[:, 0:1]
    verts_y = ry + pos[:, 1:2]
    verts_flat = np.stack((verts_x, verts_y), axis=-1).reshape(-1, 2)
    center = np.mean(verts_flat, axis=0)
    offsets = verts_flat - center

    # Sample 12 angles for the outer hexagon orientation
    phi_vals = np.radians(np.arange(0, 60, 5))
    normal_angles = phi_vals[:, None] + np.pi / 6 + np.arange(6) * np.pi / 3
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
    abs_proj = np.abs(proj)
    max_projs = np.max(abs_proj, axis=0)
    side_vals = (2.0 / _SQRT3) * max_projs
    
    # Smooth max over phi
    m = side_vals.max(axis=1)
    # m is shape (12,)
    # We want min over phi of max over sides
    # This is a min-max problem.
    # For a fixed phi, the side length is determined.
    # We want the phi that minimizes this.
    # So we take the min of the 12 values.
    # To make it smooth, we use a soft-min.
    # soft-min(x) = -logsumexp(-x/beta)*beta
    x = side_vals.max(axis=1) # shape (12,)
    m_x = x.max()
    soft_min = -m_x - np.log(np.sum(np.exp(-(x - m_x) / beta))) * beta
    
    return float(soft_min)

def _overlap_penalty_smooth(pos, ang, beta=10.0):
    """
    Smooth penalty for overlaps between inner hexagons.
    Uses distance between centers. Since all angles are same, 
    overlap depends only on distance and relative position.
    Actually, if angles are same, we can just check distance < sqrt(3) + epsilon?
    No, hexagons can interlock if rotated, but here they are aligned.
    For aligned hexagons on a honeycomb lattice, the minimum center distance 
    for non-overlap is sqrt(3).
    So penalty = sum( softplus(sqrt(3) - dist) )
    """
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = math.sqrt(dx*dx + dy*dy)
            # Min distance is sqrt(3)
            violation = _SQRT3 - dist
            if violation > 0:
                # Soft penalty
                penalty += math.log1p(math.exp(beta * violation)) / beta
    return penalty

def _objective(params):
    pos, ang = _get_positions_and_angles(params)
    side = _compute_min_outer_side_smooth(pos, ang)
    pen = _overlap_penalty_smooth(pos, ang)
    return side + 10.0 * pen

def _exact_side(pos, ang):
    """Exact calculation for final verification."""
    angles_rad = np.radians(ang)
    c = np.cos(angles_rad)
    s = np.sin(angles_rad)
    rx = _UX[None, :] * c[:, None] - _UY[None, :] * s[:, None]
    ry = _UX[None, :] * s[:, None] + _UY[None, :] * c[:, None]
    verts_x = rx + pos[:, 0:1]
    verts_y = ry + pos[:, 1:2]
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
        side_vals = (2.0 / _SQRT3) * max_projs
        return np.max(side_vals)

    # Coarse search
    phi_vals = np.radians(np.arange(0, 60, 1))
    normal_angles = phi_vals[:, None] + np.pi / 6 + np.arange(6) * np.pi / 3
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
    abs_proj = np.abs(proj)
    max_projs = np.max(abs_proj, axis=0)
    side_vals = (2.0 / _SQRT3) * max_projs
    best_side = np.max(side_vals).min()
    best_idx = np.argmin(np.max(side_vals, axis=1))
    best_phi = float(best_idx)

    # Refine
    try:
        lo = max(0.0, best_phi - 1.0)
        hi = min(60.0, best_phi + 1.0)
        res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                              options={'xatol': 1e-10, 'maxiter': 100})
        if res.success and res.fun < best_side:
            best_side = float(res.fun)
            best_phi = float(res.x)
    except Exception:
        pass

    return float(best_side), float(best_phi), float(center[0]), float(center[1])

def _is_valid(pos, ang):
    """Check if configuration is valid (no overlaps)."""
    # For aligned hexagons, just check center distances
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx*dx + dy*dy
            if dist_sq < _SQRT3 * _SQRT3 - 1e-9:
                return False
    return True

def construct():
    t0 = time.time()
    
    # Initial guess: standard honeycomb lattice
    a1_init = np.array([_SQRT3, 0.0])
    a2_init = np.array([_SQRT3 / 2.0, 1.5])
    origin_init = np.array([0.0, 0.0])
    
    init_params = np.concatenate([a1_init, a2_init, origin_init])
    
    best_params = init_params.copy()
    best_score = _objective(best_params)
    
    # Optimize using L-BFGS-B
    # We allow the basis vectors to change slightly to "squeeze" the packing
    # But we must ensure they don't become degenerate or cause overlaps
    # The penalty function handles overlaps.
    
    try:
        res = minimize(
            _objective,
            init_params,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Try a few other starting points (rotated lattice)
    for angle_deg in [30, -30, 15, -15]:
        if time.time() - t0 > DEADLINE - 10:
            break
        theta = np.radians(angle_deg)
        R = np.array([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta), math.cos(theta)]])
        a1_rot = R @ a1_init
        a2_rot = R @ a2_init
        init_rot = np.concatenate([a1_rot, a2_rot, origin_init])
        
        try:
            res = minimize(
                _objective,
                init_rot,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    # Extract best solution
    pos, ang = _get_positions_and_angles(best_params)
    
    # Verify validity
    if not _is_valid(pos, ang):
        # Fallback to standard honeycomb
        pos = np.zeros((N, 2))
        for i, (i, j) in enumerate(_LATTICE_COORDS):
            pos[i] = i * a1_init + j * a2_init
        ang = np.zeros(N)
        
    # Compute exact side
    side, phi, cx, cy = _exact_side(pos, ang)
    
    inner = []
    for i in range(N):
        inner.append([float(pos[i, 0]), float(pos[i, 1]), float(ang[i])])
        
    center = (float(cx), float(cy))
    
    return inner, center, side, phi

