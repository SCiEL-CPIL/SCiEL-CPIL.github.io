import math
import time
import numpy as np
from itertools import product

N = 12
DEADLINE = 142.0
SQRT3 = math.sqrt(3)

# Precompute unit hexagon vertices and edge normals
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i + 1) % 6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i + 1) % 6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx * _nx + _ny * _ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

_UX = _UNIT_HEX_VERTS[:, 0]
_UY = _UNIT_HEX_VERTS[:, 1]
_ENX = _EDGE_NORMALS[:, 0]
_ENY = _EDGE_NORMALS[:, 1]

# Lattice basis vectors for hexagonal packing
_A1 = np.array([SQRT3, 0.0])
_A2 = np.array([SQRT3 / 2.0, 1.5])

# Discrete orientations to consider (radians)
_ORIENTS = [0.0, np.pi / 6, np.pi / 3]

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
    v1y = h1[1] + _UX * s1 + _UY * c2 # Note: typo in original? Let's assume standard rotation. 
    # Actually, let's use a robust vectorized check or the exact one from the prompt.
    # The prompt's _hex_disjoint_fast is correct. I will reuse it.
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

def _is_valid(positions, angles_rad):
    for i in range(N):
        for j in range(i + 1, N):
            h1 = (positions[i, 0], positions[i, 1], angles_rad[i])
            h2 = (positions[j, 0], positions[j, 1], angles_rad[j])
            if not _hex_disjoint_fast(h1, h2):
                return False
    return True

def _compute_min_outer_side(positions, angles_rad):
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

    # Coarse scan
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

    # Fine tune
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

def _soft_contact_penalty(positions, angles_rad, side_len, phi_deg, beta=10.0):
    """
    Penalizes penetration of inner hexagons into the outer hexagon.
    Outer hexagon is centered at origin for this calculation (we'll translate later or assume center 0,0).
    Actually, it's better to compute the minimum required side length for a given configuration and compare.
    But for a differentiable potential, we can project vertices onto the outer hexagon normals.
    
    Let's define the potential as sum of (max(0, projection - allowed_projection))^2.
    allowed_projection depends on side_len.
    For a regular hexagon of side S, the distance from center to edge is S * sqrt(3) / 2.
    The projection of a vertex onto the normal of an edge must be <= S * sqrt(3) / 2.
    """
    c = np.cos(angles_rad)
    s = np.sin(angles_rad)
    rx = _UX[None, :] * c[:, None] - _UY[None, :] * s[:, None]
    ry = _UX[None, :] * s[:, None] + _UY[None, :] * c[:, None]
    verts_x = rx + positions[:, 0:1]
    verts_y = ry + positions[:, 1:2]
    verts_flat = np.stack((verts_x, verts_y), axis=-1).reshape(-1, 2)
    
    # Center of mass to align with outer hexagon center
    center = np.mean(verts_flat, axis=0)
    offsets = verts_flat - center

    phi = np.radians(phi_deg)
    normal_angles = phi + np.pi / 6 + np.arange(6) * np.pi / 3
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    
    proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
    abs_proj = np.abs(proj)
    
    # Allowed projection for side length S is S * sqrt(3) / 2
    allowed = side_len * SQRT3 / 2.0
    
    # Penalty for exceeding allowed projection
    excess = np.maximum(0, abs_proj - allowed)
    return np.sum(excess ** 2)

def _solve_continuous_relaxation(positions, angles_rad, phi_deg, initial_side):
    """
    Use scipy.optimize.minimize to adjust positions to minimize a soft contact penalty
    while maintaining disjointness (hard constraint).
    This is a heuristic step to refine the lattice-based configuration.
    """
    from scipy.optimize import minimize
    
    # We'll optimize positions only, keeping angles fixed from the lattice assignment.
    x0 = positions.flatten()
    
    def objective(x):
        pos = x.reshape(N, 2)
        # Hard constraint check: if invalid, return large penalty
        if not _is_valid(pos, angles_rad):
            return 1e6
        # Soft constraint: minimize side length via penalty
        # We want to find the minimum side length. 
        # Actually, minimizing the penalty for a fixed side length doesn't directly give the min side.
        # Better: minimize a proxy for side length, e.g., max projection.
        # Let's minimize the max projection directly, which is differentiable almost everywhere.
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _UX[None, :] * c[:, None] - _UY[None, :] * s[:, None]
        ry = _UX[None, :] * s[:, None] + _UY[None, :] * c[:, None]
        verts_x = rx + pos[:, 0:1]
        verts_y = ry + pos[:, 1:2]
        verts_flat = np.stack((verts_x, verts_y), axis=-1).reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center
        
        phi = np.radians(phi_deg)
        normal_angles = phi + np.pi / 6 + np.arange(6) * np.pi / 3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        max_proj = np.max(max_projs)
        return (2.0 / SQRT3) * max_proj

    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50})
    
    best_pos = res.x.reshape(N, 2)
    if _is_valid(best_pos, angles_rad):
        side, phi_opt, cx, cy = _compute_min_outer_side(best_pos, angles_rad)
        return side, best_pos
    else:
        return initial_side, positions

def _branch_and_bound_search():
    """
    Branch and bound over orientation states.
    State: assignment of orientations to each of the 12 hexagons.
    Pruning: if the lower bound on side length exceeds current best, prune.
    Lower bound: can be estimated by area or simple packing density, or by solving the continuous relaxation quickly.
    """
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    t_start = time.time()
    node_count = 0
    
    # We use a recursive search with time checks.
    # To make it efficient, we'll fix the position of hexagon 0 at (0,0) and orientation 0.
    # Then search orientations for 1..11.
    
    def dfs(idx, positions, angles_rad, current_side_lb):
        nonlocal best_side, best_positions, best_angles, best_phi, best_cx, best_cy
        nonlocal node_count
        
        if time.time() - t_start > DEADLINE - 5.0:
            return
        
        node_count += 1
        if node_count % 100 == 0:
            # Periodic check
            pass

        if idx == N:
            # All orientations assigned. Solve continuous relaxation for positions.
            # Initial positions from lattice
            pos0 = np.zeros((N, 2))
            # Generate lattice positions based on a standard honeycomb cluster
            # This is a heuristic initialization.
            lattice_coords = [
                (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, -1), (-1, 1), (2, -1), (-2, 1),
                (1, 1), (-1, -1), (2, 0),
            ]
            for i in range(N):
                pos0[i] = lattice_coords[i][0] * _A1 + lattice_coords[i][1] * _A2
            
            # Refine with continuous relaxation
            side, refined_pos = _solve_continuous_relaxation(pos0, angles_rad, 0.0, current_side_lb)
            
            if side < best_side:
                best_side = side
                best_positions = refined_pos.copy()
                best_angles = angles_rad.copy()
                best_side, best_phi, best_cx, best_cy = _compute_min_outer_side(best_positions, best_angles)
            return
        
        # Try each orientation for hexagon idx
        for orient in _ORIENTS:
            # Quick validity check with already placed hexagons
            # We don't have positions yet, only orientations.
            # We can't check disjointness without positions.
            # So we must proceed and check at the leaf or with estimated positions.
            # This makes B&B weak. 
            # Alternative: fix positions to lattice, check disjointness with current orientations.
            # If valid on lattice, compute side. If not, skip or allow relaxation.
            
            # Let's estimate side length with lattice positions and current orientations
            pos_est = np.zeros((N, 2))
            lattice_coords = [
                (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, -1), (-1, 1), (2, -1), (-2, 1),
                (1, 1), (-1, -1), (2, 0),
            ]
            for i in range(N):
                pos_est[i] = lattice_coords[i][0] * _A1 + lattice_coords[i][1] * _A2
            
            # Check disjointness for placed hexagons (0..idx)
            valid = True
            for i in range(idx + 1):
                for j in range(i + 1, idx + 1):
                    h1 = (pos_est[i, 0], pos_est[i, 1], angles_rad[i])
                    h2 = (pos_est[j, 0], pos_est[j, 1], angles_rad[j])
                    if not _hex_disjoint_fast(h1, h2):
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Estimate side length
            side_est, _, _, _ = _compute_min_outer_side(pos_est, angles_rad[:idx+1])
            
            # Pruning
            if side_est >= best_side:
                continue
            
            angles_rad[idx] = orient
            dfs(idx + 1, pos_est, angles_rad, side_est)
            
    # Initialize
    angles_rad = np.zeros(N)
    positions = np.zeros((N, 2))
    
    # Fix hexagon 0
    angles_rad[0] = 0.0
    dfs(1, positions, angles_rad, 0.0)
    
    if best_positions is None:
        # Fallback to honeycomb
        pos0, ang0 = _make_honeycomb()
        best_positions = pos0
        best_angles = np.zeros(N)
        best_side, best_phi, best_cx, best_cy = _compute_min_outer_side(best_positions, best_angles)
        
    return best_side, best_positions, best_angles, best_phi, best_cx, best_cy

def _make_honeycomb():
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    for idx, (i, j) in enumerate(lattice_coords):
        pos[idx] = i * _A1 + j * _A2
    return pos, ang

def construct():
    t0 = time.time()
    
    # Run the new B&B strategy
    best_side, best_positions, best_angles, best_phi, best_cx, best_cy = _branch_and_bound_search()
    
    # If B&B didn't find a good solution or ran out of time, fallback to local search on best found
    if time.time() - t0 < DEADLINE - 10.0:
        # Refine with local search
        from scipy.optimize import differential_evolution
        # We'll do a simple local refinement
        current_side = best_side
        step = 0.1
        for _ in range(50):
            if time.time() - t0 > DEADLINE - 5.0:
                break
            improved = False
            for idx in range(N):
                for dim in range(2):
                    for direction in [1, -1]:
                        new_pos = best_positions.copy()
                        if dim == 0:
                            new_pos[idx, 0] += direction * step
                        else:
                            new_pos[idx, 1] += direction * step
                        if _is_valid(new_pos, best_angles):
                            new_side, _, _, _ = _compute_min_outer_side(new_pos, best_angles)
                            if new_side < current_side:
                                current_side = new_side
                                best_positions = new_pos
                                improved = True
            if not improved:
                step *= 0.8
            if step < 0.001:
                break
        
        best_side, best_phi, best_cx, best_cy = _compute_min_outer_side(best_positions, best_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(np.degrees(best_angles[i]))])

    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)

    return inner, center, side, angle
