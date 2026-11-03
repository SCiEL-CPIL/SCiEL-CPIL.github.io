import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 140.0
SQRT3 = math.sqrt(3)

# Unit hexagon geometry
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_UX = _UNIT_HEX_VERTS[:, 0]
_UY = _UNIT_HEX_VERTS[:, 1]

# Edge normals for the unit hexagon (outward)
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i + 1) % 6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i + 1) % 6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx * _nx + _ny * _ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]
_ENX = _EDGE_NORMALS[:, 0]
_ENY = _EDGE_NORMALS[:, 1]


def _hex_disjoint_fast(h1, h2):
    """Exact SAT check for two unit hexagons. h = (cx, cy, angle_rad)."""
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


def _is_valid(positions, angles_rad):
    for i in range(N):
        for j in range(i + 1, N):
            h1 = (positions[i, 0], positions[i, 1], angles_rad[i])
            h2 = (positions[j, 0], positions[j, 1], angles_rad[j])
            if not _hex_disjoint_fast(h1, h2):
                return False
    return True


def _compute_min_outer_side(positions, angles_rad):
    """Compute the minimum enclosing regular hexagon side length and its orientation."""
    c = np.cos(angles_rad)
    s = np.sin(angles_rad)
    rx = _UX[None, :] * c[:, None] - _UY[None, :] * s[:, None]
    ry = _UX[None, :] * s[:, None] + _UY[None, :] * c[:, None]
    verts_x = rx + positions[:, 0:1]
    verts_y = ry + positions[:, 1:2]
    verts_flat = np.stack((verts_x, verts_y), axis=-1).reshape(-1, 2)
    center = np.mean(verts_flat, axis=0)
    offsets = verts_flat - center

    def eval_phi(phi):
        normal_angles = phi + np.pi / 6 + np.arange(6) * np.pi / 3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / SQRT3) * max_projs
        return float(np.max(side_vals))

    # Golden section search over [0, pi/3)
    lo = 0.0
    hi = math.pi / 3.0
    gr = (math.sqrt(5) - 1) / 2
    x1 = hi - gr * (hi - lo)
    x2 = lo + gr * (hi - lo)
    f1 = eval_phi(x1)
    f2 = eval_phi(x2)
    for _ in range(80):
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
    best_side = min(f1, f2, eval_phi(best_phi))
    return best_side, best_phi, float(center[0]), float(center[1])


def _soft_collision_penalty(positions, angles_rad, beta=30.0):
    """
    Smooth collision penalty using support functions of the Minkowski difference.
    For each pair (i,j), the minimum separation is computed via the support function
    of (hex_i - hex_j) in the direction from j to i. If this is < 0, they overlap.
    We use a soft max to make it differentiable.
    """
    penalty = 0.0
    c = np.cos(angles_rad)
    s = np.sin(angles_rad)
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist_sq = dx * dx + dy * dy
            if dist_sq < 1e-12:
                dist = 1e-12
            else:
                dist = math.sqrt(dist_sq)
            
            # If centers are far apart, no overlap possible
            if dist > 2.0 * SQRT3 + 0.5:
                continue
            
            # Direction from j to i
            if dist > 1e-12:
                dir_x = dx / dist
                dir_y = dy / dist
            else:
                dir_x, dir_y = 1.0, 0.0
            
            # Support function of hex_i in direction (dir_x, dir_y)
            # h_i(dir) = max over vertices v of dot(v, dir)
            # For a unit hexagon at center p_i with angle a_i:
            #   h_i(dir) = max_k dot(R(a_i)*v_k + p_i, dir)
            #            = max_k dot(R(a_i)*v_k, dir) + dot(p_i, dir)
            # Similarly for hex_j.
            # The Minkowski difference hex_i - hex_j has support:
            #   h_diff(dir) = h_i(dir) + h_j(-dir)
            # The minimum separation in direction dir is:
            #   sep = h_i(dir) + h_j(-dir) - dist  (approximately, for the direction between centers)
            # Actually, the correct formulation: the hexagons are disjoint iff for all directions u,
            #   h_i(u) + h_j(-u) >= 0  (when centered at origin relative to each other)
            # For the specific direction between centers, the overlap is:
            #   overlap = dist - (h_i(dir) + h_j(-dir))
            # But this only checks one direction. For a smooth penalty, we check the
            # 6 edge-normal directions of each hexagon.
            
            # For simplicity and differentiability, use the center-distance-based penalty
            # with the effective "radius" being the max support in the separating direction.
            c1, s1 = c[i], s[i]
            c2, s2 = c[j], s[j]
            
            # Support of hex_i in direction (dir_x, dir_y)
            # R(a_i)*v_k rotated, dot with (dir_x, dir_y)
            # = dot(v_k, R(-a_i)*(dir_x, dir_y))
            rot_dir_x_i = dir_x * c1 + dir_y * s1
            rot_dir_y_i = -dir_x * s1 + dir_y * c1
            h_i = float(np.max(_UX * rot_dir_x_i + _UY * rot_dir_y_i))
            
            # Support of hex_j in direction (-dir_x, -dir_y)
            rot_dir_x_j = -dir_x * c2 + (-dir_y) * s2
            rot_dir_y_j = -(-dir_x) * s2 + (-dir_y) * c2
            # Wait: R(-a_j) * (-dir) = R(-a_j) * (-dir_x, -dir_y)
            # Let me redo: h_j(-dir) = max_k dot(R(a_j)*v_k, -dir) = max_k dot(v_k, R(-a_j)*(-dir))
            neg_dir_x, neg_dir_y = -dir_x, -dir_y
            rot_neg_dir_x_j = neg_dir_x * c2 + neg_dir_y * s2
            rot_neg_dir_y_j = -neg_dir_x * s2 + neg_dir_y * c2
            h_j_neg = float(np.max(_UX * rot_neg_dir_x_j + _UY * rot_neg_dir_y_j))
            
            # Separation in this direction
            sep = dist - (h_i + h_j_neg)
            
            # Soft penalty: if sep < 0, penalize
            # Use softplus-like: -log(1 + exp(-beta*sep)) / beta approximates max(0, -sep)
            # Or simply: max(0, -sep)^2 with a smooth approximation
            if sep < 0:
                penalty += sep * sep
            else:
                # Add a tiny smooth term for gradient continuity
                penalty += 0.0
    
    return penalty


def _smooth_objective_LBFGS(params):
    """
    params: (37,) = [x0,y0,a0, ..., x11,y11,a11, phi]
    Returns smooth objective value.
    """
    positions = params[0:36:3].reshape(N, 1)  # x coords
    pys = params[1:36:3].reshape(N, 1)  # y coords
    angles = params[2:36:3]
    phi = params[36]
    
    pos = np.column_stack((positions.flatten(), pys.flatten()))
    
    # Compute enclosing hexagon side (smooth)
    c = np.cos(angles)
    s = np.sin(angles)
    rx = _UX[None, :] * c[:, None] - _UY[None, :] * s[:, None]
    ry = _UX[None, :] * s[:, None] + _UY[None, :] * c[:, None]
    verts_x = rx + pos[:, 0:1]
    verts_y = ry + pos[:, 1:2]
    verts_flat = np.stack((verts_x, verts_y), axis=-1).reshape(-1, 2)
    center = np.mean(verts_flat, axis=0)
    offsets = verts_flat - center
    
    normal_angles = phi + np.pi / 6 + np.arange(6) * np.pi / 3
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
    abs_proj = np.abs(proj)
    max_projs = np.max(abs_proj, axis=0)
    side_vals = (2.0 / SQRT3) * max_projs
    
    # Log-sum-exp for smooth max
    x_max = side_vals.max()
    outer_side_smooth = x_max + np.log(np.exp(20.0 * (side_vals - x_max)).sum()) / 20.0
    
    # Collision penalty
    coll_pen = _soft_collision_penalty(pos, angles, beta=30.0)
    
    lam = 100.0
    total = outer_side_smooth + lam * coll_pen
    return total


def _objective_with_grad(params):
    """Objective and numerical gradient for L-BFGS-B."""
    f = _smooth_objective_LBFGS(params)
    # Numerical gradient (central difference)
    grad = np.zeros(len(params))
    h = 1e-6
    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += h
        params_minus[i] -= h
        grad[i] = (_smooth_objective_LBFGS(params_plus) - _smooth_objective_LBFGS(params_minus)) / (2 * h)
    return f, grad


def _make_honeycomb():
    """Standard honeycomb packing of 12 hexagons."""
    a1 = np.array([SQRT3, 0.0])
    a2 = np.array([SQRT3 / 2.0, 1.5])
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    for idx, (i, j) in enumerate(lattice_coords):
        pos[idx] = i * a1 + j * a2
        ang[idx] = 0.0
    return pos, ang


def _make_ring():
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    for i in range(6):
        theta = i * np.pi / 3
        pos[i, 0] = 2.5 * SQRT3 * np.cos(theta)
        pos[i, 1] = 2.5 * SQRT3 * np.sin(theta)
    for i in range(6):
        theta = (i + 0.5) * np.pi / 3
        pos[6 + i, 0] = 1.2 * SQRT3 * np.cos(theta)
        pos[6 + i, 1] = 1.2 * SQRT3 * np.sin(theta)
    return pos, ang


def _make_compact_hex_cluster():
    a1 = np.array([SQRT3, 0.0])
    a2 = np.array([SQRT3 / 2.0, 1.5])
    coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, -1), (-2, 1), (2, 0), (-2, 0), (0, 2),
    ]
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    for idx, (i, j) in enumerate(coords):
        pos[idx] = i * a1 + j * a2
        ang[idx] = 0.0
    return pos, ang


def _make_two_rows():
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    for i in range(6):
        pos[i, 0] = i * SQRT3
        pos[i, 1] = 0.0
    for i in range(6):
        pos[6 + i, 0] = i * SQRT3 + SQRT3 / 2
        pos[6 + i, 1] = 1.5
    return pos, ang


def _pos_ang_to_params(pos, ang_rad, phi):
    params = np.zeros(37)
    for i in range(N):
        params[3 * i] = pos[i, 0]
        params[3 * i + 1] = pos[i, 1]
        params[3 * i + 2] = ang_rad[i]
    params[36] = phi
    return params


def _params_to_pos_ang(params):
    pos = np.zeros((N, 2))
    ang_rad = np.zeros(N)
    for i in range(N):
        pos[i, 0] = params[3 * i]
        pos[i, 1] = params[3 * i + 1]
        ang_rad[i] = params[3 * i + 2]
    phi = params[36]
    return pos, ang_rad, phi


def _local_search_refine(pos, ang_rad, phi, time_limit, t_start):
    """
    Local search: try small perturbations to each parameter, keep if it improves
    the exact objective (enclosing side) while maintaining validity.
    """
    best_side, _, _, _ = _compute_min_outer_side(pos, ang_rad)
    best_pos = pos.copy()
    best_ang = ang_rad.copy()
    best_phi = phi
    
    step = 0.02
    min_step = 1e-5
    no_improve = 0
    max_no_improve = 15
    
    while time.time() - t_start < time_limit and step > min_step:
        improved = False
        # Try perturbing each position and angle
        for i in range(N):
            if time.time() - t_start > time_limit:
                break
            for dx in [step, -step, 0]:
                for dy in [step, -step, 0]:
                    if dx == 0 and dy == 0:
                        continue
                    trial_pos = best_pos.copy()
                    trial_pos[i, 0] += dx
                    trial_pos[i, 1] += dy
                    if _is_valid(trial_pos, best_ang):
                        s, _, _, _ = _compute_min_outer_side(trial_pos, best_ang)
                        if s < best_side - 1e-12:
                            best_pos = trial_pos
                            best_side = s
                            improved = True
            for da in [step * 10, -step * 10]:  # angle step in radians
                trial_ang = best_ang.copy()
                trial_ang[i] += da
                if _is_valid(best_pos, trial_ang):
                    s, _, _, _ = _compute_min_outer_side(best_pos, trial_ang)
                    if s < best_side - 1e-12:
                        best_ang = trial_ang
                        best_side = s
                        improved = True
        
        # Try perturbing phi
        for dphi in [step * 5, -step * 5]:
            if time.time() - t_start > time_limit:
                break
            trial_phi = best_phi + dphi
            s, _, _, _ = _compute_min_outer_side(best_pos, best_ang)
            # Re-evaluate with new phi (the enclosing hexagon orientation)
            # Actually _compute_min_outer_side already finds the optimal phi
            # So perturbing phi separately isn't directly useful here.
            # Skip phi perturbation in local search since _compute_min_outer_side handles it.
        
        if not improved:
            no_improve += 1
            if no_improve >= max_no_improve:
                step *= 0.5
                no_improve = 0
        else:
            no_improve = 0
    
    return best_pos, best_ang, best_side, best_phi


def construct():
    t0 = time.time()
    
    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    # Initial configurations
    inits = [
        _make_honeycomb(),
        _make_compact_hex_cluster(),
        _make_ring(),
        _make_two_rows(),
    ]
    
    for init_idx, (pos0, ang0_deg) in enumerate(inits):
        if time.time() - t0 > DEADLINE - 30.0:
            break
        
        ang0 = np.radians(ang0_deg)
        
        # Check validity
        if not _is_valid(pos0, ang0):
            # Try to fix by small random perturbations
            fixed = False
            for _ in range(200):
                trial_pos = pos0.copy()
                trial_ang = ang0.copy()
                idx = np.random.randint(N)
                trial_pos[idx, 0] += np.random.uniform(-0.1, 0.1)
                trial_pos[idx, 1] += np.random.uniform(-0.1, 0.1)
                trial_ang[idx] += np.random.uniform(-0.05, 0.05)
                if _is_valid(trial_pos, trial_ang):
                    pos0 = trial_pos
                    ang0 = trial_ang
                    fixed = True
                    break
            if not fixed:
                continue
        
        # Evaluate initial
        s0, phi0, cx0, cy0 = _compute_min_outer_side(pos0, ang0)
        if s0 < best_side:
            best_side = s0
            best_positions = pos0.copy()
            best_angles = ang0.copy()
            best_phi = phi0
            best_cx = cx0
            best_cy = cy0
        
        # L-BFGS-B optimization
        remaining = DEADLINE - 25.0 - (time.time() - t0)
        if remaining < 5.0:
            break
        
        params0 = _pos_ang_to_params(pos0, ang0, phi0)
        
        # Bounds: positions in [-10, 10], angles in [-pi, pi], phi in [0, pi/3]
        bounds = [(-10, 10)] * 36 + [(0, math.pi / 3)]
        # Actually angles should be in [-pi, pi]
        bounds = []
        for i in range(N):
            bounds.append((-10, 10))  # x
            bounds.append((-10, 10))  # y
            bounds.append((-math.pi, math.pi))  # angle
        bounds.append((0, math.pi / 3))  # phi
        
        # Use L-BFGS-B with numerical gradient
        # We need to be careful with time - each function evaluation involves
        # computing the smooth objective which is O(N^2) for collisions
        maxfun = int(min(remaining * 15, 3000))  # ~15 evals/sec budget
        
        try:
            result = minimize(
                _smooth_objective_LBFGS,
                params0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxfun': maxfun, 'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
        except Exception:
            result = None
        
        if result is not None and result.fun < 1e10:
            lbfgs_pos, lbfgs_ang, lbfgs_phi = _params_to_pos_ang(result.x)
            if _is_valid(lbfgs_pos, lbfgs_ang):
                lbfgs_side, lbfgs_phi_opt, lbfgs_cx, lbfgs_cy = _compute_min_outer_side(lbfgs_pos, lbfgs_ang)
                if lbfgs_side < best_side:
                    best_side = lbfgs_side
                    best_positions = lbfgs_pos.copy()
                    best_angles = lbfgs_ang.copy()
                    best_phi = lbfgs_phi_opt
                    best_cx = lbfgs_cx
                    best_cy = lbfgs_cy
        
        # Local search refinement
        remaining2 = DEADLINE - 15.0 - (time.time() - t0)
        if remaining2 > 3.0 and best_positions is not None:
            polish_time = min(remaining2 * 0.4, 20.0)
            t_ls = time.time()
            ls_pos, ls_ang, ls_side, ls_phi = _local_search_refine(
                best_positions, best_angles, best_phi, polish_time, t_ls)
            if ls_side < best_side:
                best_side = ls_side
                best_positions = ls_pos
                best_angles = ls_ang
                best_phi = ls_phi
                # Recompute center
                _, _, best_cx, best_cy = _compute_min_outer_side(best_positions, best_angles)
    
    # Final verification
    if best_positions is None or not _is_valid(best_positions, best_angles):
        pos0, ang0_deg = _make_honeycomb()
        ang0 = np.radians(ang0_deg)
        # Try to fix
        for _ in range(500):
            if _is_valid(pos0, ang0):
                break
            idx = np.random.randint(N)
            pos0[idx, 0] += np.random.uniform(-0.1, 0.1)
            pos0[idx, 1] += np.random.uniform(-0.1, 0.1)
            ang0[idx] += np.random.uniform(-0.05, 0.05)
        best_positions = pos0
        best_angles = ang0
        best_side, best_phi, best_cx, best_cy = _compute_min_outer_side(best_positions, best_angles)
    
    # Convert angles to degrees for output
    inner = []
    for i in range(N):
        inner.append([
            float(best_positions[i, 0]),
            float(best_positions[i, 1]),
            float(math.degrees(best_angles[i]))
        ])
    
    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(math.degrees(best_phi))
    
    return inner, center, side, angle
