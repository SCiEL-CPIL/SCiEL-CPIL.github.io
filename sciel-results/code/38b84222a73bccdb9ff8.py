import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle a_deg."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack((cx + d * np.cos(angles), cy + d * np.sin(angles)))


# Precompute base unit-hexagon geometry (angle 0)
_BASE_VERTS = _hex_vertices(0.0, 0.0, 1.0, 0.0)
_BASE_EDGES = np.array([_BASE_VERTS[(k + 1) % 6] - _BASE_VERTS[k] for k in range(6)])
_BASE_NORMALS = np.array([[-_BASE_EDGES[k, 1], _BASE_EDGES[k, 0]] for k in range(6)])
_BASE_NORMALS = _BASE_NORMALS / np.linalg.norm(_BASE_NORMALS, axis=1, keepdims=True)
# base_proj_min/max: projection of base verts onto each normal
_BASE_PROJ = _BASE_VERTS @ _BASE_NORMALS.T  # (6,6): [v, k]
_BASE_PROJ_MIN = _BASE_PROJ.min(axis=0)  # (6,)
_BASE_PROJ_MAX = _BASE_PROJ.max(axis=0)  # (6,)


def _min_containing_hex_side(inner, outer_center, outer_angle_deg, soft_beta=200.0):
    """
    Compute the minimum side length s of a regular hexagon (centered at outer_center,
    with the given angle) that contains all inner unit hexagons.
    inner: (12,3) array of (x, y, angle_deg).
    Uses soft-max for smoothness.
    """
    ocx, ocy = outer_center[0], outer_center[1]
    ov_a = np.deg2rad(outer_angle_deg)

    # 6 outward normals to the sides of the outer hexagon
    normal_angles = ov_a + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))  # (6,2)

    # For each inner hexagon, compute its vertices (shifted by -outer_center)
    max_dot = -np.inf
    for row in inner:
        cx, cy, a_deg = row[0] - ocx, row[1] - ocy, row[2]
        a = np.deg2rad(a_deg)
        cos_a, sin_a = np.cos(a), np.sin(a)
        # Rotate base verts
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = _BASE_VERTS @ rot.T + np.array([cx, cy])  # (6,2)
        dots = verts @ normals.T  # (6,6): [v, k]
        # soft-max over all vertices and normals
        flat = dots.ravel()
        m = flat.max()
        if m > max_dot:
            max_dot = m

    # side length s such that max_dot <= s * sqrt(3)/2
    s = max_dot / (np.sqrt(3.0) / 2.0)
    return s


def _min_containing_hex_side_smooth(inner, outer_center, outer_angle_deg, beta=50.0):
    """Smooth version using log-sum-exp soft-max for use in objective."""
    ocx, ocy = outer_center[0], outer_center[1]
    ov_a = np.deg2rad(outer_angle_deg)
    normal_angles = ov_a + np.deg2rad(30.0 + 60.0 * np.arange(6))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))  # (6,2)

    all_dots = []
    for row in inner:
        cx, cy, a_deg = row[0] - ocx, row[1] - ocy, row[2]
        a = np.deg2rad(a_deg)
        cos_a, sin_a = np.cos(a), np.sin(a)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = _BASE_VERTS @ rot.T + np.array([cx, cy])
        dots = verts @ normals.T  # (6,6)
        all_dots.append(dots.ravel())

    flat = np.concatenate(all_dots)
    m = flat.max()
    log_sum_exp = m + np.log(np.sum(np.exp(beta * (flat - m)))) / beta
    s = log_sum_exp / (np.sqrt(3.0) / 2.0)
    return s


def _separation_penalty(positions_angles, threshold=3.0):
    """
    Compute a smooth penalty for overlapping inner hexagons using SAT.
    positions_angles: (12,3) array.
    Returns 0 if all separated, else sum of squared violations.
    """
    n = 12
    # Precompute vertices for all hexagons: (n, 6, 2)
    verts = np.empty((n, 6, 2))
    for i in range(n):
        cx, cy, a_deg = positions_angles[i]
        a = np.deg2rad(a_deg)
        cos_a, sin_a = np.cos(a), np.sin(a)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts[i] = _BASE_VERTS @ rot.T + np.array([cx, cy])

    # Precompute normals for each hexagon: (n, 6, 2)
    normals = np.empty((n, 6, 2))
    for i in range(n):
        a = np.deg2rad(positions_angles[i, 2])
        cos_a, sin_a = np.cos(a), np.sin(a)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        normals[i] = _BASE_NORMALS @ rot.T

    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            # Collect all candidate normals: 6 from hex i, 6 from hex j
            cand_normals = np.vstack([normals[i], normals[j]])  # (12,2)
            v1 = verts[i]  # (6,2)
            v2 = verts[j]  # (6,2)

            # Project
            proj1 = v1 @ cand_normals.T  # (6,12)
            proj2 = v2 @ cand_normals.T  # (6,12)
            min1 = proj1.min(axis=0)  # (12,)
            max1 = proj1.max(axis=0)  # (12,)
            min2 = proj2.min(axis=0)  # (12,)
            max2 = proj2.max(axis=0)  # (12,)

            # gap: if min2 > max1 + eps OR min1 > max2 + eps, separated
            gap1 = min2 - max1  # (12,)
            gap2 = min1 - max2  # (12,)

            # If any gap > 1e-6, separated
            if np.max(gap1) > 1e-6 or np.max(gap2) > 1e-6:
                continue
            else:
                # Overlapping: penalty based on center distance
                d = np.linalg.norm(positions_angles[i, :2] - positions_angles[j, :2])
                if d < threshold:
                    penalty += (threshold - d) ** 2 * 100.0
    return penalty


def _check_validity(inner, outer, s):
    """Validate packing; return (is_valid, error_msg)."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False, "non-finite values"
    if s < np.sqrt(12.0) - 1e-12:
        return False, "s < sqrt(12)"

    ov = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(ov[i] - ov[j]) < 1e-10:
                return False, "outer vertices not distinct"

    # Check unit edge length for inner hexagons
    for row in inner:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        for i in range(6):
            edge_len = np.linalg.norm(hv[(i + 1) % 6] - hv[i])
            if abs(edge_len - 1.0) > 1e-8:
                return False, f"unit edge length differs: {edge_len}"

    # Check inner-inner separation using SAT
    def _separated(v1, v2):
        edges = []
        for i in range(6):
            edges.append(v1[(i + 1) % 6] - v1[i])
            edges.append(v2[(i + 1) % 6] - v2[i])
        for edge in edges:
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            gap1 = min2 - max1
            gap2 = min1 - max2
            if gap1 > 1e-6 or gap2 > 1e-6:
                return True
        return False

    inner_verts = [_hex_vertices(row[0], row[1], 1.0, row[2]) for row in inner]
    for i in range(12):
        for j in range(i + 1, 12):
            if not _separated(inner_verts[i], inner_verts[j]):
                return False, f"inner hexagons {i} and {j} overlap"

    # Check inner vertices inside outer hexagon
    for row in inner:
        cx, cy, a = row
        hv = _hex_vertices(cx, cy, 1.0, a)
        for v in hv:
            for k in range(6):
                p1 = ov[k]
                p2 = ov[(k + 1) % 6]
                edge = p2 - p1
                to_v = v - p1
                cross = edge[0] * to_v[1] - edge[1] * to_v[0]
                if cross < -1e-6:
                    return False, f"inner vertex outside outer hexagon"

    return True, ""


def _hex_lattice_positions(spacing_scale=1.0):
    """Generate a hexagonal lattice arrangement of 12 unit hexagons."""
    s = spacing_scale
    h = 3.0 * s
    v = 3.0 * s * np.sqrt(3.0) / 2.0  # 2.598...
    positions = np.array([
        [0.0, 0.0],
        [h, 0.0],
        [-h, 0.0],
        [h / 2, v],
        [-h / 2, v],
        [h / 2, -v],
        [-h / 2, -v],
        [3 * h / 2, v],
        [-3 * h / 2, v],
        [3 * h / 2, -v],
        [-3 * h / 2, -v],
        [0.0, 2 * v],
    ])
    return positions


def _circular_positions(radius=2.5):
    """Generate positions on a circle."""
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    positions = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    return positions


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 580.0

    best_s = float('inf')
    best_inner = None
    best_outer = None

    # Build initial configurations
    inits = []
    base_lattice = _hex_lattice_positions(1.0)
    inits.append(base_lattice)
    for scale in [0.95, 1.05]:
        inits.append(_hex_lattice_positions(scale))
    for r in [2.0, 2.5, 3.0]:
        inits.append(_circular_positions(r))
    np.random.seed(42)
    for _ in range(4):
        inits.append(base_lattice + np.random.randn(12, 2) * 0.5)
    for _ in range(3):
        inits.append(np.random.randn(12, 2) * 3.0)

    def _objective(params, use_smooth=True):
        """
        params: 38 values: 12*(x, y, angle_deg) + 2*(ocx, ocy)
        Returns: computed s + penalty for overlaps
        """
        inner_p = params[:36].reshape(12, 3)
        ocx = params[36]
        ocy = params[37]
        outer_center = np.array([ocx, ocy])

        if use_smooth:
            s_val = _min_containing_hex_side_smooth(inner_p, outer_center, 0.0, beta=50.0)
        else:
            s_val = _min_containing_hex_side(inner_p, outer_center, 0.0)

        penalty = _separation_penalty(inner_p)
        return s_val + penalty

    def _try_optimize(init_positions, init_angle_outer, maxiter, xatol, fatol):
        """Run Nelder-Mead from a given start; return (s, inner, outer) if valid."""
        nonlocal best_s, best_inner, best_outer
        init_params = np.zeros(38)
        init_params[:24] = init_positions.flatten()
        init_params[24::3] = 0.0  # angles default 0
        init_params[36] = 0.0
        init_params[37] = 0.0

        try:
            res = minimize(_objective, init_params, method='Nelder-Mead',
                           options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol,
                                    'adaptive': True})
        except Exception:
            return

        inner_opt = res.x[:36].reshape(12, 3)
        ocx_opt = res.x[36]
        ocy_opt = res.x[37]
        outer_center_opt = np.array([ocx_opt, ocy_opt])

        # Compute exact s (non-smooth)
        s_opt = _min_containing_hex_side(inner_opt, outer_center_opt, 0.0) + 0.001
        outer_opt = np.array([ocx_opt, ocy_opt, 0.0])

        valid, msg = _check_validity(inner_opt, outer_opt, s_opt)
        if valid and s_opt < best_s:
            best_s = s_opt
            best_inner = inner_opt.copy()
            best_outer = outer_opt.copy()
            print(f"  Found: s = {best_s:.6f}, 1/s = {1/best_s:.6f}")

    # Try each initial configuration
    for i, init_pos in enumerate(inits):
        if time.time() > deadline - 60:
            break
        print(f"Start {i}/{len(inits)}...")
        _try_optimize(init_pos, 0.0, maxiter=3000, xatol=1e-7, fatol=1e-7)

    # Fallback if no valid solution found
    if best_inner is None:
        init_pos = base_lattice
        inner_fb = np.zeros((12, 3))
        inner_fb[:, :2] = init_pos
        outer_fb = np.array([0.0, 0.0, 0.0])
        s_fb = _min_containing_hex_side(inner_fb, np.array([0.0, 0.0]), 0.0) + 0.01
        valid, msg = _check_validity(inner_fb, outer_fb, s_fb)
        if valid:
            best_inner = inner_fb
            best_outer = outer_fb
            best_s = s_fb
        else:
            # Last resort: large hexagon
            best_inner = inner_fb
            best_outer = outer_fb
            best_s = 20.0

    # Polish pass with tighter tolerances
    if best_inner is not None and time.time() < deadline - 30:
        print("Polish pass...")
        init_params = np.zeros(38)
        init_params[:36] = best_inner.flatten()
        init_params[36] = best_outer[0]
        init_params[37] = best_outer[1]

        # Use non-smooth objective for polish
        def _objective_exact(params):
            inner_p = params[:36].reshape(12, 3)
            ocx = params[36]
            ocy = params[37]
            outer_center = np.array([ocx, ocy])
            s_val = _min_containing_hex_side(inner_p, outer_center, 0.0)
            penalty = _separation_penalty(inner_p)
            return s_val + penalty

        try:
            res_polish = minimize(_objective_exact, init_params, method='Nelder-Mead',
                                  options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-9,
                                           'adaptive': True})
            inner_opt = res_polish.x[:36].reshape(12, 3)
            ocx_opt = res_polish.x[36]
            ocy_opt = res_polish.x[37]
            outer_center_opt = np.array([ocx_opt, ocy_opt])
            s_opt = _min_containing_hex_side(inner_opt, outer_center_opt, 0.0) + 0.0005
            outer_opt = np.array([ocx_opt, ocy_opt, 0.0])
            valid, msg = _check_validity(inner_opt, outer_opt, s_opt)
            if valid and s_opt < best_s:
                best_s = s_opt
                best_inner = inner_opt.copy()
                best_outer = outer_opt.copy()
                print(f"Polished: s = {best_s:.6f}, 1/s = {1/best_s:.6f}")
        except Exception as e:
            print(f"Polish error: {e}")

    # Final validation
    if best_inner is not None:
        valid, msg = _check_validity(best_inner, best_outer, best_s)
        if not valid:
            print(f"Final result invalid: {msg}, using fallback")
            s = 20.0
            return best_inner, best_outer, s

    return best_inner, best_outer, best_s


# EVOLVE-BLOCK-END
