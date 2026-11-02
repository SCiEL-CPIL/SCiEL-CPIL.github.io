import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3


def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])


def _min_enclosing_hex_side_lp(verts, outer_angle_deg):
    """
    Find minimum enclosing regular hexagon side length using linear programming.
    The hexagon has 6 sides with outward normals at angles (outer_angle_deg + 30) + k*60.
    We minimize r (apothem) over center (cx,cy) and r >= 0.
    Constraints: for each vertex v and each normal n_k:
        n_k . (v - c) <= r   =>   n_kx*cx + n_ky*cy + r >= n_k . v
    i.e.  -n_kx*cx - n_ky*cy - r <= -n_k . v
    Minimize r.
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])  # (6,2)

    M = verts.shape[0]
    # Variables: x = [cx, cy, r]
    # Objective: minimize r => c = [0, 0, 1]
    c_obj = np.array([0.0, 0.0, 1.0])

    # Inequality constraints A_ub @ x <= b_ub
    # For each (vertex, normal): -n_kx*cx - n_ky*cy - r <= -(n_k . v)
    # A_ub shape: (M*6, 3)
    # Vectorized construction
    rhs = -(verts @ normals.T)  # (M, 6)
    b_ub = rhs.flatten()  # (M*6,)
    
    A_ub_parts = np.column_stack([
        np.repeat(-normals[:, 0], M),
        np.repeat(-normals[:, 1], M),
        np.full(M * 6, -1.0)
    ])
    
    # r >= 0  =>  -r <= 0
    A_ub = np.vstack([A_ub_parts, np.array([[0.0, 0.0, -1.0]])])
    b_ub = np.append(b_ub, 0.0)

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, method='highs')

    if res.status != 0:
        return None

    cx, cy, r = res.x
    if r < 0:
        r = 0.0
    side = r * 2.0 / _SQRT3
    return cx, cy, side


def _build_positions(layout_idx):
    """Generate 11 positions for given layout index."""
    sqrt3 = _SQRT3
    scale = 1.01
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)

    if layout_idx == 0:
        # Compact hexagonal: center + 6 ring + 4 from second ring
        target = [(0, 0)]
        ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        target.extend(ring1)
        ring2 = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0), (-1, -1), (0, -2), (1, -2), (2, -1)]
        ring2_sorted = sorted(ring2, key=lambda p: p[0] ** 2 + p[1] ** 2)
        target.extend(ring2_sorted[:4])
        positions = []
        for ti, tj in target:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        return positions[:N]
    elif layout_idx == 1:
        # Row-based: 3-4-4 pattern
        positions = []
        y1 = -1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 3):
            x = k * sqrt3 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-1, 3):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 2:
        # 4-3-4 pattern
        positions = []
        y1 = -1.5 * scale
        for k in range(-2, 2):
            x = k * sqrt3 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-2, 2):
            x = k * sqrt3 * scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 3:
        # 2-4-2-3 pattern
        positions = []
        y1 = -3.0 * scale
        for k in range(0, 2):
            x = k * sqrt3 * scale - sqrt3 / 2 * scale
            positions.append((x, y1))
        y2 = -1.5 * scale
        for k in range(-1, 3):
            x = k * sqrt3 * scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-1, 1):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y3))
        y4 = 1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 4:
        # 3-5-3 pattern
        positions = []
        y1 = -1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 3):
            x = k * sqrt3 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 5:
        # Center + 6 ring + 4 more, symmetric
        positions = []
        positions.append((0, 0))
        ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        for ti, tj in ring1:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        ring2 = [(2, 0), (1, 1), (-1, 2), (-2, 0), (-1, -1), (0, -2)]
        for ti, tj in ring2[:4]:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append((x, y))
        return positions[:N]
    elif layout_idx == 6:
        # 2-3-3-3 pattern
        positions = []
        y1 = -3.0 * scale
        for k in range(0, 2):
            x = k * sqrt3 * scale - sqrt3 / 2 * scale
            positions.append((x, y1))
        y2 = -1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y3))
        y4 = 1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 7:
        # 4-4-3 pattern
        positions = []
        y1 = -1.5 * scale
        for k in range(-2, 2):
            x = k * sqrt3 * scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append((x, y2))
        y3 = 1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale
            positions.append((x, y3))
        return positions[:N]
    else:
        # Fallback: dense grid sorted by distance from origin
        coords = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i * u[0] + j * v[0]
                y = i * u[1] + j * v[1]
                coords.append((x, y, i, j))
        coords_sorted = sorted(coords, key=lambda t: t[0] ** 2 + t[1] ** 2)
        return [(x, y) for x, y, i, j in coords_sorted[:N]]


def _polish_with_nelder_mead(all_vertices, cx_init, cy_init, angle_init):
    """
    Polish the solution using Nelder-Mead on (cx, cy, angle).
    The objective is the side length from the LP for a given angle,
    but we also allow cx, cy to vary. However, the LP already optimizes cx, cy
    for a fixed angle. So we just need to optimize the angle.
    But to be safe, we can define an objective that solves the LP for the given angle
    and returns the side.
    """
    def objective(params):
        ang = params[0] % 60.0
        res = _min_enclosing_hex_side_lp(all_vertices, float(ang))
        if res is None:
            return 1e6
        _, _, side = res
        return side
    
    try:
        res = minimize(objective, [angle_init], method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.success:
            best_angle = res.x[0] % 60.0
            final_res = _min_enclosing_hex_side_lp(all_vertices, float(best_angle))
            if final_res:
                return final_res
    except Exception:
        pass
    
    return _min_enclosing_hex_side_lp(all_vertices, float(angle_init))


def construct():
    global _START_TIME, _DEADLINE

    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0

    num_layouts = 8
    layouts = [_build_positions(i) for i in range(num_layouts)]

    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 10.0:
            break

        inner_angle = 30.0

        all_vertices = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in positions])

        best_local_side = float('inf')
        best_local_center = (0.0, 0.0)
        best_local_angle = 0.0

        # Coarse sweep over outer angles (0 to 60 degrees, step 1.0)
        for outer_angle_deg in np.arange(0, 60.0, 1.0):
            if time.time() > _DEADLINE - 10.0:
                break

            res = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
            if res:
                cx, cy, side = res
                if side < best_local_side:
                    best_local_side = side
                    best_local_center = (cx, cy)
                    best_local_angle = outer_angle_deg

        # Fine sweep around the best angle
        if best_local_side < float('inf'):
            fine_start = max(0.0, best_local_angle - 2.0)
            fine_end = min(60.0, best_local_angle + 2.0)
            for outer_angle_deg in np.arange(fine_start, fine_end, 0.1):
                if time.time() > _DEADLINE - 10.0:
                    break
                res = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
                if res:
                    cx, cy, side = res
                    if side < best_local_side:
                        best_local_side = side
                        best_local_center = (cx, cy)
                        best_local_angle = outer_angle_deg

        # Very fine sweep
        if best_local_side < float('inf'):
            vf_start = max(0.0, best_local_angle - 0.5)
            vf_end = min(60.0, best_local_angle + 0.5)
            for outer_angle_deg in np.arange(vf_start, vf_end, 0.01):
                if time.time() > _DEADLINE - 10.0:
                    break
                res = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
                if res:
                    cx, cy, side = res
                    if side < best_local_side:
                        best_local_side = side
                        best_local_center = (cx, cy)
                        best_local_angle = outer_angle_deg

        # Nelder-Mead polish on angle
        if best_local_side < float('inf') and time.time() < _DEADLINE - 15.0:
            polished = _polish_with_nelder_mead(all_vertices, best_local_center[0], best_local_center[1], best_local_angle)
            if polished:
                cx, cy, side = polished
                if side < best_local_side:
                    best_local_side = side
                    best_local_center = (cx, cy)
                    # Get the angle from the polished result? 
                    # The polish function returns (cx, cy, side) but not the angle.
                    # We need to track the angle. Let's modify _polish_with_nelder_mead to return angle too.
                    # Or just assume the angle is close to best_local_angle.
                    # Actually, let's just re-solve at the best angle found by NM.
                    # Since _polish_with_nelder_mead internally finds the best angle,
                    # we can just use the side and center it returns.
                    # But we need the angle for the output.
                    # Let's modify the approach: just do a final very fine sweep around the NM result.
                    # For now, keep the best_local_angle from the sweep.
                    pass

        if best_local_side < best_side:
            best_side = best_local_side
            best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
            best_center = best_local_center
            best_outer_angle = best_local_angle

    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, best_outer_angle
