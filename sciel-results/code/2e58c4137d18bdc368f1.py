# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 600.0

def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    return np.array([
        [cx + s * np.cos(a + np.deg2rad(60 * i)),
         cy + s * np.sin(a + np.deg2rad(60 * i))]
        for i in range(6)
    ])

def _min_separation(v1, v2):
    """SAT check: return minimum gap between two convex hexagons."""
    n1 = len(v1)
    n2 = len(v2)
    min_gap = float('inf')
    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i + 1) % n1]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal /= norm
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        gap = max(proj1.min() - proj2.max(), proj2.min() - proj1.max())
        min_gap = min(min_gap, gap)
    for i in range(n2):
        p1 = v2[i]
        p2 = v2[(i + 1) % n2]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal /= norm
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        gap = max(proj1.min() - proj2.max(), proj2.min() - proj1.max())
        min_gap = min(min_gap, gap)
    return min_gap

def _check_all_separated(inner_hex_data):
    """Check all pairs are separated by > 1e-6."""
    for i in range(11):
        for j in range(i + 1, 11):
            v1 = _hex_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, inner_hex_data[i, 2])
            v2 = _hex_vertices(inner_hex_data[j, 0], inner_hex_data[j, 1], 1.0, inner_hex_data[j, 2])
            gap = _min_separation(v1, v2)
            if gap < 1e-6:
                return False
    return True

def _compute_outer_side(inner_hex_data, outer_center, outer_angle):
    """
    Minimum side length of outer hexagon (center, angle) containing all inner hexagons.
    """
    cx, cy = outer_center
    a = np.deg2rad(outer_angle)
    
    all_points = []
    for row in inner_hex_data:
        verts = _hex_vertices(row[0], row[1], 1.0, row[2])
        all_points.append(verts)
    all_points = np.vstack(all_points)
    all_points_shifted = all_points - np.array([cx, cy])
    
    max_s = 0.0
    for i in range(6):
        normal_angle = a + np.deg2rad(60 * i + 30)
        normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
        projections = all_points_shifted @ normal
        max_proj = projections.max()
        s_i = max_proj * 2.0 / np.sqrt(3.0)
        if s_i > max_s:
            max_s = s_i
    return max(0.0, max_s)

def _make_initial():
    """Hexagonal cluster: center + 6 ring + 4 of second ring."""
    sqrt3 = np.sqrt(3.0)
    pos = [[0.0, 0.0]]
    for i in range(6):
        ang = np.deg2rad(30 + 60 * i)
        pos.append([sqrt3 * np.cos(ang), sqrt3 * np.sin(ang)])
    for i in range(4):
        ang = np.deg2rad(60 * i)
        pos.append([2 * sqrt3 * np.cos(ang), 2 * sqrt3 * np.sin(ang)])
    inner = np.zeros((11, 3))
    for i, (x, y) in enumerate(pos):
        inner[i] = [x, y, 0.0]
    return inner

def _objective(x, outer_center, outer_angle):
    inner = x.reshape(11, 3)
    side = _compute_outer_side(inner, outer_center, outer_angle)
    # Overlap penalty
    for i in range(11):
        for j in range(i + 1, 11):
            v1 = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
            v2 = _hex_vertices(inner[j, 0], inner[j, 1], 1.0, inner[j, 2])
            gap = _min_separation(v1, v2)
            if gap < 1e-6:
                side += (1e-6 - gap) * 500.0
    return side

def hexagon_packing_11():
    best_side = float('inf')
    best_inner = None
    
    inner_init = _make_initial()
    
    # Check initial validity
    if _check_all_separated(inner_init):
        s0 = _compute_outer_side(inner_init, np.array([0.0, 0.0]), 0.0)
        best_side = s0
        best_inner = inner_init.copy()
    
    # Optimize with different outer angles
    for oa in [0.0, 30.0, 60.0, 15.0, 45.0]:
        if time.time() - _START_TIME > _DEADLINE - 30:
            break
        outer_c = np.array([0.0, 0.0])
        x0 = inner_init.flatten()
        try:
            res = minimize(_objective, x0, args=(outer_c, oa),
                           method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 5000})
            if res.fun < best_side:
                inner_opt = res.x.reshape(11, 3)
                if _check_all_separated(inner_opt):
                    s_opt = _compute_outer_side(inner_opt, outer_c, oa)
                    if s_opt < best_side:
                        best_side = s_opt
                        best_inner = inner_opt.copy()
        except Exception:
            pass
    
    if best_inner is None:
        # Fallback to seed
        inner_hex_data = np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0],
            [-1.25, -2.17, 0], [1.25, -2.17, 0],
            [-3.75, 2.17, 0], [3.75, 2.17, 0],
            [-3.75, -2.17, 0], [3.75, -2.17, 0],
        ])
        return inner_hex_data, np.array([0.0, 0.0, 0.0]), 8.0
    
    final_side = _compute_outer_side(best_inner, np.array([0.0, 0.0]), 0.0)
    if final_side < np.sqrt(11):
        final_side = np.sqrt(11)
    
    return best_inner, np.array([0.0, 0.0, 0.0]), float(final_side)

# EVOLVE-BLOCK-END
