# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 620.0  # safety margin before 630s kill
_t0 = time.time()


def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with side d and rotation angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _min_outer_side(inner_hexes, outer_center):
    """Compute minimum outer hexagon side length s for containment."""
    s_max = 0.0
    # Edge normals of a regular hexagon with angle 0 are at 30, 90, 150, 210, 270, 330 deg
    normals = np.array([[np.cos(np.radians(30 + 60*k)), np.sin(np.radians(30 + 60*k))] for k in range(6)])
    for i in range(len(inner_hexes)):
        vi = _hex_vertices(inner_hexes[i][0], inner_hexes[i][1], 1.0, inner_hexes[i][2])
        for v in vi:
            rel = v - outer_center
            for n in normals:
                proj = np.dot(rel, n)
                s_needed = proj / (np.sqrt(3) / 2)
                if s_needed > s_max:
                    s_max = s_needed
    return max(s_max, np.sqrt(12.0))


def _overlap_penalty(inner_hexes):
    """Compute penalty for overlapping inner hexagons using SAT."""
    penalty = 0.0
    n = len(inner_hexes)
    for i in range(n):
        vi = _hex_vertices(inner_hexes[i][0], inner_hexes[i][1], 1.0, inner_hexes[i][2])
        for j in range(i + 1, n):
            vj = _hex_vertices(inner_hexes[j][0], inner_hexes[j][1], 1.0, inner_hexes[j][2])
            separated = False
            for verts in (vi, vj):
                for k in range(6):
                    p1 = verts[k]
                    p2 = verts[(k + 1) % 6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm > 1e-15:
                        normal = normal / norm
                        proj_i = np.dot(vi, normal)
                        proj_j = np.dot(vj, normal)
                        if proj_i.max() < proj_j.min() - 1e-6 or proj_j.max() < proj_i.min() - 1e-6:
                            separated = True
                            break
                if separated:
                    break
            if not separated:
                # Compute overlap amount for penalty
                min_overlap = np.inf
                for verts in (vi, vj):
                    for k in range(6):
                        p1 = verts[k]
                        p2 = verts[(k + 1) % 6]
                        edge = p2 - p1
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm > 1e-15:
                            normal = normal / norm
                            proj_i = np.dot(vi, normal)
                            proj_j = np.dot(vj, normal)
                            overlap = proj_i.max() - proj_j.min()
                            if overlap > 0:
                                min_overlap = min(min_overlap, overlap)
                            overlap2 = proj_j.max() - proj_i.min()
                            if overlap2 > 0:
                                min_overlap = min(min_overlap, overlap2)
                if min_overlap < np.inf:
                    penalty += min_overlap ** 2
    return penalty


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    global _t0
    _t0 = time.time()
    
    # Initial configuration: compact hexagonal arrangement
    init_positions = [
        (0.0, 0.0),      # center
        (2.5, 0.0),      # right
        (1.25, 2.165),   # top-right
        (-1.25, 2.165),  # top-left
        (-2.5, 0.0),     # left
        (-1.25, -2.165), # bottom-left
        (1.25, -2.165),  # bottom-right
        (3.75, 2.165),   # far top-right
        (-3.75, 2.165),  # far top-left
        (-3.75, -2.165), # far bottom-left
        (3.75, -2.165),  # far bottom-right
        (0.0, -4.33),    # bottom center
    ]
    
    # Variables: x, y for each of 12 hexagons (24 vars), all angles fixed at 0
    x0 = np.array([p[0] for p in init_positions] + [p[1] for p in init_positions])
    
    def objective(x):
        if time.time() - _t0 > _DEADLINE:
            return 1e10
        positions = [(x[2*i], x[2*i+1]) for i in range(12)]
        inner_hexes = [(positions[i][0], positions[i][1], 0.0) for i in range(12)]
        cx = np.mean([p[0] for p in positions])
        cy = np.mean([p[1] for p in positions])
        s = _min_outer_side(inner_hexes, np.array([cx, cy]))
        penalty = _overlap_penalty(inner_hexes)
        return s + 10.0 * penalty
    
    best_x = x0.copy()
    best_obj = objective(x0)
    
    # Nelder-Mead optimization
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-12})
        if res.fun < best_obj:
            best_x = res.x
            best_obj = res.fun
    except Exception:
        pass
    
    # Local refinement
    if time.time() - _t0 < _DEADLINE - 30:
        try:
            res2 = minimize(objective, best_x, method='Nelder-Mead',
                            options={'maxiter': 15000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res2.fun < best_obj:
                best_x = res2.x
                best_obj = res2.fun
        except Exception:
            pass
    
    # Final polish with very small steps
    if time.time() - _t0 < _DEADLINE - 10:
        try:
            res3 = minimize(objective, best_x, method='Nelder-Mead',
                            options={'maxiter': 20000, 'xatol': 1e-14, 'fatol': 1e-16})
            if res3.fun < best_obj:
                best_x = res3.x
                best_obj = res3.fun
        except Exception:
            pass
    
    # Construct final result
    positions = [(best_x[2*i], best_x[2*i+1]) for i in range(12)]
    inner_hex_data = np.array([[p[0], p[1], 0.0] for p in positions])
    
    cx = np.mean([p[0] for p in positions])
    cy = np.mean([p[1] for p in positions])
    s = _min_outer_side([(p[0], p[1], 0.0) for p in positions], np.array([cx, cy]))
    s = max(s, np.sqrt(12.0))
    
    outer_hex_data = np.array([cx, cy, 0.0])
    
    return inner_hex_data, outer_hex_data, float(s)


# EVOLVE-BLOCK-END
