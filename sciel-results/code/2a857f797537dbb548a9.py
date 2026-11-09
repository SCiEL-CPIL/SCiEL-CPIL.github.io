"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon (flat-top) with center (cx,cy), side, rotation."""
    theta = np.radians(angle_deg)
    angles = theta + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _check_containment(inner_list, ox, oy, s, oangle_deg):
    """Check all inner vertices inside outer hexagon. Return max violation (0 if valid)."""
    all_v = []
    for px, py, pangle in inner_list:
        v = _hex_vertices(px, py, 1.0, pangle)
        all_v.append(v)
    all_v = np.vstack(all_v)
    
    theta = np.radians(oangle_deg)
    dx = all_v[:, 0] - ox
    dy = all_v[:, 1] - oy
    c = np.cos(-theta)
    s_ = np.sin(-theta)
    lx = c * dx - s_ * dy
    ly = s_ * dx + c * dy
    
    edge_angles = np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
    dists = np.column_stack([lx, ly]) @ normals.T  # (72, 6)
    max_dist = s * np.sqrt(3) / 2
    return np.max(dists) - max_dist

def _check_overlap(inner_list):
    """Check no two inner hexagons overlap. Return max penetration (0 if valid)."""
    # For two convex hexagons, check if any vertex of one is inside the other
    n = len(inner_list)
    max_pen = 0.0
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, ai = inner_list[i]
            xj, yj, aj = inner_list[j]
            # Check vertices of i inside j, and vertices of j inside i
            vi = _hex_vertices(xi, yi, 1.0, ai)
            vj = _hex_vertices(xj, yj, 1.0, aj)
            for v in vi:
                # Check if v is inside hexagon j
                dx = v[0] - xj
                dy = v[1] - yj
                theta = np.radians(aj)
                c = np.cos(-theta)
                s_ = np.sin(-theta)
                lx = c * dx - s_ * dy
                ly = s_ * dx + c * dy
                edge_angles = np.radians(30 + 60 * np.arange(6))
                normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
                dists = normals @ np.array([lx, ly])
                max_dist = 1.0 * np.sqrt(3) / 2
                pen = np.max(dists) - max_dist
                if pen > max_pen:
                    max_pen = pen
            for v in vj:
                dx = v[0] - xi
                dy = v[1] - yi
                theta = np.radians(ai)
                c = np.cos(-theta)
                s_ = np.sin(-theta)
                lx = c * dx - s_ * dy
                ly = s_ * dx + c * dy
                edge_angles = np.radians(30 + 60 * np.arange(6))
                normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
                dists = normals @ np.array([lx, ly])
                max_dist = 1.0 * np.sqrt(3) / 2
                pen = np.max(dists) - max_dist
                if pen > max_pen:
                    max_pen = pen
    return max_pen

def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin
    
    sqrt3 = np.sqrt(3)
    
    # Hexagonal lattice: 4 columns x 3 rows
    inner = []
    for c in range(4):
        for r in range(3):
            x = c * 1.5
            y = r * sqrt3 + (sqrt3/2 if c % 2 == 1 else 0)
            inner.append([x, y, 0.0])
    
    # Center the configuration
    xs = [p[0] for p in inner]
    ys = [p[1] for p in inner]
    cx0 = (min(xs) + max(xs)) / 2
    cy0 = (min(ys) + max(ys)) / 2
    for i in range(len(inner)):
        inner[i][0] -= cx0
        inner[i][1] -= cy0
    
    # Initial outer hexagon estimate
    ox, oy, s, oangle = 0.0, 0.0, 4.0, 0.0
    
    # Compute initial s needed
    viol = _check_containment(inner, ox, oy, s, oangle)
    if viol > 0:
        s += viol / (sqrt3 / 2) + 0.05
    
    def cost(params):
        if time.time() > deadline:
            return 1e10
        ox_p, oy_p, s_p, oangle_p = params
        if s_p < 1.0:
            return 1e10
        viol = _check_containment(inner, ox_p, oy_p, s_p, oangle_p)
        return s_p + 1000 * max(viol, 0)
    
    x0 = np.array([ox, oy, s, oangle])
    try:
        result = minimize(cost, x0, method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
        if time.time() > deadline:
            return inner, (ox, oy), s, oangle
        ox, oy, s, oangle = result.x
        s = max(s, 1.0)
    except Exception:
        pass
    
    # Final verification and correction
    viol = _check_containment(inner, ox, oy, s, oangle)
    if viol > 1e-6:
        s += viol / (sqrt3 / 2) + 0.01
    
    return inner, (float(ox), float(oy)), float(s), float(oangle)
# EVOLVE-BLOCK-END
