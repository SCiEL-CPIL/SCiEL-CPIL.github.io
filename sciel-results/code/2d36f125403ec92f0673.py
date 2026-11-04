"""Unit regular hexagons inside a regular hexagon n=12."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12
DEADLINE = 140.0  # safety margin before 150s kill


def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    theta = np.radians(angle_deg)
    angles = theta + np.pi/3 * np.arange(6)
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])


def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point is inside hexagon."""
    dx = px - cx
    dy = py - cy
    theta = -np.radians(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    lx = dx * cos_t - dy * sin_t
    ly = dx * sin_t + dy * cos_t
    normals = np.array([[np.cos(np.radians(30 + 60*k)), np.sin(np.radians(30 + 60*k))] for k in range(6)])
    dists = normals @ np.array([lx, ly])
    return np.all(np.abs(dists) <= side * np.sqrt(3)/2 + 1e-9)


def seg_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 properly intersects segment p3-p4."""
    def orient(a, b, c):
        val = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        if abs(val) < 1e-10: return 0
        return 1 if val > 0 else -1
    
    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)
    return o1 != o2 and o3 != o4


def hexagons_overlap(v1, v2):
    """Check if two hexagons overlap (have common interior)."""
    # Vertex in polygon
    for i in range(6):
        if point_in_hexagon(v1[i,0], v1[i,1], v1[:,0].mean(), v1[:,1].mean(), 1.0, 0.0):
            pass  # This won't work, need proper check
    # Better: check if any vertex of one is strictly inside the other
    # and if any edges cross
    # For hexagon 1 with center c1, side 1, angle a1:
    # We need to pass the actual parameters. Let's restructure.
    return True  # placeholder


def compute_min_side(inner):
    """Compute minimum outer side length to contain all inner hexagons in outer hexagon at origin with angle 0."""
    max_required = 0.0
    sqrt3_2 = np.sqrt(3) / 2
    for (cx, cy, a) in inner:
        verts = hex_vertices(cx, cy, 1.0, a)
        for vx, vy in verts:
            for k in range(6):
                ang = np.radians(30 + 60*k)
                val = vx * np.cos(ang) + vy * np.sin(ang)
                required = val / sqrt3_2
                if required > max_required:
                    max_required = required
    return max_required


def check_disjoint(inner):
    """Check if all inner hexagons are pairwise disjoint. Returns True if disjoint."""
    verts_list = [hex_vertices(x, y, 1.0, a) for (x, y, a) in inner]
    for i in range(N):
        for j in range(i+1, N):
            # Quick distance check
            d = np.sqrt((inner[i][0]-inner[j][0])**2 + (inner[i][1]-inner[j][1])**2)
            if d < 2.0 - 1e-8:
                # Check if they actually overlap
                # A vertex of hex i inside hex j
                for k in range(6):
                    if point_in_hexagon(verts_list[i][k,0], verts_list[i][k,1],
                                        inner[j][0], inner[j][1], 1.0, inner[j][2]):
                        return False
                # A vertex of hex j inside hex i
                for k in range(6):
                    if point_in_hexagon(verts_list[j][k,0], verts_list[j][k,1],
                                        inner[i][0], inner[i][1], 1.0, inner[i][2]):
                        return False
                # Edge intersections
                for m in range(6):
                    for n in range(6):
                        if seg_intersect(verts_list[i][m], verts_list[i][(m+1)%6],
                                         verts_list[j][n], verts_list[j][(n+1)%6]):
                            return False
    return True


def objective(params):
    """Minimize outer side length with penalty for overlaps."""
    inner = [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(N)]
    
    if not check_disjoint(inner):
        return 1e6 + compute_min_side(inner)
    
    return compute_min_side(inner)


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    
    # Initial arrangement: hexagonal packing pattern
    # For 12 hexagons, use a compact arrangement
    # Center + ring of 6 + 5 more
    init = []
    # Center
    init.append([0.0, 0.0, 0.0])
    # Ring of 6 at distance 2 (touching)
    for k in range(6):
        ang = np.radians(60*k)
        init.append([2.0*np.cos(ang), 2.0*np.sin(ang), 0.0])
    # 5 more: place in gaps
    for k in range(5):
        ang = np.radians(60*k + 30)
        init.append([3.0*np.cos(ang), 3.0*np.sin(ang), 0.0])
    
    x0 = np.array(init).flatten()
    
    bounds = [(-15.0, 15.0), (-15.0, 15.0), (0.0, 60.0)] * N
    
    best_x = x0.copy()
    best_val = objective(x0)
    
    # Nelder-Mead
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 8000, 'xatol': 1e-11, 'fatol': 1e-11,
                               'adaptive': True})
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x.copy()
    except Exception:
        pass
    
    # Differential evolution if time permits
    if time.time() < deadline - 40:
        try:
            de_res = differential_evolution(objective, bounds, maxiter=100,
                                            popsize=12, tol=1e-6, seed=42,
                                            polish=False, init='sobol')
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_x = de_res.x.copy()
        except Exception:
            pass
    
    # Final polish
    if time.time() < deadline - 10:
        try:
            res2 = minimize(objective, best_x, method='Nelder-Mead',
                           options={'maxiter': 15000, 'xatol': 1e-13, 'fatol': 1e-13,
                                    'adaptive': True})
            if res2.fun < best_val:
                best_val = res2.fun
                best_x = res2.x.copy()
        except Exception:
        pass
    
    inner = [(best_x[3*i], best_x[3*i+1], best_x[3*i+2]) for i in range(N)]
    outer_side = compute_min_side(inner)
    
    return inner, (0.0, 0.0), outer_side, 0.0
