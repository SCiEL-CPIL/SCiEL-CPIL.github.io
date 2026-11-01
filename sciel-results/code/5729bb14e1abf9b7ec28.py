import numpy as np
import time

N = 11

def _hex_vertices(cx, cy, r, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    return np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])

def _point_in_hex(px, py, cx, cy, r, angle_deg):
    a = np.radians(angle_deg)
    dx, dy = px - cx, py - cy
    ca, sa = np.cos(-a), np.sin(-a)
    rx = ca * dx - sa * dy
    ry = sa * dx + ca * dy
    apothem = r * np.sqrt(3) / 2
    for k in range(6):
        theta = np.radians(30 + 60 * k)
        nx, ny = np.cos(theta), np.sin(theta)
        if rx * nx + ry * ny > apothem + 1e-9:
            return False
    return True

def _min_outer_circumradius(inner_pts, ocx, ocy, oangle):
    """inner_pts: list of (x,y) vertices of all inner hexagons. Find min circumradius of outer hex."""
    lo, hi = 0.0, 50.0
    for _ in range(50):
        mid = (lo + hi) / 2
        ok = all(_point_in_hex(vx, vy, ocx, ocy, mid, oangle) for vx, vy in inner_pts)
        if ok:
            hi = mid
        else:
            lo = mid
    return hi

def _axial_to_pixel(q, r):
    x = 2.0 * (q + r / 2.0)
    y = np.sqrt(3) * r
    return (x, y)

def _hex_cluster(n):
    """Return list of (x,y) for n hexagons in compact hexagonal cluster."""
    if n == 1:
        return [(0.0, 0.0)]
    
    # Generate positions in rings, sorted by distance
    positions_axial = [(0, 0)]
    # Ring 1
    ring1 = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]
    positions_axial.extend(ring1)
    # Ring 2
    ring2 = [(2,0),(2,-1),(2,-2),(1,-2),(0,-2),(-1,-1),(-2,0),(-2,1),(-2,2),(-1,2),(0,2),(1,1)]
    positions_axial.extend(ring2)
    # Ring 3 if needed
    ring3 = [(3,0),(3,-1),(3,-2),(3,-3),(2,-3),(1,-3),(0,-3),(-1,-2),(-2,-1),(-3,0),(-3,1),(-3,2),(-3,3),(-2,3),(-1,3),(0,3),(1,2),(2,1)]
    positions_axial.extend(ring3)
    
    # Sort by distance from center
    def d2(ax):
        x, y = _axial_to_pixel(*ax)
        return x*x + y*y
    positions_axial.sort(key=d2)
    
    return [_axial_to_pixel(q, r) for q, r in positions_axial[:n]]

def construct():
    t0 = time.time()
    deadline = t0 + 130.0
    
    # Hexagonal cluster of 11
    centers = _hex_cluster(N)
    cx = np.mean([p[0] for p in centers])
    cy = np.mean([p[1] for p in centers])
    
    # All vertices of all inner hexagons (unit hexagons, angle=0)
    all_verts = []
    for (px, py) in centers:
        verts = _hex_vertices(px, py, 1.0, 0.0)
        all_verts.extend([(float(vx), float(vy)) for vx, vy in verts])
    
    best_side = 100.0
    best_oangle = 0.0
    best_oc = (cx, cy)
    
    # Search over outer hexagon rotation and center offset
    for oangle in [0, 15, 30, 45, 60]:
        for dx in [-1.0, -0.5, 0, 0.5, 1.0]:
            for dy in [-1.0, -0.5, 0, 0.5, 1.0]:
                ocx, ocy = cx + dx, cy + dy
                side = _min_outer_circumradius(all_verts, ocx, ocy, float(oangle))
                if side < best_side:
                    best_side = side
                    best_oangle = float(oangle)
                    best_oc = (ocx, ocy)
                
                if time.time() > deadline - 10:
                    break
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
    
    # Fine-tune around best
    if time.time() < deadline - 5:
        for da in np.arange(-3, 3, 0.5):
            for ddx in np.arange(-0.5, 0.5, 0.25):
                for ddy in np.arange(-0.5, 0.5, 0.25):
                    oa = best_oangle + da
                    ocx = best_oc[0] + ddx
                    ocy = best_oc[1] + ddy
                    side = _min_outer_circumradius(all_verts, ocx, ocy, float(oa))
                    if side < best_side:
                        best_side = side
                        best_oangle = float(oa)
                        best_oc = (ocx, ocy)
                    
                    if time.time() > deadline - 5:
                        break
                if time.time() > deadline - 5:
                    break
            if time.time() > deadline - 5:
                break
    
    inner = [[float(px), float(py), 0.0] for px, py in centers]
    return inner, best_oc, best_side, best_oangle

# EVOLVE-BLOCK-START
# construct is defined above
# EVOLVE-BLOCK-END
