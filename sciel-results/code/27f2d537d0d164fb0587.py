import math
import time
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

N = 12

# ---- Geometry helpers -------------------------------------------------------
def _hex_vertices(cx, cy, s, ang_deg):
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for k in range(6):
        t = math.radians(60.0 * k + a)
        px, py = s * math.cos(t), s * math.sin(t)
        rx = cx + ca * px - sa * py
        ry = cy + sa * px + ca * py
        out.append((rx, ry))
    return out

def _build_cluster():
    s3 = math.sqrt(3.0)
    neighbors = [
        (s3, 0.0),
        (s3/2.0, 1.5),
        (-s3/2.0, 1.5),
        (-s3, 0.0),
        (-s3/2.0, -1.5),
        (s3/2.0, -1.5),
    ]
    cells = [(0.0, 0.0)]
    for (dx, dy) in neighbors:
        cells.append((dx, dy))
    
    ring2 = []
    for a in neighbors:
        for b in neighbors:
            px = a[0] + b[0]
            py = a[1] + b[1]
            if any(abs(px - q) < 1e-9 and abs(py - r) < 1e-9 for (q, r) in ring2):
                continue
            if any(abs(px - q) < 1e-9 and abs(py - r) < 1e-9 for (q, r) in cells):
                continue
            ring2.append((px, py))
            
    ring2_sorted = sorted(ring2, key=lambda p: math.atan2(p[1], p[0]))
    best_window = None
    best_spread = float('inf')
    m = len(ring2_sorted)
    # We need 5 more cells to reach 12 (1+6+5=12)
    for i in range(m):
        window = [ring2_sorted[(i + j) % m] for j in range(5)]
        mx = 0.0
        for j in range(5):
            for k in range(j + 1, 5):
                d = math.hypot(window[j][0] - window[k][0],
                                window[j][1] - window[k][1])
                if d > mx:
                    mx = d
        if mx < best_spread:
            best_spread = mx
            best_window = window
            
    cells.extend(best_window)
    return [[x, y, 0.0] for (x, y) in cells]

CLUSTER = _build_cluster()
assert len(CLUSTER) == 12

# Pre-compute all vertices and convex hull
_ALL_VERTS = []
for (ix, iy, ia) in CLUSTER:
    _ALL_VERTS.extend(_hex_vertices(ix, iy, 1.0, ia))

_ALL_VERTS = np.array(_ALL_VERTS)
_HULL = ConvexHull(_ALL_VERTS)
_HULL_VERTS = _ALL_VERTS[_HULL.vertices]

def construct():
    t0 = time.time()
    deadline = 150.0 - 5.0  # Safety margin
    
    best_s = float('inf')
    best_cx = 0.0
    best_cy = 0.0
    best_ang = 0.0
    
    # Scan rotation angles 0 to 60 degrees
    # Step size 1 degree initially, can refine if time permits
    angles = np.arange(0, 60, 1.0)
    
    for ang_deg in angles:
        if time.time() > deadline:
            break
            
        a = math.radians(ang_deg)
        # Normals for the outer hexagon
        # Angles of normals: ang_deg + k*60
        normal_angles = np.deg2rad(ang_deg + 60.0 * np.arange(6))
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        
        # Constraints: -n_j . c - A <= -n_j . p_i
        # Variables: cx, cy, A
        # A is the apothem (distance from center to side)
        # s = 2 * A / sqrt(3)
        
        A_ub = []
        b_ub = []
        
        # For each hull vertex and each normal
        for p in _HULL_VERTS:
            for n in normals:
                # -n_x * cx - n_y * cy - 1 * A <= - (n_x * p_x + n_y * p_y)
                A_ub.append([-n[0], -n[1], -1.0])
                b_ub.append(-(n[0] * p[0] + n[1] * p[1]))
                
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Objective: Minimize A (index 2)
        c_obj = [0.0, 0.0, 1.0]
        
        # Bounds: A >= 0. cx, cy unbounded (or large)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                A_opt = res.x[2]
                s_opt = 2.0 * A_opt / math.sqrt(3.0)
                if s_opt < best_s:
                    best_s = s_opt
                    best_cx = res.x[0]
                    best_cy = res.x[1]
                    best_ang = ang_deg
        except Exception:
            pass

    if best_s == float('inf'):
        # Fallback: large hexagon
        # Calculate bounding box of cluster
        min_x = np.min(_ALL_VERTS[:, 0])
        max_x = np.max(_ALL_VERTS[:, 0])
        min_y = np.min(_ALL_VERTS[:, 1])
        max_y = np.max(_ALL_VERTS[:, 1])
        cx0 = (min_x + max_x) / 2.0
        cy0 = (min_y + max_y) / 2.0
        # Rough estimate of required side
        width = max_x - min_x
        height = max_y - min_y
        s_est = max(width, height) / math.sqrt(3.0) + 2.0
        return CLUSTER, (cx0, cy0), float(s_est), 0.0

    return CLUSTER, (best_cx, best_cy), float(best_s), float(best_ang)
