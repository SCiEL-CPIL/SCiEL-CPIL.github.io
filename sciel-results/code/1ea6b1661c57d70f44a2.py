"""Hexagon packing via trust-region reflective optimization with penalty constraints."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(angle_deg, cx, cy):
    """Vertices of a regular hexagon with side 1, centered at (cx,cy), rotated by angle_deg."""
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s], [s, c]])
    base = np.array([[np.cos(np.radians(60 * k)), np.sin(np.radians(60 * k))] for k in range(6)])
    return R @ base.T + np.array([cx, cy])[:, None]

def _min_dist_poly(P, Q):
    """Minimum distance between two convex polygons (sets of vertices)."""
    if len(P) == 0 or len(Q) == 0:
        return np.inf
    # Check if any vertex of P is inside Q or vice versa -> 0
    def point_in_poly(pt, poly):
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > pt[1]) != (yj > pt[1])) and (pt[0] < (xj - xi) * (pt[1] - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
    for p in P:
        if point_in_poly(p, Q):
            return 0.0
    for q in Q:
        if point_in_poly(q, P):
            return 0.0
    best = np.inf
    for i in range(len(P)):
        p1 = P[i]
        p2 = P[(i + 1) % len(P)]
        for j in range(len(Q)):
            q1 = Q[j]
            q2 = Q[(j + 1) % len(Q)]
            # segment-segment distance
            d = _seg_seg_dist(p1, p2, q1, q2)
            if d < best:
                best = d
    return best

def _seg_seg_dist(p1, p2, q1, q2):
    """Distance between two segments."""
    def point_seg_dist(p, a, b):
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-15)
        t = max(0.0, min(1.0, t))
        proj = a + t * ab
        return np.linalg.norm(p - proj)
    d1 = point_seg_dist(p1, q1, q2)
    d2 = point_seg_dist(p2, q1, q2)
    d3 = point_seg_dist(q1, p1, p2)
    d4 = point_seg_dist(q2, p1, p2)
    return min(d1, d2, d3, d4)

def construct():
    n = N
    # Initial guess: grid-like placement
    rng = np.random.default_rng(42)
    # Place hexagons in a rough hexagonal grid
    positions = []
    for row in range(-2, 3):
        for col in range(-2, 3):
            x = 2.5 * col + 1.25 * row
            y = 2.165 * row
            positions.append([x, y, 0.0])
    # Trim to n hexagons
    positions = positions[:n]
    
    # Parameters: for each hex: (x, y, angle_rad)
    x0 = []
    for (x, y, a) in positions:
        x0.extend([x, y, np.radians(a)])
    x0 = np.array(x0, dtype=float)
    
    # Precompute normals for axis-aligned outer hexagon
    # Normals at 30, 90, 150, 210, 270, 330 degrees
    angles_n = np.radians(30 + np.arange(6) * 60)
    normals = np.column_stack([np.cos(angles_n), np.sin(angles_n)])
    
    def objective(params):
        # params: [x1,y1,a1, x2,y2,a2, ...]
        verts_list = []
        for i in range(n):
            xi = params[3*i]
            yi = params[3*i+1]
            ai = params[3*i+2]
            verts = _hex_vertices(np.degrees(ai), xi, yi)
            verts_list.append(verts)
        
        # 1. Calculate Outer Hexagon Side Length (Axis-Aligned)
        # r = max( v . n_k ) for all v, k
        max_proj = 0.0
        for verts in verts_list:
            # verts shape (6, 2)
            dots = verts @ normals.T
            m = np.max(dots)
            if m > max_proj:
                max_proj = m
                
        r = max_proj
        if r < 0:
            r = 0.0
        outer_side = 2.0 * r / np.sqrt(3.0)
        
        # 2. Calculate Overlap Penalty
        min_pair = np.inf
        for i in range(n):
            for j in range(i+1, n):
                ci = np.array([params[3*i], params[3*i+1]])
                cj = np.array([params[3*j], params[3*j+1]])
                if np.linalg.norm(ci - cj) > 2.0:
                    continue
                
                d = _min_dist_poly(verts_list[i], verts_list[j])
                if d < min_pair:
                    min_pair = d
        
        penalty = 0.0
        if min_pair < 0:
            penalty += 1000.0 * abs(min_pair)
        
        return outer_side + penalty
    
    # Bounds
    bounds = [(-10, 10)] * (3 * n)
    for i in range(n):
        bounds[3*i+2] = (-np.pi, np.pi)
    
    # Try multiple initializations
    best_result = None
    best_obj = np.inf
    
    for trial in range(5):
        if trial == 0:
            start = x0.copy()
        else:
            start = x0 + rng.normal(0, 0.5, size=3*n)
        
        try:
            res = minimize(objective, start, method='trust-constr',
                          bounds=bounds, options={'maxiter': 300, 'gtol': 1e-4, 'xtol': 1e-4})
            if res.fun < best_obj:
                best_obj = res.fun
                best_result = res.x
        except Exception:
            pass
    
    if best_result is None:
        best_result = x0
    
    # Extract final positions
    inner = []
    for i in range(n):
        x = float(best_result[3*i])
        y = float(best_result[3*i+1])
        a_deg = float(np.degrees(best_result[3*i+2]))
        inner.append([x, y, a_deg])
    
    # Compute outer hexagon side
    verts_list = []
    for (x, y, a_deg) in inner:
        verts = _hex_vertices(a_deg, x, y)
        verts_list.append(verts)
        
    max_proj = 0.0
    for verts in verts_list:
        dots = verts @ normals.T
        m = np.max(dots)
        if m > max_proj:
            max_proj = m
            
    outer_side = 2.0 * max_proj / np.sqrt(3.0)
    
    return inner, (0.0, 0.0), float(outer_side), 0.0
# EVOLVE-BLOCK-END
