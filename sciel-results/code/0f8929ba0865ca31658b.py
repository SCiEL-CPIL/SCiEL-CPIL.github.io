import numpy as np
import time
from scipy.optimize import minimize, SLSQP

N = 11
DEADLINE = time.time() + 120.0

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    a = np.radians(angle_deg)
    # Unit hexagon vertices at angles 0, 60, 120, 180, 240, 300
    base = np.array([[np.cos(np.radians(60*i)), np.sin(np.radians(60*i))] for i in range(6)])
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return cx + s * R.T @ base.T, R

def _seg_dist(p, a, b):
    ab = b - a
    t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-30)
    t = max(0.0, min(1.0, t))
    return np.linalg.norm(p - (a + t * ab))

def _poly_dist(v1, v2):
    # v1, v2 are (6,2) arrays of vertices for two convex hexagons
    # Check if any edge of v1 intersects any edge of v2, or if one vertex is inside the other
    # For simplicity, compute min distance between all edge pairs and vertex-vertex
    d = 1e18
    for i in range(6):
        a1 = v1[i]; b1 = v1[(i+1)%6]
        for j in range(6):
            a2 = v2[j]; b2 = v2[(j+1)%6]
            # vertex to segment distances
            d = min(d, _seg_dist(a1, a2, b2), _seg_dist(b1, a2, b2), _seg_dist(a2, a1, b1), _seg_dist(b2, a1, b1))
    return d

def _point_in_hex(px, py, verts):
    # verts is (6,2)
    # Check winding / cross products
    inside = True
    for i in range(6):
        a = verts[i]; b = verts[(i+1)%6]
        cross = (b[0]-a[0])*(py-a[1]) - (b[1]-a[1])*(px-a[0])
        if cross < -1e-12:
            inside = False
            break
    return inside

def _min_dist_between_hex(cx1, cy1, ang1, cx2, cy2, ang2):
    v1, _ = _hex_vertices(cx1, cy1, ang1)
    v2, _ = _hex_vertices(cx2, cy2, ang2)
    d = _poly_dist(v1, v2)
    # Check if any vertex of v1 is inside v2 or vice versa
    for v in v1:
        if _point_in_hex(v[0], v[1], v2):
            return 0.0
    for v in v2:
        if _point_in_hex(v[0], v[1], v1):
            return 0.0
    return d

def _hex_dist_to_outer(cx, cy, ang, ocx, ocy, side, oang):
    """Minimum distance from inner hex to outer hex boundary (must be >= 0 for containment).
       We want the inner hex to be inside the outer hex.
       Return negative of max penetration (or min distance to boundary)."""
    v_inner, _ = _hex_vertices(cx, cy, ang)
    v_outer, _ = _hex_vertices(ocx, ocy, oang, s=side)
    
    # Check if all vertices of inner hex are inside outer hex
    min_penetration = 1e18
    for v in v_inner:
        px, py = v
        if not _point_in_hex(px, py, v_outer):
            # Compute distance to outer boundary
            dist_to_boundary = 1e18
            for i in range(6):
                a = v_outer[i]; b = v_outer[(i+1)%6]
                dist_to_boundary = min(dist_to_boundary, _seg_dist(np.array([px,py]), a, b))
            # It's outside, so penetration is negative distance
            min_penetration = min(min_penetration, -dist_to_boundary)
        else:
            # It's inside; compute min distance to boundary for this vertex
            dist_to_boundary = 1e18
            for i in range(6):
                a = v_outer[i]; b = v_outer[(i+1)%6]
                dist_to_boundary = min(dist_to_boundary, _seg_dist(np.array([px,py]), a, b))
            min_penetration = min(min_penetration, dist_to_boundary)
    return min_penetration

def _objective(params):
    """params: x0,y0,a0, x1,y1,a1, ..., x_{N-1},y_{N-1},a_{N-1}, ocx, ocy, side
       a_i in degrees, oang fixed at 0 for simplicity (we'll try multiple)."""
    n = N
    inner_params = params[:3*n]
    ocx, ocy, side = params[3*n], params[3*n+1], params[3*n+2]
    
    # Extract inner hexagons
    inner = []
    for i in range(n):
        x = inner_params[3*i]
        y = inner_params[3*i+1]
        a = inner_params[3*i+2]
        inner.append((x, y, a))
    
    # Compute pairwise non-overlap: min distance must be >= 0
    penalty = 0.0
    for i in range(n):
        x1, y1, a1 = inner[i]
        for j in range(i+1, n):
            x2, y2, a2 = inner[j]
            d = _min_dist_between_hex(x1, y1, a1, x2, y2, a2)
            if d < 0:
                penalty += d * d
            # Also, the distance should be >= 0 (no overlap). We want d >= 0.
            # Penalty for overlap (d < 0)
    
    # Containment: each inner hex must be inside outer hex
    for i in range(n):
        x, y, a = inner[i]
        pen = _hex_dist_to_outer(x, y, a, ocx, ocy, side, 0.0)
        if pen < 0:
            penalty += pen * pen * 10  # heavier penalty for containment violation
    
    return side + penalty * 100

def construct():
    global DEADLINE
    start_time = time.time()
    
    # Multiple outer rotations to try
    best_side = 1e18
    best_inner = None
    best_center = (0.0, 0.0)
    best_oang = 0.0
    
    for oang in [0.0, 30.0, 60.0]:
        if time.time() > DEADLINE - 30:
            break
        
        # Initial guess: place hexagons in a hexagonal lattice pattern
        # Outer hex side roughly needs to fit N hexagons
        # A rough estimate: side ~ (N-1) * sqrt(3) / 2 * 2 + 2 or so
        init_side = (N - 1) * 1.75 + 2.0
        
        # Create initial positions in a hexagonal grid
        inner0 = []
        positions = []
        for row in range(N):
            for col in range(N):
                # Hex grid spacing
                dx = 1.75  # ~ sqrt(3)
                dy = 1.5
                x = col * dx + (row % 2) * dx / 2
                y = row * dy
                positions.append((x, y, 0.0))
        
        # Take first N positions (or arrange differently)
        # Actually, let's use a more compact initial arrangement
        # Place them in a hexagonal cluster
        positions = []
        # Center at 0
        # Ring 0: center
        positions.append((0.0, 0.0, 0.0))
        # Ring 1: 6 hexagons
        for i in range(6):
            a = np.radians(60 * i)
            r = 1.75
            positions.append((r * np.cos(a), r * np.sin(a), 0.0))
        # Ring 2: 12 hexagons
        for i in range(12):
            a = np.radians(30 + 30 * i)
            r = 3.5
            positions.append((r * np.cos(a), r * np.sin(a), 0.0))
        # Ring 3: 18 hexagons
        for i in range(18):
            a = np.radians(20 + 20 * i)
            r = 5.25
            positions.append((r * np.cos(a), r * np.sin(a), 0.0))
        
        # We need exactly N=11
        # Use first 11
        positions = positions[:N]
        
        # Build initial params
        params0 = []
        for (x, y, a) in positions:
            params0.extend([x, y, a])
        params0.extend([0.0, 0.0, init_side])
        params0 = np.array(params0)
        
        # Try L-BFGS-B
        try:
            res = minimize(_objective, params0, method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success or (time.time() < DEADLINE - 20):
                side_candidate = res.x[3*N + 2]
                if side_candidate < best_side and side_candidate > 0:
                    # Check validity
                    inner_candidate = []
                    for i in range(N):
                        inner_candidate.append([float(res.x[3*i]), float(res.x[3*i+1]), float(res.x[3*i+2])])
                    
                    # Quick validity check
                    valid = True
                    ocx, ocy = float(res.x[3*N]), float(res.x[3*N+1])
                    
                    for i in range(N):
                        for j in range(i+1, N):
                            d = _min_dist_between_hex(inner_candidate[i][0], inner_candidate[i][1], inner_candidate[i][2],
                                                      inner_candidate[j][0], inner_candidate[j][1], inner_candidate[j][2])
                            if d < -1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                        pen = _hex_dist_to_outer(inner_candidate[i][0], inner_candidate[i][1], inner_candidate[i][2],
                                                 ocx, ocy, side_candidate, oang)
                        if pen < -1e-6:
                            valid = False
                            break
                        if not valid:
                            break
                    
                    if valid:
                        best_side = side_candidate
                        best_inner = inner_candidate
                        best_center = (ocx, ocy)
                        best_oang = oang
        except Exception as e:
            pass
        
        # SLSQP polish if we have a candidate
        if best_side < 1e18 and best_oang == oang and time.time() < DEADLINE - 20:
            try:
                def _obj_slsqp(p):
                    n = N
                    inner_p = p[:3*n]
                    ocx, ocy, side = p[3*n], p[3*n+1], p[3*n+2]
                    inner = []
                    for i in range(n):
                        inner.append((inner_p[3*i], inner_p[3*i+1], inner_p[3*i+2]))
                    
                    penalty = 0.0
                    for i in range(n):
                        x1, y1, a1 = inner[i]
                        for j in range(i+1, n):
                            x2, y2, a2 = inner[j]
                            d = _min_dist_between_hex(x1, y1, a1, x2, y2, a2)
                            if d < 0:
                                penalty += d * d
                        pen = _hex_dist_to_outer(x1, y1, a1, ocx, ocy, side, oang)
                        if pen < 0:
                            penalty += pen * pen * 10
                    return side + penalty * 100
                
                p0 = np.array([v for row in best_inner for v in row] + [best_center[0], best_center[1], best_side])
                res2 = minimize(_obj_slsqp, p0, method='SLSQP', maxiter=200, ftol=1e-14)
                
                if res2.success or (time.time() < DEADLINE - 10):
                    side2 = res2.x[3*N+2]
                    if side2 < best_side:
                        valid = True
                        inner2 = [[float(res2.x[3*i]), float(res2.x[3*i+1]), float(res2.x[3*i+2])] for i in range(N)]
                        ocx2, ocy2 = float(res2.x[3*N]), float(res2.x[3*N+1])
                        
                        for i in range(N):
                            for j in range(i+1, N):
                                d = _min_dist_between_hex(inner2[i][0], inner2[i][1], inner2[i][2],
                                                          inner2[j][0], inner2[j][1], inner2[j][2])
                                if d < -1e-6:
                                    valid = False
                                    break
                            if not valid: break
                            pen = _hex_dist_to_outer(inner2[i][0], inner2[i][1], inner2[i][2], ocx2, ocy2, side2, oang)
                            if pen < -1e-6:
                                valid = False
                                break
                            if not valid: break
                        
                        if valid:
                            best_side = side2
                            best_inner = inner2
                            best_center = (ocx2, ocy2)
                            best_oang = oang
            except Exception:
                pass
    
    # Fallback if optimization didn't find anything
    if best_side >= 1e18 or best_inner is None:
        # Use a simple linear arrangement as fallback
        best_inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        best_center = (0.0, 0.0)
        best_side = 40.0
        best_oang = 0.0
    
    return best_inner, best_center, float(best_side), float(best_oang)
