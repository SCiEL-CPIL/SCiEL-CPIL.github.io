import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    sqrt3 = math.sqrt(3.0)

    def get_hex_vertices(cx, cy, angle_deg):
        angles = [math.radians(angle_deg + 60*k) for k in range(6)]
        return [(cx + math.cos(a), cy + math.sin(a)) for a in angles]

    def point_in_polygon(pt, poly):
        x, y = pt
        n = len(poly)
        inside = False
        p1x, p1y = poly[0]
        for i in range(1, n + 1):
            p2x, p2y = poly[i % n]
            if (y > p1y) != (y > p2y):
                xint = (p2x - p1x) * (y - p1y) / (p2y - p1y) + p1x
                if x < xint:
                    inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def segments_intersect(p1, p2, p3, p4):
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

    def polygons_disjoint(v1, v2):
        for v in v1:
            if point_in_polygon(v, v2):
                return False
        for v in v2:
            if point_in_polygon(v, v1):
                return False
        n1 = len(v1)
        n2 = len(v2)
        for i in range(n1):
            p1 = v1[i]
            p2 = v1[(i+1)%n1]
            for j in range(n2):
                p3 = v2[j]
                p4 = v2[(j+1)%n2]
                if segments_intersect(p1, p2, p3, p4):
                    return False
        return True

    def get_min_L(vertices_list):
        # The outer hexagon is centered at (0,0) with side length L.
        # Its vertices are at distance L from center, with angles 30, 90, ...
        # The condition for a point (x,y) to be inside a regular hexagon of side L centered at origin
        # is that its projection on the 6 normal directions is <= L * sqrt(3)/2.
        # The normals are at 30, 90, 150, 210, 270, 330 degrees.
        max_proj = 0.0
        normals = [math.radians(30 + 60*k) for k in range(6)]
        for v in vertices_list:
            for ang in normals:
                proj = v[0]*math.cos(ang) + v[1]*math.sin(ang)
                if proj > max_proj:
                    max_proj = proj
        # L * sqrt(3)/2 >= max_proj  => L >= max_proj / (sqrt(3)/2)
        return max_proj / (sqrt3 / 2.0)

    def min_distance_polygons(v1, v2):
        # Returns negative if overlapping, positive if disjoint.
        # Approximation: min distance between vertices and edges.
        min_d = float('inf')
        n1 = len(v1)
        n2 = len(v2)
        
        # Check vertex of v1 to edges of v2 and vice versa
        for i in range(n1):
            p = v1[i]
            for j in range(n2):
                p1 = v2[j]
                p2 = v2[(j+1)%n2]
                # Distance from point p to segment p1-p2
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                if dx == 0 and dy == 0:
                    d = math.hypot(p[0]-p1[0], p[1]-p1[1])
                else:
                    t = ((p[0]-p1[0])*dx + (p[1]-p1[1])*dy) / (dx*dx + dy*dy)
                    t = max(0, min(1, t))
                    proj_x = p1[0] + t*dx
                    proj_y = p1[1] + t*dy
                    d = math.hypot(p[0]-proj_x, p[1]-proj_y)
                if d < min_d:
                    min_d = d
        
        # If they are overlapping, the minimum distance is 0, but we need a penalty.
        # We check if any vertex is inside the other polygon.
        for v in v1:
            if point_in_polygon(v, v2):
                return -1.0 # Overlapping
        for v in v2:
            if point_in_polygon(v, v1):
                return -1.0 # Overlapping
        
        # Check edge intersections
        for i in range(n1):
            p1 = v1[i]
            p2 = v1[(i+1)%n1]
            for j in range(n2):
                p3 = v2[j]
                p4 = v2[(j+1)%n2]
                if segments_intersect(p1, p2, p3, p4):
                    return -1.0 # Overlapping

        return min_d

    def objective(params):
        if time.time() > deadline:
            return 1e10
            
        verts_list = []
        all_vertices = []
        for i in range(N):
            x, y, th = params[3*i], params[3*i+1], params[3*i+2]
            verts = get_hex_vertices(x, y, th)
            verts_list.append(verts)
            all_vertices.extend(verts)
        
        L_req = get_min_L(all_vertices)
        
        # Overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = min_distance_polygons(verts_list[i], verts_list[j])
                if d < 0:
                    penalty += 1000.0 # Heavy penalty for overlap
                else:
                    # Small penalty for being too close? No, just avoid overlap.
                    pass
                    
        return L_req + penalty

    # Initial configurations
    initializations = []

    # 1. 3-4-3-2 cluster
    # Axial coords
    coords = [
        (0,0), (1,0), (2,0),
        (-1,1), (0,1), (1,1), (2,1),
        (-1,2), (0,2), (1,2),
        (0,3), (1,3)
    ]
    params_cluster = []
    for q, r in coords:
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        # Center the cluster
        xs = [1.5 * q for q, r in coords]
        ys = [sqrt3 * (r + q / 2.0) for q, r in coords]
        cx = sum(xs)/len(xs)
        cy = sum(ys)/len(ys)
        params_cluster.extend([x - cx, y - cy, 0.0])
    initializations.append(params_cluster)

    # 2. 4x3 grid
    coords2 = []
    for r in range(3):
        for q in range(4):
            coords2.append((q, r))
    params_grid = []
    xs = [1.5 * q for q, r in coords2]
    ys = [sqrt3 * (r + q / 2.0) for q, r in coords2]
    cx = sum(xs)/len(xs)
    cy = sum(ys)/len(ys)
    for q, r in coords2:
        x = 1.5 * q - cx
        y = sqrt3 * (r + q / 2.0) - cy
        params_grid.extend([x, y, 0.0])
    initializations.append(params_grid)

    # 3. Random
    np.random.seed(42)
    params_rand = []
    for _ in range(N):
        x = np.random.uniform(-3, 3)
        y = np.random.uniform(-3, 3)
        th = np.random.uniform(0, 60)
        params_rand.extend([x, y, th])
    initializations.append(params_rand)

    best_params = None
    best_score = float('inf')

    for init_params in initializations:
        if time.time() > deadline - 30.0:
            break
        try:
            # Nelder-Mead is good for small N and non-smooth penalties
            res = minimize(objective, init_params, method='Nelder-Mead', 
                           options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Polish with L-BFGS-B if time permits
    if best_params is not None and time.time() < deadline - 10.0:
        try:
            res = minimize(objective, best_params, method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        best_params = initializations[0]

    # Final evaluation
    positions = []
    all_vertices = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        th = best_params[3*i+2]
        positions.append([x, y, th])
        verts = get_hex_vertices(x, y, th)
        all_vertices.extend(verts)
        
    L_final = get_min_L(all_vertices)
    
    # Verify disjointness
    is_valid = True
    verts_list = [get_hex_vertices(pos[0], pos[1], pos[2]) for pos in positions]
    for i in range(N):
        for j in range(i+1, N):
            if not polygons_disjoint(verts_list[i], verts_list[j]):
                is_valid = False
                break
        if not is_valid:
            break
    
    if not is_valid:
        # Fallback to initial cluster
        best_params = initializations[0]
        positions = []
        all_vertices = []
        for i in range(N):
            x = best_params[3*i]
            y = best_params[3*i+1]
            th = best_params[3*i+2]
            positions.append([x, y, th])
            verts = get_hex_vertices(x, y, th)
            all_vertices.extend(verts)
        L_final = get_min_L(all_vertices)

    return positions, (0.0, 0.0), L_final, 0.0
