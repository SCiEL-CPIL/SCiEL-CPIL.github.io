import numpy as np
import time
from scipy.optimize import differential_evolution

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 130s safety margin for 150s limit

    # Flat-top regular hexagon of side 1, centered at origin.
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees.
    base_hex = np.array([
        [np.cos(np.radians(60 * i)), np.sin(np.radians(60 * i))] for i in range(6)
    ])

    def hex_to_world(cx, cy, angle_deg):
        a = np.radians(angle_deg)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        return base_hex @ R.T + np.array([cx, cy])

    def min_dist_between_hexes(hex1, hex2):
        """Minimum distance between two convex polygons (hexagons).
        Returns 0 if they intersect or touch, > 0 if separated."""
        # Check if any edge of hex1 intersects any edge of hex2
        # Also check if any vertex of one is inside the other
        from scipy.spatial import ConvexHull
        try:
            # Check edge intersections
            def segments_intersect(p1, p2, p3, p4):
                def cross(o, a, b):
                    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
                d1 = cross(p3, p4, p1)
                d2 = cross(p3, p4, p2)
                d3 = cross(p1, p2, p3)
                d4 = cross(p1, p2, p4)
                if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
                   ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                    return True
                return False
            
            for i in range(6):
                for j in range(6):
                    p1 = hex1[i]
                    p2 = hex1[(i+1) % 6]
                    p3 = hex2[j]
                    p4 = hex2[(j+1) % 6]
                    if segments_intersect(p1, p2, p3, p4):
                        return 0.0
            
            # Check if any vertex of hex1 is inside hex2
            def point_in_polygon(point, polygon):
                x, y = point
                n = len(polygon)
                inside = False
                j = n - 1
                for i in range(n):
                    xi, yi = polygon[i]
                    xj, yj = polygon[j]
                    if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                        inside = not inside
                    j = i
                return inside
            
            for i in range(6):
                if point_in_polygon(hex1[i], hex2):
                    return 0.0
            for i in range(6):
                if point_in_polygon(hex2[i], hex1):
                    return 0.0
            
            # Minimum distance between edges
            min_d = float('inf')
            def point_segment_dist(p, a, b):
                ab = b - a
                ap = p - a
                ab2 = np.dot(ab, ab)
                if ab2 == 0:
                    return np.linalg.norm(p - a)
                t = np.dot(ap, ab) / ab2
                t = max(0, min(1, t))
                proj = a + t * ab
                return np.linalg.norm(p - proj)
            
            for i in range(6):
                for j in range(6):
                    p1 = hex1[i]
                    p2 = hex1[(i+1) % 6]
                    p3 = hex2[j]
                    p4 = hex2[(j+1) % 6]
                    d = min(
                        point_segment_dist(p1, p3, p4),
                        point_segment_dist(p2, p3, p4),
                        point_segment_dist(p3, p1, p2),
                        point_segment_dist(p4, p1, p2)
                    )
                    if d < min_d:
                        min_d = d
            return min_d
        except:
            return 0.0

    def get_all_hexes(params):
        """params: array of length 3*N: [x0, y0, a0, x1, y1, a1, ...]"""
        hexes = []
        for i in range(N):
            cx = params[3*i]
            cy = params[3*i + 1]
            ang = params[3*i + 2]
            hexes.append(hex_to_world(cx, cy, ang))
        return hexes

    def max_vertex_radius(params):
        hexes = get_all_hexes(params)
        max_r = 0
        for h in hexes:
            for v in h:
                r = np.linalg.norm(v)
                if r > max_r:
                    max_r = r
        return max_r

    def outer_hex_contain(params, L):
        """Check if all vertices of all inner hexes are inside regular hexagon
        centered at origin with side L, aligned with axes (flat-top)."""
        # For a flat-top regular hexagon with side L centered at origin:
        # The six sides are at distance apothem = L * sqrt(3) / 2 from center.
        # The normals point at angles 0, 60, 120, 180, 240, 300.
        apothem = L * np.sqrt(3) / 2
        hexes = get_all_hexes(params)
        for h in hexes:
            for v in h:
                for k in range(6):
                    theta = np.radians(60 * k)
                    n = np.array([np.cos(theta), np.sin(theta)])
                    if np.dot(v, n) > apothem + 1e-9:
                        return False
        return True

    def minimal_L(params):
        """Find the minimal L such that all hexes fit in a flat-top outer hexagon."""
        # We can also rotate the outer hexagon. Let's optimize over rotation too.
        # But for simplicity, first try fixed rotation.
        max_r = max_vertex_radius(params)
        # For a point at distance r from center, the minimal L needed when outer hex is
        # rotated optimally: we need to check all 6 directions.
        # Actually, let's do a grid search over outer rotation.
        hexes = get_all_hexes(params)
        all_verts = []
        for h in hexes:
            all_verts.extend(h.tolist())
        all_verts = np.array(all_verts)
        
        best_L = float('inf')
        for alpha_deg in range(0, 30, 1):
            alpha = np.radians(alpha_deg)
            max_proj = 0
            for k in range(6):
                theta = alpha + k * np.pi / 3
                n = np.array([np.cos(theta), np.sin(theta)])
                projs = all_verts @ n
                max_p = np.max(projs)
                if max_p > max_proj:
                    max_proj = max_p
            L = 2 * max_proj / np.sqrt(3)
            if L < best_L:
                best_L = L
        return best_L

    def overlap_penalty(params):
        hexes = get_all_hexes(params)
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = min_dist_between_hexes(hexes[i], hexes[j])
                if d < 1e-6:
                    penalty += 10.0
                elif d < 0.01:
                    penalty += (0.01 - d) * 1000.0
        return penalty

    def objective(params):
        if time.time() > deadline:
            return 1e10
        L = minimal_L(params)
        pen = overlap_penalty(params)
        if pen > 0:
            return L + pen * 10.0
        return L

    # Initial guess: hexagonal packing
    # 1 center + 6 ring1 + 4 ring2 (selected)
    # Flat-top hex packing:
    # Ring 1: 6 hexes at distance sqrt(3) from center, at angles 0, 60, 120, 180, 240, 300
    # For flat-top hexes touching, center-to-center distance is 2 * apothem = sqrt(3)
    s3 = np.sqrt(3)
    
    init_x = [0.0]
    init_y = [0.0]
    init_a = [0.0]
    
    # Ring 1: 6 hexes
    for k in range(6):
        theta = np.radians(60 * k)
        cx = s3 * np.cos(theta)
        cy = s3 * np.sin(theta)
        init_x.append(cx)
        init_y.append(cy)
        init_a.append(0.0)
    
    # Ring 2: pick 4 more to make 11 total
    # Distance from center for ring 2: 2*sqrt(3)
    # Angles: 0, 60, 120, 180, 240, 300
    # But we need to avoid overlapping with ring 1 hexes.
    # Let's place at 30, 90, 150, 210, 270, 330 (between ring 1 hexes)
    # Distance: 2*sqrt(3)
    for k in range(4):
        theta = np.radians(30 + 120 * k)  # 30, 150, 270, ...
        # Actually let's use 30, 90, 150, 210 for 4 hexes
        cx = 2 * s3 * np.cos(theta)
        cy = 2 * s3 * np.sin(theta)
        init_x.append(cx)
        init_y.append(cy)
        init_a.append(0.0)
    
    # That gives 1 + 6 + 4 = 11
    init_params = np.array([x, y, a] for x, y, a in zip(init_x, init_y, init_a)).flatten()
    
    # Bounds
    bounds = [(-10, 10)] * N + [(-10, 10)] * N + [(0, 180)] * N
    
    # Run differential evolution
    best_result = None
    best_L = float('inf')
    
    try:
        result = differential_evolution(
            objective, bounds, x0=init_params,
            maxiter=50, popsize=20, tol=1e-4,
            mutation=(0.3, 0.8), recombination=0.8,
            seed=42, polish=False, init='latinhypercube'
        )
        if result.success or result.fun < best_L:
            if time.time() < deadline - 5:
                best_L = result.fun
                best_params = result.x
        else:
            best_params = init_params
            best_L = objective(init_params)
    except Exception as e:
        print(f"DE failed: {e}")
        best_params = init_params
        best_L = objective(init_params)
    
    # Try to improve with local optimization if time permits
    if time.time() < deadline - 20:
        from scipy.optimize import minimize
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.fun < best_L:
                best_L = res.fun
                best_params = res.x
        except Exception as e:
            print(f"Local opt failed: {e}")
    
    # Final check and clean up
    final_L = minimal_L(best_params)
    final_pen = overlap_penalty(best_params)
    
    if final_pen > 0.01:
        # Try to separate overlapping hexes slightly
        # Just use the initial configuration which should be valid
        best_params = init_params
        final_L = minimal_L(best_params)
    
    # Extract centers and angles
    inner_array = np.zeros((N, 3))
    for i in range(N):
        inner_array[i, 0] = best_params[3*i]
        inner_array[i, 1] = best_params[3*i + 1]
        inner_array[i, 2] = best_params[3*i + 2]
    
    # Find optimal outer rotation
    hexes = get_all_hexes(best_params)
    all_verts = []
    for h in hexes:
        all_verts.extend(h.tolist())
    all_verts = np.array(all_verts)
    
    best_L_out = float('inf')
    best_alpha = 0.0
    for alpha_deg in np.linspace(0, 30, 61):
        alpha = np.radians(alpha_deg)
        max_proj = 0
        for k in range(6):
            theta = alpha + k * np.pi / 3
            n = np.array([np.cos(theta), np.sin(theta)])
            projs = all_verts @ n
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
        L = 2 * max_proj / np.sqrt(3)
        if L < best_L_out:
            best_L_out = L
            best_alpha = alpha_deg
    
    return inner_array, (0.0, 0.0), best_L_out, best_alpha
