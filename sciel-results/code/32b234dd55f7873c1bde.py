import numpy as np
import time
from scipy.optimize import minimize

N = 12

# Precompute normals for the outer hexagon (angle 0)
# Vertices at 0, 60, ...
# Normals to sides are at 30, 90, ...
# Side length S. Distance from center to side is S * sqrt(3) / 2
# Normal vectors
angles_normal = np.pi/6 + np.arange(6) * np.pi/3
normals = np.column_stack((np.cos(angles_normal), np.sin(angles_normal)))

def get_hex_vertices(center, angle_deg, side=1.0):
    """
    Returns vertices of a regular hexagon.
    center: (x, y)
    angle_deg: rotation in degrees
    side: side length
    """
    theta = np.deg2rad(angle_deg)
    # Vertices at angles theta, theta+60, ...
    # Standard hexagon vertices for side 1 are at distance 1 from center.
    # So radius = side.
    r = side
    angles = theta + np.arange(6) * np.pi/3
    x = center[0] + r * np.cos(angles)
    y = center[1] + r * np.sin(angles)
    return np.column_stack((x, y))

def point_in_hexagon(p, center, side, angle_deg=0.0):
    """
    Returns True if point p is inside the hexagon.
    Uses the half-plane method.
    """
    theta = np.deg2rad(angle_deg)
    # Normals to the sides
    # Sides are perpendicular to vectors at theta + 30, theta + 90, ...
    normals_theta = theta + np.pi/6 + np.arange(6) * np.pi/3
    nx = np.cos(normals_theta)
    ny = np.sin(normals_theta)
    
    dx = p[0] - center[0]
    dy = p[1] - center[1]
    
    # Distance from center along normal
    dists = dx * nx + dy * ny
    max_dist = side * np.sqrt(3) / 2
    
    return np.all(dists <= max_dist + 1e-9)

def dist_point_to_hex_boundary(p, center, side, angle_deg=0.0):
    """
    Returns the distance from point p to the boundary of the hexagon.
    Positive if outside, negative if inside.
    """
    theta = np.deg2rad(angle_deg)
    normals_theta = theta + np.pi/6 + np.arange(6) * np.pi/3
    nx = np.cos(normals_theta)
    ny = np.sin(normals_theta)
    
    dx = p[0] - center[0]
    dy = p[1] - center[1]
    
    dists = dx * nx + dy * ny
    max_dist = side * np.sqrt(3) / 2
    
    # The "excess" is the maximum of (dist - max_dist)
    # If all dists < max_dist, it's inside. The most "outside" component is the max.
    # Actually, if inside, all dists < max_dist.
    # The distance to the boundary is max_dist - min(dists)? No.
    # The point is inside if all dists <= max_dist.
    # The distance to the closest boundary is min(max_dist - dists) if inside?
    # No, the distance to the boundary along the normal is max_dist - dist.
    # The geometric distance to the boundary is the minimum of these values?
    # Yes, for a convex polygon, the distance to the boundary from an interior point
    # is the minimum of the distances to the supporting lines.
    
    excess = np.max(dists - max_dist)
    if excess > 0:
        return excess
    else:
        # Inside. Distance to boundary is min(max_dist - dists)
        return -(max_dist - np.min(dists))

def min_dist_hexagons(hex1_verts, hex2_verts):
    """
    Approximate min distance between two convex polygons.
    Returns negative if overlapping (penetration depth approx).
    """
    min_d = np.inf
    overlap = False
    
    # Check if any vertex of 1 is inside 2
    # We need center and side for hex2 to use point_in_hexagon?
    # We only have vertices.
    # We can use a point-in-polygon algorithm for convex polygons.
    
    def point_in_convex_poly(p, poly):
        # Check cross products
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i+1)%n]
            # Vector p1->p2
            v1 = p2 - p1
            # Vector p1->p
            v2 = p - p1
            # Cross product z-component
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            if cross < -1e-9:
                return False
        return True

    # Check vertices of hex1 against hex2
    for v in hex1_verts:
        if point_in_convex_poly(v, hex2_verts):
            # Find distance to boundary of hex2
            # Approximate by distance to closest edge
            min_dist_edge = np.inf
            for i in range(6):
                p1 = hex2_verts[i]
                p2 = hex2_verts[(i+1)%6]
                # Project v onto line p1-p2
                dp = p2 - p1
                dp_len_sq = np.dot(dp, dp)
                if dp_len_sq == 0: continue
                t = np.dot(v - p1, dp) / dp_len_sq
                t = np.clip(t, 0, 1)
                closest = p1 + t * dp
                d = np.linalg.norm(v - closest)
                if d < min_dist_edge:
                    min_dist_edge = d
            # It is inside, so distance is negative
            return -min_dist_edge
            
    # Check vertices of hex2 against hex1
    for v in hex2_verts:
        if point_in_convex_poly(v, hex1_verts):
            min_dist_edge = np.inf
            for i in range(6):
                p1 = hex1_verts[i]
                p2 = hex1_verts[(i+1)%6]
                dp = p2 - p1
                dp_len_sq = np.dot(dp, dp)
                if dp_len_sq == 0: continue
                t = np.dot(v - p1, dp) / dp_len_sq
                t = np.clip(t, 0, 1)
                closest = p1 + t * dp
                d = np.linalg.norm(v - closest)
                if d < min_dist_edge:
                    min_dist_edge = d
            return -min_dist_edge
            
    # If no vertex inside, check edge intersections
    # If they intersect, they overlap.
    # We can return a small negative number or 0.
    # For gradient purposes, 0 is bad.
    # Let's check edge-edge min distance.
    
    for i in range(6):
        p1 = hex1_verts[i]
        p2 = hex1_verts[(i+1)%6]
        for j in range(6):
            p3 = hex2_verts[j]
            p4 = hex2_verts[(j+1)%6]
            
            # Line segment intersection check
            # d1 = cross(p4-p3, p1-p3)
            # d2 = cross(p4-p3, p2-p3)
            # d3 = cross(p2-p1, p3-p1)
            # d4 = cross(p2-p1, p4-p1)
            
            v1 = p2 - p1
            v2 = p4 - p3
            v3 = p3 - p1
            
            denom = v1[0]*v2[1] - v1[1]*v2[0]
            if abs(denom) < 1e-9:
                # Parallel
                # Calculate min distance between segments
                # Project p3, p4 onto line 1
                # This is getting complex.
                # For now, if parallel, just compute point-to-segment distances
                for p in [p3, p4]:
                    t = np.dot(p - p1, v1) / np.dot(v1, v1)
                    t = np.clip(t, 0, 1)
                    closest = p1 + t * v1
                    d = np.linalg.norm(p - closest)
                    min_d = min(min_d, d)
                for p in [p1, p2]:
                    t = np.dot(p - p3, v2) / np.dot(v2, v2)
                    t = np.clip(t, 0, 1)
                    closest = p3 + t * v2
                    d = np.linalg.norm(p - closest)
                    min_d = min(min_d, d)
                continue
                
            t1 = (v3[0]*v2[1] - v3[1]*v2[0]) / denom
            t2 = (v3[0]*v1[1] - v3[1]*v1[0]) / denom
            
            if 0 <= t1 <= 1 and 0 <= t2 <= 1:
                # Intersect
                return 0.0 # Or small negative
                
    return min_d

def cost_func(params):
    S = params[0]
    if S < 0: S = 1e-6
    
    inner = []
    for i in range(N):
        x = params[1 + 3*i]
        y = params[2 + 3*i]
        ang = params[3 + 3*i]
        inner.append((x, y, ang))
        
    penalty = 0.0
    
    # Wall penalty
    for x, y, ang in inner:
        verts = get_hex_vertices((x, y), ang)
        for v in verts:
            d = dist_point_to_hex_boundary(v, (0,0), S, 0.0)
            if d > 0:
                penalty += d * d
                
    # Collision penalty
    verts_list = []
    for x, y, ang in inner:
        verts_list.append(get_hex_vertices((x, y), ang))
        
    for i in range(N):
        for j in range(i+1, N):
            d = min_dist_hexagons(verts_list[i], verts_list[j])
            if d < 0:
                penalty += (-d) * (-d)
            # If d == 0, they touch. No penalty.
            # If d > 0, no penalty.
            
    return S + 10.0 * penalty

def construct():
    start_time = time.time()
    best_S = 100.0
    best_params = None
    
    # Initial guess
    # Center at 0,0
    # 6 around at dist 2
    # 5 more at dist 4
    
    for restart in range(5):
        if time.time() - start_time > 120:
            break
            
        # Randomize initial angles and positions slightly
        np.random.seed(restart)
        
        init = [4.0] # S
        # Center
        init.extend([0.0, 0.0, 0.0])
        # Ring 1
        for k in range(6):
            ang = k * 60
            rad = np.deg2rad(ang)
            x = 2.0 * np.cos(rad)
            y = 2.0 * np.sin(rad)
            init.extend([x, y, 0.0])
        # Ring 2 (5 hexagons)
        for k in range(5):
            ang = (k * 72) + 36 # Offset
            rad = np.deg2rad(ang)
            x = 4.0 * np.cos(rad)
            y = 4.0 * np.sin(rad)
            init.extend([x, y, 0.0])
            
        init = np.array(init)
        
        # Optimize
        try:
            res = minimize(cost_func, init, method='Powell', options={'maxiter': 500, 'ftol': 1e-6})
            if res.fun < best_S:
                # Check validity
                # Extract S
                S_val = res.x[0]
                # Check penalties
                # We need to re-evaluate to ensure it's not just a low S with high penalty
                # But cost_func includes penalty.
                # If penalty is high, cost is high.
                # So if res.fun is low, penalty is likely low.
                
                # Verify
                inner = []
                for i in range(N):
                    x = res.x[1 + 3*i]
                    y = res.x[2 + 3*i]
                    ang = res.x[3 + 3*i]
                    inner.append([x, y, ang])
                    
                # Quick check: are all vertices inside?
                valid = True
                verts_list = []
                for x, y, ang in inner:
                    verts = get_hex_vertices((x, y), ang)
                    verts_list.append(verts)
                    for v in verts:
                        d = dist_point_to_hex_boundary(v, (0,0), S_val, 0.0)
                        if d > 1e-5:
                            valid = False
                            break
                    if not valid: break
                            
                if valid:
                    # Check collisions
                    for i in range(N):
                        for j in range(i+1, N):
                            d = min_dist_hexagons(verts_list[i], verts_list[j])
                            if d < -1e-5:
                                valid = False
                                break
                        if not valid: break
                        
                    if valid:
                        best_S = S_val
                        best_params = res.x.copy()
                        
        except Exception as e:
            pass

    if best_params is None:
        # Fallback
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

    inner = []
    for i in range(N):
        x = best_params[1 + 3*i]
        y = best_params[2 + 3*i]
        ang = best_params[3 + 3*i]
        inner.append([x, y, ang])
        
    return inner, (0.0, 0.0), best_S, 0.0
