import numpy as np
import time
import math

N = 12

def hex_vertices(center, side, angle):
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, 7)[:-1] + angle
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def check_validity(inner, outer_center, outer_side, outer_angle):
    if outer_side <= 0:
        return False
    
    outer_verts = hex_vertices(outer_center, outer_side, np.radians(outer_angle))
    inner_verts = []
    
    for i in range(N):
        x, y, ang = inner[i]
        v = hex_vertices((x, y), 1.0, np.radians(ang))
        inner_verts.append(v)
        
    # 1. Containment
    for i in range(N):
        v = inner_verts[i]
        for pt in v:
            if not point_in_polygon(pt, outer_verts):
                return False

    # 2. Non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            if polygons_intersect(inner_verts[i], inner_verts[j]):
                return False
                
    return True

def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    # For convex polygon, check if point is to the left of all edges (assuming CCW)
    # Or just check cross product sign consistency.
    # Vertices from hex_vertices are CCW?
    # 0, 60, 120... yes.
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if cross < -1e-9:
            return False
    return True

def polygons_intersect(poly1, poly2):
    # SAT for convex polygons
    def project(axis, poly):
        proj = np.dot(poly, axis)
        return proj.min(), proj.max()
        
    axes = []
    for poly in [poly1, poly2]:
        for i in range(len(poly)):
            edge = poly[(i+1)%len(poly)] - poly[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
                
    for axis in axes:
        min1, max1 = project(axis, poly1)
        min2, max2 = project(axis, poly2)
        if max1 < min2 or max2 < min1:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_score = np.inf
    best_config = None
    
    # Strategy: 
    # 1. Generate diverse topologies (honeycomb, grid, ring)
    # 2. For each topology, optimize the outer side length and orientation
    #    using a simple hill-climbing or Nelder-Mead on the outer parameters only,
    #    while keeping inner relative positions fixed? 
    #    No, the problem allows inner rotation and translation.
    #    However, for a "honeycomb" topology, the inner hexagons are locked in relative positions
    #    to be a rigid cluster. We can then just rotate and translate the cluster, and adjust outer.
    #    This reduces the search space significantly and is likely to find the global optimum for that topology.
    
    # Let's generate a few specific topologies:
    # Topology 1: Honeycomb cluster (7 center, 5 ring)
    # Topology 2: 3x4 Grid
    # Topology 3: Random init
    
    def optimize_topology(inner_init, time_budget):
        # inner_init: list of (x, y, angle_deg)
        # We assume the inner hexagons form a rigid body initially.
        # We will optimize: 
        # - Outer side length (S)
        # - Outer angle (theta_outer)
        # - Cluster rotation (theta_cluster)
        # - Cluster translation (tx, ty) relative to outer center?
        #   Actually, we can fix outer center at (0,0) and optimize cluster center (tx, ty) and rotation.
        
        # Parameters: [S, theta_outer, theta_cluster, tx, ty]
        # Note: Inner hexagons relative to cluster center are fixed from init.
        
        # Precompute relative positions in cluster frame (angle 0)
        # But wait, inner_init angles are absolute. 
        # Let's define the cluster frame such that inner positions are relative to (0,0) with their given angles.
        # When we rotate the cluster by alpha, all inner hexagons rotate by alpha around cluster center?
        # No, rigid body means: 
        #   pos_i' = R(alpha) * (pos_i - c) + c'
        #   ang_i' = ang_i + alpha
        # Where c is cluster center.
        
        # Let's simplify: 
        # 1. Compute bounding box of the inner cluster (in its local frame).
        # 2. Find min enclosing hexagon for that cluster.
        # This is a simpler geometric problem: MinEnclosingHexagon(points)
        
        # Actually, the problem is: Given a set of hexagons (rigid cluster), find the min side length hexagon
        # that can contain them, allowing rotation of the cluster and the container.
        
        # Objective: Minimize S
        # Variables: alpha (cluster rotation), beta (container rotation), tx, ty (cluster center offset)
        # Constraint: All inner hex vertices inside outer hex.
        
        def objective(params):
            S, beta, alpha, tx, ty = params
            
            # Outer hex
            outer_c = (0, 0)
            outer_verts = hex_vertices(outer_c, S, np.radians(beta))
            
            # Inner hexagons
            total_violation = 0.0
            for i in range(N):
                x, y, ang_deg = inner_init[i]
                ang_rad = np.radians(ang_deg)
                
                # Transform to global
                # Rotate around origin then translate?
                # Let's say cluster center is (tx, ty).
                # Local pos (x, y). Global pos = R(alpha)*(x,y) + (tx, ty)
                ca, sa = np.cos(np.radians(alpha)), np.sin(np.radians(alpha))
                gx = ca * x - sa * y + tx
                gy = sa * x + ca * y + ty
                g_ang = ang_deg + alpha
                
                v = hex_vertices((gx, gy), 1.0, np.radians(g_ang))
                
                # Check containment
                for pt in v:
                    if not point_in_polygon(pt, outer_verts):
                        # Penalty: distance outside
                        # Estimate distance to boundary
                        min_d = np.inf
                        for k in range(6):
                            p1 = outer_verts[k]
                            p2 = outer_verts[(k+1)%6]
                            d = distance_point_to_segment(pt, p1, p2)
                            min_d = min(min_d, d)
                        total_violation += min_d * 10.0
            
            return S + total_violation
            
        # Initial guess for params
        # Estimate required S from bounding box
        max_r = max(math.hypot(p[0], p[1]) for p in inner_init)
        S_init = max_r + 2.0
        
        init_params = [S_init, 0.0, 0.0, 0.0, 0.0]
        
        # Optimization
        from scipy.optimize import minimize
        
        best_local = None
        best_local_val = np.inf
        
        # Try multiple random starts for alpha, beta
        for _ in range(5):
            if time.time() > deadline:
                break
            
            params0 = [
                S_init * (0.9 + 0.2*np.random.rand()),
                np.random.rand() * 60.0,
                np.random.rand() * 60.0,
                np.random.rand() * 2.0 - 1.0,
                np.random.rand() * 2.0 - 1.0
            ]
            
            try:
                res = minimize(objective, params0, method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.fun < best_local_val:
                    best_local_val = res.fun
                    best_local = res.x
            except Exception:
                pass
            
        if best_local is not None:
            S, beta, alpha, tx, ty = best_local
            
            # Construct the final config
            inner_final = []
            ca, sa = np.cos(np.radians(alpha)), np.sin(np.radians(alpha))
            for i in range(N):
                x, y, ang_deg = inner_init[i]
                gx = ca * x - sa * y + tx
                gy = sa * x + ca * y + ty
                g_ang = ang_deg + alpha
                inner_final.append((gx, gy, g_ang))
                
            outer_center = (0, 0)
            outer_side = S
            outer_angle = beta
            
            # Verify
            if check_validity(inner_final, outer_center, outer_side, outer_angle):
                return outer_side, inner_final, outer_center, outer_angle, outer_angle
            
        return None
        
    # --- Topology Generators ---
    
    def gen_honeycomb():
        # 7 in hex cluster + 5 around
        inner = []
        # Center
        inner.append((0, 0, 0))
        # Ring 1 (6 hexagons)
        # Distance between centers of touching hexagons:
        # For pointy-top (vertex at 90), dist is 2? 
        # Our hex_vertices: vertex at 0 deg is (1,0). 
        # This is "flat-top" if we consider y up? No, standard math angle.
        # Let's just use distance 2.0 and adjust angles.
        for k in range(6):
            ang = np.radians(k * 60)
            inner.append((2.0 * np.cos(ang), 2.0 * np.sin(ang), 0))
        # Ring 2 (need 5 more, total 12)
        # Add 5 from ring 2
        # Ring 2 distances: 
        # Neighbors of ring 1 neighbors.
        # Let's just place 5 randomly in a wider ring and let optimizer fix?
        # Better: place them at specific spots.
        # Spots: (3.0, 0), (1.5, 2.6), (-1.5, 2.6), (-3.0, 0), (-1.5, -2.6) -- approx
        # Let's use angles 0, 72, 144, 216, 288 at distance 3.5
        for k in range(5):
            ang = np.radians(k * 72)
            inner.append((3.5 * np.cos(ang), 3.5 * np.sin(ang), 0))
        return inner

    def gen_grid():
        inner = []
        # 3x4 grid
        # Spacing:
        # Hex width 2, height 1.732
        # Horizontal spacing: 2? No, they overlap.
        # If vertex at 0, width is 2.
        # Staggered grid is better.
        # Row 0: x = 0, 3, 6
        # Row 1: x = 1.5, 4.5
        # Row 2: x = 0, 3, 6
        # Row 3: x = 1.5, 4.5 (if 4 rows)
        # Let's do 3 rows of 4.
        # This is tight.
        # Let's use 4 columns, 3 rows.
        for row in range(3):
            for col in range(4):
                x = col * 3.0
                y = row * 2.6
                if row % 2 == 1:
                    x += 1.5
                inner.append((x, y, 0))
        # Center it
        cx = sum(p[0] for p in inner) / N
        cy = sum(p[1] for p in inner) / N
        inner = [(x-cx, y-cy, a) for x,y,a in inner]
        return inner

    def gen_random():
        inner = []
        for i in range(N):
            x = np.random.rand() * 6.0 - 3.0
            y = np.random.rand() * 6.0 - 3.0
            a = np.random.rand() * 60.0
            inner.append((x, y, a))
        return inner

    topologies = [gen_honeycomb(), gen_grid()]
    
    # Add a few random ones
    for _ in range(3):
        topologies.append(gen_random())
        
    for topo in topologies:
        if time.time() > deadline:
            break
        
        res = optimize_topology(topo, 10.0)
        if res is not None:
            S, inner, oc, oa = res
            if S < best_score:
                best_score = S
                best_config = (inner, oc, S, oa)
                
    # If no config found, fallback
    if best_config is None:
        # Fallback to a known valid, though large, config
        # 4x3 grid with large spacing
        inner = []
        for row in range(3):
            for col in range(4):
                x = col * 4.0 - 6.0
                y = row * 3.5 - 3.5
                inner.append((x, y, 0))
        outer_side = 12.0
        outer_center = (0, 0)
        outer_angle = 0.0
        # Verify fallback
        if not check_validity(inner, outer_center, outer_side, outer_angle):
             # Even larger
             outer_side = 15.0
        return inner, outer_center, outer_side, outer_angle

    inner, oc, S, oa = best_config
    return inner, oc, S, oa

# Helper functions needed for optimize_topology objective

def distance_point_to_segment(p, a, b):
    ab = b - a
    ap = p - a
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq < 1e-9:
        return np.linalg.norm(p - a)
    t = np.dot(ap, ab) / ab_len_sq
    t = np.clip(t, 0, 1)
    proj = a + t * ab
    return np.linalg.norm(p - proj)
