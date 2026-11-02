import math
import time
import numpy as np
from scipy.optimize import minimize, basinhopping

N = 11

def _hex_vertices(cx, cy, s, angle_deg):
    """
    Returns vertices of a regular hexagon centered at (cx, cy) with side s
    and rotation angle_deg (degrees).
    """
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return verts

def _gjk_distance(poly1, poly2):
    """
    Compute the minimum distance between two convex polygons using GJK algorithm.
    Returns 0 if they intersect, otherwise the distance.
    poly1, poly2: list of (x, y) tuples.
    """
    def support_dir(d):
        # Support function for Minkowski difference P1 - P2
        # Maximize dot(p, d) for p in P1 and minimize dot(p, d) for p in P2
        # Equivalent to maximizing dot(p1 - p2, d)
        
        # Find p1 in P1 that maximizes dot(p1, d)
        max_dot1 = -float('inf')
        p1 = poly1[0]
        for p in poly1:
            d1 = p[0]*d[0] + p[1]*d[1]
            if d1 > max_dot1:
                max_dot1 = d1
                p1 = p
        
        # Find p2 in P2 that minimizes dot(p2, d) -> maximizes dot(-p2, d)
        min_dot2 = float('inf')
        p2 = poly2[0]
        for p in poly2:
            d2 = p[0]*d[0] + p[1]*d[1]
            if d2 < min_dot2:
                min_dot2 = d2
                p2 = p
                
        return (p1[0] - p2[0], p1[1] - p2[1])

    # Initial simplex
    d = (1.0, 0.0)
    a = support_dir(d)
    
    # If origin is inside the support point, they intersect? 
    # No, GJK builds a simplex.
    
    # Simple GJK implementation for 2D
    # Simplex starts with one point
    simplex = [a]
    
    for _ in range(100):
        # If origin is in simplex, distance is 0
        if len(simplex) == 1:
            p = simplex[0]
            if p[0]*p[0] + p[1]*p[1] < 1e-8:
                return 0.0
            d = (-p[0], -p[1])
        elif len(simplex) == 2:
            a, b = simplex
            # Check if origin is in triangle (0, a, b)
            # We can check if 0 is in the wedge of the segment ab
            # Or simpler: compute closest point on segment ab to 0
            # If closest point is 0, distance is 0
            # Otherwise, set d to direction from closest point to 0
            
            # Vector ab
            abx = b[0] - a[0]
            aby = b[1] - a[1]
            
            # Vector ao
            aox = 0.0 - a[0]
            aoy = 0.0 - a[1]
            
            # Vector bo
            box = 0.0 - b[0]
            boy = 0.0 - b[1]
            
            # Dot products
            d1 = aox*abx + aoy*aby
            d2 = box*abx + boy*aby
            
            if d1 <= 0:
                # Closest to a
                d = (a[0], a[1])
                simplex = [a]
            elif d2 >= 0:
                # Closest to b
                d = (b[0], b[1])
                simplex = [b]
            else:
                # Closest is on segment
                # Project 0 onto ab
                # t = -d1 / (abx^2 + aby^2)
                # Closest point c = a + t * ab
                # Direction from c to 0 is -c
                # But for GJK, we just need the direction to the closest point on the simplex
                # If 0 is not in simplex, the direction is from the closest point on the boundary to 0
                # In 2D, if 0 is not in the triangle formed by origin and edge, 
                # the direction is perpendicular to the edge? No.
                # The direction is from the closest point on the segment to the origin.
                # Since we are in the "middle" case, the closest point is on the segment.
                # The vector from that point to 0 is the new search direction.
                # Let's compute the projection.
                t = -d1 / (abx*abx + aby*aby)
                cx = a[0] + t * abx
                cy = a[1] + t * aby
                d = (-cx, -cy)
                # Keep the whole segment? Or reduce?
                # In GJK, we keep the simplex. If we add a new point, we might get a triangle.
                # But in 2D, if we have a segment and the new point is in the direction of the closest point,
                # we might just need to check if the new point crosses the origin.
                # Actually, standard GJK:
                # If 0 is in simplex, return 0.
                # Else, find closest point on simplex to 0.
                # New direction is from that point to 0.
                # Add new support point.
                # If new point is in direction of 0 (dot > 0), we might have crossed.
                pass
        
        elif len(simplex) == 3:
            # Triangle
            a, b, c = simplex
            # Check if 0 is inside triangle
            # Using cross products
            def cross(o, a, b):
                return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
            
            d1 = cross(0, a, b)
            d2 = cross(0, b, c)
            d3 = cross(0, c, a)
            
            has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
            has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
            
            if not (has_neg and has_pos):
                return 0.0
            
            # Find closest edge
            # If 0 is outside, we discard the vertex furthest from 0
            # Or simpler: reduce simplex to the edge closest to 0
            # This is getting complex. 
            # For robustness in this specific task (hexagons), 
            # a simpler approach might be to just check vertex-in-polygon and edge-intersection 
            # but that was the old slow way.
            
            # Let's try a different, more robust GJK for 2D or just use shapely if available? 
            # No shapely. 
            # Let's stick to a simpler distance check for hexagons specifically?
            # Or use the fact that they are convex polygons.
            
            # Actually, for the purpose of this improvement, 
            # let's use a simpler "distance to segment" based GJK reduction.
            # If we have a triangle, we can check which edge is closest to 0.
            
            # Distance from 0 to edge AB
            def dist_point_segment(p, a, b):
                abx = b[0]-a[0]
                aby = b[1]-a[1]
                apx = p[0]-a[0]
                apy = p[1]-a[1]
                ab_len_sq = abx*abx + aby*aby
                if ab_len_sq == 0:
                    return math.hypot(apx, apy)
                t = (apx*abx + apy*aby) / ab_len_sq
                t = max(0, min(1, t))
                projx = a[0] + t*abx
                projy = a[1] + t*aby
                return math.hypot(p[0]-projx, p[1]-projy)
            
            d_ab = dist_point_segment((0,0), a, b)
            d_bc = dist_point_segment((0,0), b, c)
            d_ca = dist_point_segment((0,0), c, a)
            
            min_d = min(d_ab, d_bc, d_ca)
            
            if min_d == d_ab:
                # Closest to AB, discard C
                simplex = [a, b]
                # Direction is from closest point on AB to 0
                # We already computed this in the 2-point case logic, 
                # but we need to re-derive d.
                # Let's just re-calculate d from the new simplex in the next iteration.
                # To avoid infinite loops, we need to set d properly.
                # The direction should be from the closest point on the retained simplex to 0.
                # We'll let the loop handle it by setting d based on the new simplex.
                # But we need to break the loop if we don't make progress.
                pass
            elif min_d == d_bc:
                simplex = [b, c]
            else:
                simplex = [c, a]
                
        # New support point
        b_new = support_dir(d)
        
        # Check if b_new is in the direction of 0
        # If dot(b_new, d) <= 0, we have found the minimum distance
        if b_new[0]*d[0] + b_new[1]*d[1] <= 0:
            # The distance is the distance from 0 to the closest point on the current simplex
            # We need to calculate this exactly.
            if len(simplex) == 1:
                p = simplex[0]
                return math.hypot(p[0], p[1])
            elif len(simplex) == 2:
                a, b = simplex
                abx = b[0]-a[0]
                aby = b[1]-a[1]
                aox = -a[0]
                aoy = -a[1]
                d1 = aox*abx + aoy*aby
                if d1 <= 0:
                    return math.hypot(a[0], a[1])
                box = -b[0]
                boy = -b[1]
                d2 = box*abx + boy*aby
                if d2 >= 0:
                    return math.hypot(b[0], b[1])
                # Project 0 onto AB
                t = -d1 / (abx*abx + aby*aby)
                cx = a[0] + t*abx
                cy = a[1] + t*aby
                return math.hypot(cx, cy)
            else:
                # Triangle
                # Since we didn't return 0, 0 is outside.
                # We can just use the min distance to edges
                a, b, c = simplex
                def dist_point_segment(p, a, b):
                    abx = b[0]-a[0]
                    aby = b[1]-a[1]
                    apx = p[0]-a[0]
                    apy = p[1]-a[1]
                    ab_len_sq = abx*abx + aby*aby
                    if ab_len_sq == 0:
                        return math.hypot(apx, apy)
                    t = (apx*abx + apy*aby) / ab_len_sq
                    t = max(0, min(1, t))
                    projx = a[0] + t*abx
                    projy = a[1] + t*aby
                    return math.hypot(p[0]-projx, p[1]-projy)
                
                return min(dist_point_segment((0,0), a, b), 
                           dist_point_segment((0,0), b, c), 
                           dist_point_segment((0,0), c, a))
        
        # Add new point
        simplex.append(b_new)
        
        # If we have > 3 points, we need to reduce. 
        # But in 2D, the simplex should never exceed 3 points if we reduce properly.
        # If it does, we have a bug in the reduction logic.
        # For safety, if we have 4 points, we just keep the 3 furthest? No, that's wrong.
        # The reduction logic above should handle it by discarding points.
        # However, the logic above only reduces from 3 to 2.
        # If we start with 1, go to 2, then 3, then add a 4th?
        # In 2D GJK, you never have a 4th point because the simplex is a triangle.
        # If you add a point to a triangle, you check if 0 is inside. If not, you discard a vertex.
        # My logic above handles 3->2.
        # But I didn't handle the case where I add a point to a 2-simplex to make a 3-simplex, 
        # and then in the next iteration I have 3 points.
        # So the flow is:
        # 1 point -> add -> 2 points
        # 2 points -> add -> 3 points
        # 3 points -> check 0 inside? No -> reduce to 2 points -> add -> 3 points?
        # Wait, if I reduce to 2, then add a new point, I have 3.
        # So the simplex size oscillates between 2 and 3.
        # This is fine.
        
        # But I need to make sure I don't go into an infinite loop.
        # The condition `dot(b_new, d) <= 0` should eventually trigger.
        
    # Fallback if GJK fails to converge
    # Use a simpler check
    min_dist = float('inf')
    for p1 in poly1:
        for p2 in poly2:
            d = math.hypot(p1[0]-p2[0], p1[1]-p2[1])
            if d < min_dist:
                min_dist = d
    return min_dist

def _pairwise_penalty(p1, p2):
    """
    p1, p2: (cx, cy, angle_deg)
    Returns penalty if they overlap.
    """
    c1x, c1y, a1 = p1
    c2x, c2y, a2 = p2
    
    dx = c1x - c2x
    dy = c1y - c2y
    if dx*dx + dy*dy > 4.0:
        return 0.0
        
    v1 = _hex_vertices(c1x, c1y, 1.0, a1)
    v2 = _hex_vertices(c2x, c2y, 1.0, a2)
    
    dist = _gjk_distance(v1, v2)
    
    if dist < 0.0:
        return 0.0
    elif dist < 1e-6:
        # Touching or extremely close. 
        # For packing, we want them disjoint. 
        # If they are touching, is that allowed? 
        # "Disjoint" usually means interiors do not intersect. 
        # So touching is allowed. 
        # However, for optimization stability, a small penalty for being very close might help?
        # No, if touching is allowed, penalty should be 0.
        return 0.0
    else:
        # Overlap
        return dist**2

def _outer_penalty(inner_params, outer_params):
    """
    inner_params: list of (cx, cy, angle)
    outer_params: (cx, cy, s, angle)
    Returns penalty if any inner hexagon is not fully inside the outer hexagon.
    """
    ocx, ocy, os, oang = outer_params
    
    penalty = 0.0
    outer_verts = _hex_vertices(ocx, ocy, os, oang)
    
    for ip in inner_params:
        icx, icy, iang = ip
        v_in = _hex_vertices(icx, icy, 1.0, iang)
        
        # Check if any vertex of inner is outside outer
        # Or if they intersect
        # If inner is inside outer, the distance from inner to outer (in the sense of "inner is in outer") 
        # is not directly GJK. 
        # GJK gives distance between convex sets. 
        # If one is inside the other, GJK returns 0.
        # We need to check if inner is OUTSIDE outer.
        
        # If inner is outside outer, distance > 0.
        # If inner is inside outer, distance = 0 (but they are not intersecting in the "boundary" sense).
        # Wait, GJK for "inner inside outer" returns 0.
        # How do we distinguish "inside" from "intersecting"?
        
        # Check if any vertex of inner is outside outer.
        # If a vertex is outside, it's a violation.
        
        # Let's check vertices of inner against outer.
        for pt in v_in:
            # Check if pt is inside outer
            # Use winding number or ray casting? 
            # Or use the fact that outer is convex.
            # Check if pt is on the same side of all edges as the center of outer.
            
            # This is O(6) per vertex. 6*6=36 checks.
            # Total 11*36 = 396 checks.
            # This is fast.
            
            # Let's implement point_in_hexagon
            # Rotate pt to outer's frame
            a = math.radians(oang)
            dx = pt[0] - ocx
            dy = pt[1] - ocy
            rx = dx * math.cos(-a) - dy * math.sin(-a)
            ry = dx * math.sin(-a) + dy * math.cos(-a)
            
            apothem = os * math.sqrt(3) / 2
            inside = True
            for i in range(3):
                theta_n = math.radians(30 + i * 60)
                proj = rx * math.cos(theta_n) + ry * math.sin(theta_n)
                if proj > apothem:
                    inside = False
                    break
                if proj < -apothem:
                    inside = False
                    break
            
            if not inside:
                # Calculate how far outside
                # We can use the same projection logic to find the distance
                # But for penalty, just summing the violation amount is enough.
                # Let's calculate the distance to the boundary.
                # The max projection violation is the distance to the supporting line.
                # But the distance to the hexagon boundary is the min distance to any edge.
                
                # For simplicity, let's just use a large penalty if any vertex is outside.
                # Or better, calculate the exact distance.
                
                # Let's use GJK to find the distance from pt to the outer hexagon?
                # No, pt is a point. 
                # Distance from point to convex polygon.
                
                # Let's just use the projection violation as a proxy for distance.
                # It's an upper bound on the distance.
                
                # Re-calculate projections
                max_viol = 0.0
                for i in range(3):
                    theta_n = math.radians(30 + i * 60)
                    proj = rx * math.cos(theta_n) + ry * math.sin(theta_n)
                    if proj > apothem:
                        viol = proj - apothem
                        if viol > max_viol:
                            max_viol = viol
                    if proj < -apothem:
                        viol = -apothem - proj
                        if viol > max_viol:
                            max_viol = viol
                penalty += max_viol**2
                
    return penalty

def _objective(x):
    """
    x: [cx1, cy1, a1, ..., cxN, cyN, aN, ocx, ocy, os, oang]
    """
    n_inner = N
    inner_params = []
    for i in range(n_inner):
        cx = x[i*3]
        cy = x[i*3+1]
        a = x[i*3+2]
        inner_params.append((cx, cy, a))
        
    ocx = x[n_inner*3]
    ocy = x[n_inner*3+1]
    os = x[n_inner*3+2]
    oang = x[n_inner*3+3]
    outer_params = (ocx, ocy, os, oang)
    
    pen_pair = 0.0
    for i in range(n_inner):
        for j in range(i+1, n_inner):
            pen_pair += _pairwise_penalty(inner_params[i], inner_params[j])
            
    pen_outer = _outer_penalty(inner_params, outer_params)
    
    return os + 100.0 * (pen_pair + pen_outer)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Initial guess: Lattice packing
    init_inner = []
    init_inner.append((0.0, 0.0, 0.0))
    for i in range(6):
        a = math.radians(60*i)
        x = 1.5 * math.cos(a)
        y = 1.5 * math.sin(a)
        init_inner.append((x, y, 0.0))
    # Ring 2 (5 more to make 11)
    init_inner.append((3.0, 0.0, 0.0))
    init_inner.append((-3.0, 0.0, 0.0))
    init_inner.append((0.0, 3.0, 0.0))
    init_inner.append((0.0, -3.0, 0.0))
    
    x0 = []
    for p in init_inner:
        x0.append(p[0])
        x0.append(p[1])
        x0.append(p[2])
        
    x0.extend([0.0, 0.0, 5.0, 0.0])
    
    bounds = []
    for i in range(N):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
        bounds.append((0, 360))
        
    bounds.append((-10, 10))
    bounds.append((-10, 10))
    bounds.append((3.0, 10.0))
    bounds.append((0, 360))
    
    best_x = x0
    best_val = _objective(x0)
    
    # Use Basinhopping
    # Basinhopping requires a local optimizer. L-BFGS-B is good for bounded problems.
    
    def local_minimizer(x):
        res = minimize(_objective, x, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 100, 'ftol': 1e-4, 'gtol': 1e-4})
        return res.x, res.fun, {}
    
    try:
        res = basinhopping(_objective, x0, niter=20, minimizer_kwargs={'local_minimizer': local_minimizer}, 
                           seed=42, stepsize=0.5, time_limit=120)
        
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
            
    except Exception:
        pass
        
    if time.time() > deadline:
        return _format_result(best_x)
        
    # Final polish with tighter tolerance
    try:
        res2 = minimize(_objective, best_x, method='L-BFGS-B', bounds=bounds, 
                        options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8})
        if res2.fun < best_val:
            best_val = res2.fun
            best_x = res2.x
    except Exception:
        pass
        
    return _format_result(best_x)

def _format_result(x):
    inner = []
    for i in range(N):
        cx = x[i*3]
        cy = x[i*3+1]
        a = x[i*3+2]
        inner.append([cx, cy, a])
        
    ocx = x[N*3]
    ocy = x[N*3+1]
    os = x[N*3+2]
    oang = x[N*3+3]
    
    return inner, (ocx, ocy), os, oang

if __name__ == "__main__":
    r = construct()
    print(r)
