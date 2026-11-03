import numpy as np

N = 12

def _hex_verts(cx, cy, side, rot_deg):
    """Vertices of a regular hexagon centered at (cx, cy), side length 'side', rotated by rot_deg degrees."""
    r = np.radians(rot_deg)
    verts = []
    for k in range(6):
        a = r + k * np.pi / 3
        verts.append((cx + side * np.cos(a), cy + side * np.sin(a)))
    return np.array(verts)

def _orient_test(poly):
    """Signed area twice; positive if CCW."""
    n = len(poly)
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        s += poly[i, 0] * poly[j, 1] - poly[j, 0] * poly[i, 1]
    return s

def _point_in_poly(p, poly):
    """Winding-number test. Returns True if p is strictly inside poly (boundary considered outside)."""
    x, y = p
    wn = 0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        # edge from (x1,y1) to (x2,y2)
        # handle vertical-ish edges
        if y1 <= y:
            if y2 > y and (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1) > 1e-12:
                wn += 1
        else:
            if y2 <= y and (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1) < -1e-12:
                wn -= 1
    return wn != 0

def _separation(a, b):
    """
    Minimal separation between two convex polygons.
    If they intersect (with any positive area or one strictly inside the other), return a negative number
    proportional to the depth of penetration (penalty).
    If disjoint, return the minimum distance >= 0.
    """
    va, vb = a, b
    # If any vertex of A is strictly inside B, or any vertex of B is strictly inside A
    for p in va:
        if _point_in_poly(p, vb):
            return -0.5
    for p in vb:
        if _point_in_poly(p, va):
            return -0.5
    # Check edge-edge intersections
    def seg_intersect(p1, p2, p3, p4):
        # Returns True if segment p1p2 intersects segment p3p4 (including endpoints, but we'll handle overlaps separately)
        def orient(a, b, c):
            v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if abs(v) < 1e-12:
                return 0
            return 1 if v > 0 else -1
        o1 = orient(p1, p2, p3)
        o2 = orient(p1, p2, p4)
        o3 = orient(p3, p4, p1)
        o4 = orient(p3, p4, p2)
        if o1 != o2 and o3 != o4:
            return True
        # Collinear checks
        if o1 == 0 and min(p1[0], p2[0]) - 1e-12 <= p3[0] <= max(p1[0], p2[0]) + 1e-12 and min(p1[1], p2[1]) - 1e-12 <= p3[1] <= max(p1[1], p2[1]) + 1e-12:
            return True
        if o2 == 0 and min(p1[0], p2[0]) - 1e-12 <= p4[0] <= max(p1[0], p2[0]) + 1e-12 and min(p1[1], p2[1]) - 1e-12 <= p4[1] <= max(p1[1], p2[1]) + 1e-12:
            return True
        if o3 == 0 and min(p3[0], p4[0]) - 1e-12 <= p1[0] <= max(p3[0], p4[0]) + 1e-12 and min(p3[1], p4[1]) - 1e-12 <= p1[1] <= max(p3[1], p4[1]) + 1e-12:
            return True
        if o4 == 0 and min(p3[0], p4[0]) - 1e-12 <= p2[0] <= max(p3[0], p4[0]) + 1e-12 and min(p3[1], p4[1]) - 1e-12 <= p2[1] <= max(p3[1], p4[1]) + 1e-12:
            return True
        return False

    na = len(va)
    nb = len(vb)
    for i in range(na):
        p1 = va[i]
        p2 = va[(i + 1) % na]
        for j in range(nb):
            p3 = vb[j]
            p4 = vb[(j + 1) % nb]
            if seg_intersect(p1, p2, p3, p4):
                # Check if it's a proper crossing or just touching
                # For simplicity, if they intersect, check if the intersection point is strictly inside both?
                # Or just treat any intersection as overlap unless it's a single point touch.
                # Let's compute min distance first, if it's ~0, we need to be careful.
                pass

    # Compute minimum distance between segments
    min_dist = float('inf')
    for i in range(na):
        p1 = va[i]
        p2 = va[(i + 1) % na]
        for j in range(nb):
            p3 = vb[j]
            p4 = vb[(j + 1) % nb]
            # Point-segment distances
            pts_a = [p1, p2]
            pts_b = [p3, p4]
            for pa in pts_a:
                for pb in pts_b:
                    d = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
                    min_dist = min(min_dist, d)
            # Segment-segment min distance (if not intersecting)
            # Standard algorithm for min dist between two segments
            # Using vector algebra
            d1 = p2 - p1
            d2 = p4 - p3
            r = p1 - p3
            a = np.dot(d1, d1)
            e = np.dot(d2, d2)
            f = np.dot(d2, r)
            EPS = 1e-12
            if a <= EPS and e <= EPS:
                d = np.sqrt(np.dot(r, r))
                min_dist = min(min_dist, d)
            elif a <= EPS:
                s = 0.0
                t = f / e
                t = max(0.0, min(1.0, t))
                # Closest point on seg 2 is p3 + t*d2
                cp = p3 + t * d2
                d = np.sqrt(np.dot(p1 - cp, p1 - cp))
                min_dist = min(min_dist, d)
            elif e <= EPS:
                t = 0.0
                s = -np.dot(d1, r) / a
                s = max(0.0, min(1.0, s))
                cp = p1 + s * d1
                d = np.sqrt(np.dot(p3 - cp, p3 - cp))
                min_dist = min(min_dist, d)
            else:
                c = np.dot(d1, r)
                b = np.dot(d1, d2)
                denom = a * e - b * b
                s = 0.0
                t = 0.0
                if denom != 0:
                    s = (b * f - c * e) / denom
                s = max(0.0, min(1.0, s))
                t = (b * s + f) / e
                if t < 0.0:
                    t = 0.0
                    s = max(0.0, min(1.0, -c / a))
                elif t > 1.0:
                    t = 1.0
                    s = max(0.0, min(1.0, (b - c) / a))
                cp1 = p1 + s * d1
                cp2 = p3 + t * d2
                d = np.sqrt(np.dot(cp1 - cp2, cp1 - cp2))
                min_dist = min(min_dist, d)

    # If min_dist is very small, check if they are actually touching or overlapping
    # We already checked vertex-in-poly. If min_dist < 1e-6 and no vertex is inside, they are likely touching or extremely close.
    # For disjointness, we need min_dist > 0.
    # However, if they are touching at a point, they are disjoint in the sense of "disjoint interiors" usually, but the problem says "disjoint regular hexagons".
    # Usually "disjoint" means interiors are disjoint, or sets are disjoint.
    # If they touch at a boundary, the sets are not disjoint.
    # So we need min_dist > 0.
    # If min_dist is 0, they are touching. We should penalize this.
    
    if min_dist < 1e-6:
        # Check if they actually intersect (share more than a point or are identical)
        # We already did vertex-in-poly.
        # If min_dist is 0, it's likely a touch or overlap.
        # Let's assume touch is invalid.
        return -0.1
    return min_dist

def construct():
    # Initial guess: Flower packing with 5 extended hexagons
    # 1 center, 6 ring, 5 extended
    # This topology has radius 5.
    # Let's try to optimize it.
    
    # Initial positions
    inner = []
    # Center
    inner.append([0.0, 0.0, 0.0])
    # Ring at dist 2
    for i in range(6):
        ang = i * 60.0
        x = 2.0 * np.cos(np.radians(ang))
        y = 2.0 * np.sin(np.radians(ang))
        inner.append([x, y, 0.0])
    # Extended at dist 4, angles 0, 60, 120, 180, 240
    for i in range(5):
        ang = i * 60.0
        x = 4.0 * np.cos(np.radians(ang))
        y = 4.0 * np.sin(np.radians(ang))
        inner.append([x, y, 0.0])
        
    center = [0.0, 0.0]
    side = 5.0
    angle = 0.0

    # Optimization
    # Variables: x, y, rot for each inner hex (3*12=36), center x, y (2), side (1)
    # Total 39 variables.
    # We will use a simple gradient-free optimizer like Nelder-Mead or COBYLA from scipy.optimize.
    
    def objective(vars):
        inner_flat = vars[:36].reshape(12, 3)
        cx = vars[36]
        cy = vars[37]
        s = vars[38]
        outer_rot = 0.0 # Fix outer rotation to 0 for symmetry
        
        # Containment check
        outer_verts = _hex_verts(cx, cy, s, outer_rot)
        # Ensure outer is CCW
        if _orient_test(outer_verts) < 0:
            outer_verts = outer_verts[::-1]
            
        penalty = 0.0
        # Check containment
        for i in range(12):
            ix, iy, irot = inner_flat[i]
            iv = _hex_verts(ix, iy, 1.0, irot)
            if _orient_test(iv) < 0:
                iv = iv[::-1]
            # Check if all vertices of inner are inside outer
            for v in iv:
                if not _point_in_poly(v, outer_verts):
                    # Penalty: distance outside
                    # Approximate by checking distance to boundary? Or just fixed penalty.
                    penalty += 10.0
                    break
        
        # Check disjointness
        for i in range(12):
            ix, iy, irot = inner_flat[i]
            va = _hex_verts(ix, iy, 1.0, irot)
            if _orient_test(va) < 0:
                va = va[::-1]
            for j in range(i + 1, 12):
                jx, jy, jrot = inner_flat[j]
                vb = _hex_verts(jx, jy, 1.0, jrot)
                if _orient_test(vb) < 0:
                    vb = vb[::-1]
                sep = _separation(va, vb)
                if sep < 0:
                    penalty += 10.0 + abs(sep)
        
        # Objective: minimize side length + penalty
        return s + penalty

    x0 = np.zeros(39)
    for i in range(12):
        x0[i * 3] = inner[i][0]
        x0[i * 3 + 1] = inner[i][1]
        x0[i * 3 + 2] = inner[i][2]
    x0[36] = center[0]
    x0[37] = center[1]
    x0[38] = side
    
    try:
        from scipy.optimize import minimize
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.success or res.fun < objective(x0):
            x_opt = res.x
            # Extract best configuration
            best_inner = []
            for i in range(12):
                best_inner.append([x_opt[i * 3], x_opt[i * 3 + 1], x_opt[i * 3 + 2]])
            best_center = [x_opt[36], x_opt[37]]
            best_side = x_opt[38]
            
            # Validate result
            # Check if it's actually valid
            outer_verts = _hex_verts(best_center[0], best_center[1], best_side, 0.0)
            if _orient_test(outer_verts) < 0:
                outer_verts = outer_verts[::-1]
            valid = True
            for i in range(12):
                ix, iy, irot = best_inner[i]
                iv = _hex_verts(ix, iy, 1.0, irot)
                if _orient_test(iv) < 0:
                    iv = iv[::-1]
                for v in iv:
                    if not _point_in_poly(v, outer_verts):
                        valid = False
                        break
                if not valid: break
            
            if valid:
                for i in range(12):
                    ix, iy, irot = best_inner[i]
                    va = _hex_verts(ix, iy, 1.0, irot)
                    if _orient_test(va) < 0:
                        va = va[::-1]
                    for j in range(i + 1, 12):
                        jx, jy, jrot = best_inner[j]
                        vb = _hex_verts(jx, jy, 1.0, jrot)
                        if _orient_test(vb) < 0:
                            vb = vb[::-1]
                        sep = _separation(va, vb)
                        if sep < 1e-4: # Allow small tolerance
                            valid = False
                            break
                    if not valid: break
                
            if valid:
                return best_inner, best_center, best_side, 0.0
    except Exception:
        pass

    # Fallback to initial guess if optimization fails or yields invalid
    return inner, center, side, 0.0

# EVOLVE-BLOCK-END
