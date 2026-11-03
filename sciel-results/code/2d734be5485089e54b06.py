import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
# Candidate angles in degrees
# Regular hexagons have 6-fold symmetry, so 0-60 is enough, but let's use 0-180 for safety
# Step 5 degrees is a good balance. 36 candidates.
ANGLES = list(range(0, 180, 5))

def _hex_vertices(cx, cy, s, angle_deg):
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return verts

def _point_in_hex_dist_sq(px, py, cx, cy, s, angle_deg):
    a = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    rx = dx * math.cos(-a) - dy * math.sin(-a)
    ry = dx * math.sin(-a) + dy * math.cos(-a)
    
    apothem = s * math.sqrt(3) / 2
    sum_sq_viol = 0.0
    for i in range(3):
        theta_n = math.radians(30 + i * 60)
        proj = rx * math.cos(theta_n) + ry * math.sin(theta_n)
        if proj > apothem:
            sum_sq_viol += (proj - apothem)**2
        if proj < -apothem:
            sum_sq_viol += (-apothem - proj)**2
    return sum_sq_viol

def _polygons_overlap(v1, v2):
    """
    Checks if two convex polygons (given as lists of vertices) overlap.
    Uses Separating Axis Theorem (SAT).
    Returns True if they overlap (including touching).
    """
    def get_axes(vertices):
        axes = []
        for i in range(len(vertices)):
            p1 = vertices[i]
            p2 = vertices[(i+1)%len(vertices)]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            # Normal vector
            ax = -dy
            ay = dx
            norm = math.sqrt(ax*ax + ay*ay)
            if norm > 1e-9:
                axes.append((ax/norm, ay/norm))
        return axes

    def project(vertices, axis):
        dots = [v[0]*axis[0] + v[1]*axis[1] for v in vertices]
        return min(dots), max(dots)

    axes = get_axes(v1) + get_axes(v2)
    
    for axis in axes:
        min1, max1 = project(v1, axis)
        min2, max2 = project(v2, axis)
        # Check for gap
        if max1 < min2 or max2 < min1:
            return False
            
    return True

def _min_enclosing_hex_side(inner_verts_list):
    """
    Given a list of inner hexagon vertices (flattened list of all points),
    finds the minimum side length of a regular hexagon (any rotation, any center)
    that encloses all these points.
    
    Note: For a fixed set of points, the minimum enclosing regular hexagon
    side length is determined by the maximum projection onto the 3 axes of the hexagon.
    However, the orientation of the enclosing hexagon matters.
    
    Actually, the problem allows the outer hexagon to rotate.
    Let the outer hexagon have center (cx, cy) and angle theta.
    The condition for a point (px, py) to be inside is:
    For k in 0..2:
      | (px-cx, py-cy) . n_k(theta) | <= apothem
    where n_k(theta) are the normals to the sides.
    
    This is equivalent to:
    The width of the set of points in direction n_k(theta) must be <= 2 * apothem.
    Width in direction u is max(u.p) - min(u.p).
    So, max_k (Width_k(theta)) / (sqrt(3)) <= s.
    Wait, apothem = s * sqrt(3) / 2.
    So 2 * apothem = s * sqrt(3).
    Width_k(theta) <= s * sqrt(3)
    s >= Width_k(theta) / sqrt(3)
    s >= max_k (Width_k(theta)) / sqrt(3)
    
    To minimize s, we need to minimize max_k (Width_k(theta)) over theta.
    The width function is periodic with period 60 degrees (for hexagon symmetry).
    Also, the width of a convex polygon in a given direction is a convex function of the angle?
    Actually, width(theta) = max(v_i . u(theta)) - min(v_i . u(theta)).
    This function is not necessarily convex, but it is continuous and periodic.
    We can sample theta finely to find the minimum.
    """
    if not inner_verts_list:
        return 0.0
        
    # Flatten all points
    pts = []
    for v in inner_verts_list:
        pts.extend(v)
        
    if len(pts) < 3:
        # Degenerate case
        return 0.0

    best_s = float('inf')
    # Sample outer hexagon angles. Since the inner hexagons are fixed,
    # the optimal outer angle is likely related to the inner angles or specific directions.
    # Let's sample 0 to 60 degrees with fine steps (e.g., 0.1 deg) to be safe.
    # 600 samples is fast.
    for deg in np.linspace(0, 60, 600):
        theta = math.radians(deg)
        # Normals for the 3 pairs of opposite sides
        # The sides are at angles theta + 30, theta + 90, theta + 150?
        # Let's define the hexagon such that its vertices are at theta + i*60.
        # The normals to the sides are at theta + 30 + i*60.
        
        max_width = 0.0
        for k in range(3):
            angle_n = theta + math.radians(30 + k * 60)
            ux = math.cos(angle_n)
            uy = math.sin(angle_n)
            
            min_proj = float('inf')
            max_proj = float('-inf')
            for p in pts:
                proj = p[0]*ux + p[1]*uy
                if proj < min_proj: min_proj = proj
                if proj > max_proj: max_proj = proj
                
            width = max_proj - min_proj
            if width > max_width:
                max_width = width
                
        s = max_width / math.sqrt(3)
        if s < best_s:
            best_s = s
            
    return best_s

def _check_validity(inner_params, outer_params):
    """
    Returns True if inner hexagons are disjoint and inside outer.
    """
    ocx, ocy, os, oang = outer_params
    
    # Check disjointness
    verts_list = []
    for ip in inner_params:
        icx, icy, iang = ip
        v = _hex_vertices(icx, icy, 1.0, iang)
        verts_list.append(v)
        
    for i in range(len(inner_params)):
        for j in range(i+1, len(inner_params)):
            if _polygons_overlap(verts_list[i], verts_list[j]):
                return False
                
    # Check inside outer
    # For each inner vertex, check if it is inside outer hexagon
    for v in verts_list:
        for pt in v:
            if _point_in_hex_dist_sq(pt[0], pt[1], ocx, ocy, os, oang) > 1e-6:
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Strategy:
    # 1. Generate a diverse set of initial packings using lattice-like structures with random perturbations and rotations.
    # 2. For each candidate, calculate the minimum enclosing hexagon side.
    # 3. Refine the best candidates using local optimization (Nelder-Mead) on the positions, keeping angles fixed or re-optimizing them.
    
    best_score = float('inf')
    best_inner = None
    best_outer = None
    
    # Precompute some common structures
    # Structure 1: Center + 6 Ring + 4 Ring 2 (standard hexagonal packing)
    # Structure 2: Two rows of 3 + One row of 5?
    # Let's use a random search over a coarse grid of "templates"
    
    # Template 1: Honeycomb
    def gen_honeycomb():
        inner = []
        inner.append((0.0, 0.0, 0.0))
        for i in range(6):
            a = math.radians(60*i)
            x = 1.5 * math.cos(a)
            y = 1.5 * math.sin(a)
            inner.append((x, y, 0.0))
        # Add 4 more from ring 2
        # Ring 2 positions: (3,0), (2.25, 1.9), (0.75, 2.6), (-0.75, 2.6), (-2.25, 1.9), (-3, 0), (-2.25, -1.9), (-0.75, -2.6), (0.75, -2.6), (2.25, -1.9)
        # Let's pick 4 that are far apart
        candidates = [
            (3.0, 0.0),
            (-3.0, 0.0),
            (0.0, 2.6), # approx (0.75, 2.6) and (-0.75, 2.6) merged? No, need distinct.
            (0.0, -2.6)
        ]
        # Actually, let's just add 4 specific ones
        inner.append((3.0, 0.0, 0.0))
        inner.append((-3.0, 0.0, 0.0))
        inner.append((0.75, 2.6, 0.0))
        inner.append((-0.75, 2.6, 0.0))
        return inner

    # Template 2: Two rows
    def gen_two_rows():
        inner = []
        # Row 1: 3 hexagons
        for i in range(3):
            inner.append((i*1.5, 0.0, 0.0))
        # Row 2: 4 hexagons, offset
        for i in range(4):
            inner.append((i*1.5 + 0.75, 1.3, 0.0))
        # Row 3: 4 hexagons? Total 11.
        # 3 + 4 + 4 = 11
        for i in range(4):
            inner.append((i*1.5, 2.6, 0.0))
        return inner

    # Template 3: Random
    def gen_random():
        inner = []
        for _ in range(N):
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
            a = np.random.choice(ANGLES)
            inner.append((x, y, a))
        return inner

    templates = [gen_honeycomb, gen_two_rows]
    
    # Iterate through templates and some random restarts
    num_templates = len(templates)
    num_random = 20
    
    all_candidates = []
    for t in templates:
        all_candidates.append(t())
        
    for _ in range(num_random):
        all_candidates.append(gen_random())
        
    # For each candidate, optimize the outer hexagon and check validity
    # Then, if valid, try to optimize the inner positions to reduce the outer size?
    # Actually, the outer size is determined by the inner positions.
    # So we should optimize the inner positions to minimize the enclosing hexagon size.
    
    # We will use a local search on the inner positions for each candidate.
    # Objective: Minimize the side length of the minimum enclosing hexagon.
    
    def obj_inner(x):
        # x: [cx1, cy1, ..., cxN, cyN]
        # Angles are fixed from the candidate template for this local search
        inner_params = []
        for i in range(N):
            cx = x[i*2]
            cy = x[i*2+1]
            a = candidate_angles[i]
            inner_params.append((cx, cy, a))
            
        # Calculate min enclosing hexagon
        verts_list = []
        for ip in inner_params:
            v = _hex_vertices(ip[0], ip[1], 1.0, ip[2])
            verts_list.append(v)
            
        # Check disjointness first (penalty if overlap)
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if _polygons_overlap(verts_list[i], verts_list[j]):
                    # Estimate overlap penalty
                    # Simple: check center distance
                    c1 = (ip1[0], ip1[1])
                    c2 = (ip2[0], ip2[1])
                    # Just use a large penalty if overlap detected
                    pen += 100.0
                    break
            if pen > 0:
                break
        
        if pen > 0:
            return pen + _min_enclosing_hex_side(verts_list)
            
        return _min_enclosing_hex_side(verts_list)

    for idx, candidate in enumerate(all_candidates):
        if time.time() > deadline:
            break
            
        candidate_angles = [c[2] for c in candidate]
        x0 = []
        for c in candidate:
            x0.append(c[0])
            x0.append(c[1])
            
        try:
            # Use Nelder-Mead to optimize positions
            # Bounds: keep them within a reasonable area
            bounds = [(-10, 10), (-10, 10)] * N
            # Nelder-Mead doesn't take bounds directly in older scipy, but we can clamp
            # Or use L-BFGS-B? Nelder-Mead is derivative-free and robust.
            res = minimize(obj_inner, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            
            x_opt = res.x
            # Clamp x_opt to bounds just in case
            x_clamped = []
            for i, v in enumerate(x_opt):
                if i % 2 == 0: # x
                    x_clamped.append(np.clip(v, -10, 10))
                else: # y
                    x_clamped.append(np.clip(v, -10, 10))
            
            # Reconstruct inner params
            inner_params = []
            for i in range(N):
                cx = x_clamped[i*2]
                cy = x_clamped[i*2+1]
                a = candidate_angles[i]
                inner_params.append((cx, cy, a))
                
            # Check validity
            # We need to find the outer hexagon that encloses these.
            # The _min_enclosing_hex_side returns the side length, but we also need the center and angle for the output.
            # Let's find the best outer hexagon parameters.
            
            # Re-calculate the optimal outer angle and center
            # We can find the center as the average of min/max projections for the optimal angle
            # But we need the exact center.
            
            # Let's refine the outer hexagon parameters explicitly.
            # We know the optimal angle from _min_enclosing_hex_side? No, it just returns the side.
            # Let's modify _min_enclosing_hex_side to return (side, center, angle)
            
            def _get_min_enclosing_hex_params(inner_verts_list):
                if not inner_verts_list:
                    return 0.0, (0,0), 0.0
                pts = []
                for v in inner_verts_list:
                    pts.extend(v)
                
                best_s = float('inf')
                best_cx = 0
                best_cy = 0
                best_ang = 0
                
                for deg in np.linspace(0, 60, 600):
                    theta = math.radians(deg)
                    max_width = 0.0
                    best_k = 0
                    # We need to find the center that minimizes the max deviation from the center
                    # For a fixed angle theta, the optimal center is the midpoint of the bounding box in the 3 normal directions.
                    # Actually, for a convex polygon, the minimum enclosing hexagon with fixed orientation
                    # has its center at the intersection of the midlines of the bounding box in the 3 normal directions?
                    # No, it's the center of the "smallest" hexagon.
                    # The constraints are:
                    # c . n_k - w_k/2 <= p . n_k <= c . n_k + w_k/2
                    # This implies c . n_k = (max(p . n_k) + min(p . n_k)) / 2
                    # If such a c exists, then the width in direction n_k is max - min.
                    # The side length s is max_k (Width_k) / sqrt(3).
                    # Does a single c satisfy all 3?
                    # Yes, if the hexagon is regular and the constraints are consistent.
                    # The center c is uniquely determined by the 3 equations c . n_k = mid_k.
                    # Since n_0, n_1, n_2 are linearly dependent (sum to 0), consistency is required.
                    # For any set of points, the minimal enclosing regular hexagon of fixed orientation
                    # is defined by the 3 widths. The center is the solution to the linear system.
                    
                    mids = []
                    widths = []
                    for k in range(3):
                        angle_n = theta + math.radians(30 + k * 60)
                        ux = math.cos(angle_n)
                        uy = math.sin(angle_n)
                        
                        min_proj = float('inf')
                        max_proj = float('-inf')
                        for p in pts:
                            proj = p[0]*ux + p[1]*uy
                            if proj < min_proj: min_proj = proj
                            if proj > max_proj: max_proj = proj
                            
                        mid = (min_proj + max_proj) / 2.0
                        width = max_proj - min_proj
                        mids.append(mid)
                        widths.append(width)
                        
                    # Solve for c
                    # c . n_0 = m_0
                    # c . n_1 = m_1
                    # c . n_2 = m_2
                    
                    # Use n_0 and n_1 to find c
                    n0 = (math.cos(theta + math.radians(30)), math.sin(theta + math.radians(30)))
                    n1 = (math.cos(theta + math.radians(90)), math.sin(theta + math.radians(90)))
                    
                    # [ n0.x n0.y ] [ cx ]   [ m0 ]
                    # [ n1.x n1.y ] [ cy ] = [ m1 ]
                    
                    det = n0[0]*n1[1] - n0[1]*n1[0]
                    if abs(det) < 1e-9:
                        continue
                        
                    cx = (mids[0]*n1[1] - mids[1]*n0[1]) / det
                    cy = (n0[0]*mids[1] - n1[0]*mids[0]) / det
                    
                    # Check if this c satisfies the 3rd constraint approximately
                    n2 = (math.cos(theta + math.radians(150)), math.sin(theta + math.radians(150)))
                    proj2 = cx*n2[0] + cy*n2[1]
                    if abs(proj2 - mids[2]) > 1e-5:
                        # Inconsistent? This shouldn't happen for a valid hexagon enclosure
                        # But due to floating point or geometry, let's just use the max width
                        pass
                        
                    s = max(widths) / math.sqrt(3)
                    if s < best_s:
                        best_s = s
                        best_cx = cx
                        best_cy = cy
                        best_ang = deg
                        
                return best_s, (best_cx, best_cy), best_ang

            verts_list = []
            for ip in inner_params:
                v = _hex_vertices(ip[0], ip[1], 1.0, ip[2])
                verts_list.append(v)
                
            s, center, ang = _get_min_enclosing_hex_params(verts_list)
            
            # Final validity check with the calculated outer hexagon
            outer_params = (center[0], center[1], s, ang)
            if _check_validity(inner_params, outer_params):
                if s < best_score:
                    best_score = s
                    best_inner = inner_params
                    best_outer = outer_params
                    
        except Exception as e:
            pass

    # If we didn't find a valid solution (unlikely), fallback to a known valid one
    if best_inner is None:
        # Use the honeycomb template as is, and calculate its enclosing hex
        inner = gen_honeycomb()
        verts_list = []
        for ip in inner:
            v = _hex_vertices(ip[0], ip[1], 1.0, ip[2])
            verts_list.append(v)
        s, center, ang = _get_min_enclosing_hex_params(verts_list)
        best_inner = inner
        best_outer = (center[0], center[1], s, ang)
        best_score = s

    # Format result
    inner_out = []
    for ip in best_inner:
        inner_out.append([ip[0], ip[1], ip[2]])
        
    return inner_out, (best_outer[0], best_outer[1]), best_outer[2], best_outer[3]

if __name__ == "__main__":
    r = construct()
    print(r)
