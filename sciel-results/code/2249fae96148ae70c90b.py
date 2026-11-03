import numpy as np
import time
import sys

N = 12
TIME_LIMIT = 145.0  # Safety margin for 150s limit
START_TIME = time.time()

def _hex_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon."""
    x, y = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * (np.pi / 3)
    return np.array([
        [x + side * np.cos(a), y + side * np.sin(a)] 
        for a in angles
    ])

def _project_points_onto_poly(p, poly):
    """
    Project a set of points onto a convex polygon.
    Returns the closest point on the polygon boundary or interior for each point.
    Since we only care about containment (inside), if a point is outside,
    we project it onto the closest edge.
    """
    n = len(poly)
    closest_pts = np.zeros_like(p)
    
    for i in range(len(p)):
        pt = p[i]
        min_dist = float('inf')
        closest = None
        
        # Check each edge
        for k in range(n):
            a = poly[k]
            b = poly[(k+1)%n]
            
            # Segment closest point
            ab = b - a
            ap = pt - a
            t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12)
            t = np.clip(t, 0.0, 1.0)
            proj = a + t * ab
            d = np.linalg.norm(pt - proj)
            
            if d < min_dist:
                min_dist = d
                closest = proj
                
        closest_pts[i] = closest if closest is not None else pt
        
    return closest_pts

def _is_point_in_convex_poly(pt, poly):
    """Check if a point is inside a convex polygon."""
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i+1)%n]
        cross = (b[0]-a[0])*(pt[1]-a[1]) - (b[1]-a[1])*(pt[0]-a[0])
        if cross < -1e-9:
            return False
    return True

def _polygons_overlap(v1, v2):
    """
    Check if two convex polygons overlap using SAT.
    Returns True if they overlap, False if disjoint.
    """
    n1, n2 = len(v1), len(v2)
    
    # Check axes from both polygons
    for i in range(n1):
        edge = v1[(i+1) % n1] - v1[i]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal = normal / norm
        
        proj1 = np.array([np.dot(p, normal) for p in v1])
        proj2 = np.array([np.dot(p, normal) for p in v2])
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
            return False  # Separating axis found, no overlap
            
    return True  # No separating axis, polygons overlap

def _compute_collision_response(v1, v2, c1, c2):
    """
    Compute position correction for two overlapping convex polygons.
    Returns (delta_c1, delta_c2) to separate them.
    Uses SAT to find minimum translation vector (MTV).
    """
    n1, n2 = len(v1), len(v2)
    min_overlap = float('inf')
    min_axis = None
    
    # Check axes from both polygons
    for i in range(n1):
        edge = v1[(i+1) % n1] - v1[i]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal = normal / normal if False else normal / norm
        
        proj1 = np.array([np.dot(p, normal) for p in v1])
        proj2 = np.array([np.dot(p, normal) for p in v2])
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        overlap = min(max1, max2) - max(min1, min2)
        
        if overlap <= 1e-9:
            return np.zeros(2), np.zeros(2) # Disjoint or touching
            
        if overlap < min_overlap:
            min_overlap = overlap
            # Ensure normal points from v1 to v2
            mid1 = (min1 + max1) / 2
            mid2 = (min2 + max2) / 2
            if mid2 < mid1:
                normal = -normal
            min_axis = normal
            
    for i in range(n2):
        edge = v2[(i+1) % n2] - v2[i]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal = normal / norm
        
        proj1 = np.array([np.dot(p, normal) for p in v1])
        proj2 = np.array([np.dot(p, normal) for p in v2])
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        overlap = min(max1, max2) - max(min1, min2)
        
        if overlap <= 1e-9:
            return np.zeros(2), np.zeros(2)
            
        if overlap < min_overlap:
            min_overlap = overlap
            mid1 = (min1 + max1) / 2
            mid2 = (min2 + max2) / 2
            if mid2 < mid1:
                normal = -normal
            min_axis = normal
            
    if min_axis is None:
        return np.zeros(2), np.zeros(2)
        
    # Distribute correction equally (masses are equal)
    # Move v1 by -0.5 * overlap * axis, v2 by +0.5 * overlap * axis
    delta1 = -0.5 * min_overlap * min_axis
    delta2 = 0.5 * min_overlap * min_axis
    
    return delta1, delta2

def _simulate(centers, angles, outer_side, outer_center, outer_angle, steps=2000, dt=0.01):
    """
    Rigid body dynamics simulation to resolve overlaps and containment.
    """
    centers = centers.copy()
    # Velocities are not used in PBD, but we can use iterative projection
    # Let's do iterative projection with relaxation
    
    outer_verts = _hex_vertices(outer_center, outer_side, outer_angle)
    
    for step in range(steps):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
            
        # 1. Containment constraint: Push all inner hex vertices inside outer hex
        for i in range(N):
            inner_verts = _hex_vertices(centers[i], 1.0, angles[i])
            for k in range(6):
                p = inner_verts[k]
                if not _is_point_in_convex_poly(p, outer_verts):
                    # Project point onto outer boundary
                    closest = _project_points_onto_poly(p.reshape(1,2), outer_verts)[0]
                    # Move center so that this vertex moves to closest
                    # Since vertex = center + offset, delta_center = closest - p
                    delta = closest - p
                    centers[i] += delta * 0.5 # Relaxation factor
            
        # 2. Pairwise collision constraint
        for i in range(N):
            for j in range(i + 1, N):
                v1 = _hex_vertices(centers[i], 1.0, angles[i])
                v2 = _hex_vertices(centers[j], 1.0, angles[j])
                
                if _polygons_overlap(v1, v2):
                    d1, d2 = _compute_collision_response(v1, v2, centers[i], centers[j])
                    centers[i] += d1 * 0.5
                    centers[j] += d2 * 0.5
                    
    return centers

def _outer_hex_side(centers, angles, outer_angle=0.0):
    """Compute the minimum outer side length required to contain all hexagons."""
    max_proj = 0.0
    for k in range(6):
        angle = np.deg2rad(outer_angle + k * 60)
        normal = np.array([np.cos(angle), np.sin(angle)])
        
        for i in range(N):
            verts = _hex_vertices(centers[i], 1.0, angles[i])
            for v in verts:
                proj = np.dot(v, normal)
                if proj > max_proj:
                    max_proj = proj
                    
    return max_proj * 2.0 / np.sqrt(3.0)

def _is_valid(inner_hexes, outer_side, outer_center, outer_angle, eps=1e-9):
    """Fast validity check."""
    outer_verts = _hex_vertices(outer_center, outer_side, outer_angle)
    inner_verts = [_hex_vertices(h[0:2], 1.0, h[2]) for h in inner_hexes]
    
    # Check disjointness
    for i in range(len(inner_hexes)):
        for j in range(i + 1, len(inner_hexes)):
            if _polygons_overlap(inner_verts[i], inner_verts[j]):
                return False, "overlap"
        
        # Check containment
        for p in inner_verts[i]:
            inside = True
            for k in range(6):
                a = outer_verts[k]
                b = outer_verts[(k+1)%6]
                cross_ab = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
                if cross_ab < -eps:
                    inside = False
                    break
            if not inside:
                return False, "outside"
                
    return True, "valid"

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    np.random.seed(42)
    
    # Try different outer angles
    for outer_angle in [0.0, 30.0]:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        # Multiple random initializations
        for restart in range(30):
            if time.time() - start_time > TIME_LIMIT - 2:
                break
                
            # Random initial positions
            if restart == 0:
                # Hexagonal lattice initial
                pts = [[0,0]]
                for k in range(6):
                    angle = np.pi/3 * k
                    pts.append([np.sqrt(3)*np.cos(angle), np.sqrt(3)*np.sin(angle)])
                for k in range(6):
                    angle = np.pi/3 * k + np.pi/6
                    pts.append([2*np.sqrt(3)*np.cos(angle), 2*np.sqrt(3)*np.sin(angle)])
                    if len(pts) == 12:
                        break
                centers = np.array(pts)
            else:
                # Random positions in a disk
                r = np.sqrt(np.random.uniform(0, 16))
                theta = np.random.uniform(0, 2*np.pi)
                centers = np.array([[r*np.cos(theta), r*np.sin(theta)] for _ in range(N)])
            
            angles = np.zeros(N)
            
            # Initial side estimate
            current_side = _outer_hex_side(centers, angles, outer_angle)
            
            # Simulate with shrinking boundary
            # Start with a slightly larger boundary to allow movement, then shrink
            # Or just use the current required side and let PBD push things in
            # Actually, PBD needs a fixed boundary to push against.
            # We can use an iterative shrinking scheme:
            # 1. Estimate side S
            # 2. Run PBD with boundary S
            # 3. Check validity
            # 4. If valid, update best
            # 5. If not, maybe increase S slightly and retry? 
            # Better: Run PBD with a boundary that is the current required side.
            # If overlaps remain, it means the side is too small or PBD failed.
            
            # To make it robust, we can run PBD for a few steps, then re-evaluate side.
            # But PBD with a fixed boundary S might get stuck if S is too small.
            # Strategy: Use a large boundary initially, let them settle, then shrink?
            # No, we want to minimize S.
            
            # Let's try:
            # 1. Run PBD with boundary = current_side
            # 2. If valid, record
            # 3. If not valid, maybe the initial guess was bad.
            
            # Actually, a better strategy for PBD packing:
            # Start with a very large boundary, let them settle into a dense cluster (minimize diameter?)
            # But PBD with a large boundary doesn't minimize diameter.
            
            # Alternative: 
            # Use the "shrinking box" approach.
            # Start with a large S, run PBD to resolve overlaps.
            # Then decrease S slightly, run PBD again.
            # Repeat until S is small enough that PBD can't resolve overlaps or time runs out.
            
            S = _outer_hex_side(centers, angles, outer_angle) * 1.5 # Start 50% larger
            S_min = S * 0.5 # Don't go below 50% of initial
            
            # Shrink S gradually
            while S > S_min and time.time() - start_time < TIME_LIMIT - 2:
                # Run PBD with boundary S
                new_centers = _simulate(centers, angles, S, (0.0, 0.0), outer_angle, steps=100)
                
                # Check validity
                inner_hexes = [[new_centers[i][0], new_centers[i][1], 0.0] for i in range(N)]
                is_valid, reason = _is_valid(inner_hexes, S, (0.0, 0.0), outer_angle)
                
                if is_valid:
                    if S < best_side:
                        best_side = S
                        best_inner = inner_hexes
                        best_outer_center = (0.0, 0.0)
                        best_outer_angle = outer_angle
                        
                    # Try to shrink further
                    S *= 0.95
                    centers = new_centers
                else:
                    # If invalid, we can't shrink. Break.
                    break
                    
            # Update centers for next restart if it failed? No, restart is random.
            
    if best_inner is not None:
        return (best_inner, best_outer_center, best_side, best_outer_angle)
    
    # Fallback
    return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0
