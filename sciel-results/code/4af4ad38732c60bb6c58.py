"""Program for packing 11 unit regular hexagons inside a regular hexagon."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, angle_deg):
    """Return 6 vertices of a unit regular hexagon."""
    a = math.radians(angle_deg)
    pts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        pts.append((cx + math.cos(theta), cy + math.sin(theta)))
    return np.array(pts)

def _signed_dist_to_hex_edge(px, py, edge, outward_normal):
    """Signed distance from point to line containing edge. Positive means outside."""
    # edge: (x1,y1,x2,y2), outward_normal: (nx,ny) unit normal pointing outward
    ex, ey = edge
    return (px - ex) * outward_normal[0] + (py - ey) * outward_normal[1]

def _point_in_hexagon(px, py, cx, cy, R, angle_deg):
    """Check if point (px,py) is inside regular hexagon centered at (cx,cy) with side R, rotated angle_deg."""
    # Transform to local coordinates
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    dx = px - cx
    dy = py - cy
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    # In local coords, hexagon vertices at angles 0,60,...,300 degrees
    # The hexagon is the intersection of 6 half-planes.
    # For a regular hexagon with circumradius R (side length R), 
    # the inradius (apothem) is R * cos(pi/6) = R * sqrt(3)/2
    # The constraints are: |lx| <= R * cos(pi/6), and for the 60-degree tilted axes...
    # Actually, simpler: the hexagon with vertices at angle k*pi/3 has apothem R*cos(pi/6) in the direction of each edge normal.
    # Edge normals are at angles 30, 90, 150, 210, 270, 330 degrees.
    apothem = R * math.cos(math.pi / 6)
    for i in range(6):
        theta = math.pi / 6 + i * math.pi / 3.0
        nx = math.cos(theta)
        ny = math.sin(theta)
        if lx * nx + ly * ny > apothem + 1e-9:
            return False
    return True

def _hexagon_overlap(i, j, centers, angles, R=1.0):
    """Check if two unit hexagons overlap. Returns 0 if no overlap, else estimate of penetration."""
    cx1, cy1 = centers[i]
    cx2, cy2 = centers[j]
    a1 = angles[i]
    a2 = angles[j]
    
    # Quick reject: distance between centers
    dx = cx2 - cx1
    dy = cy2 - cy1
    dist = math.sqrt(dx*dx + dy*dy)
    if dist > 2.0:  # Max possible distance for two unit hexagons to not overlap is 2.0 (touching)
        return 0.0
    
    # Check vertices of hex1 against hex2 and vice versa
    v1 = _hex_vertices(cx1, cy1, a1)
    v2 = _hex_vertices(cx2, cy2, a2)
    
    penetration = 0.0
    
    for p in v1:
        if _point_in_hexagon(p[0], p[1], cx2, cy2, R, a2):
            penetration = 1.0
    
    for p in v2:
        if _point_in_hexagon(p[0], p[1], cx1, cy1, R, a1):
            penetration = 1.0
    
    # Check edge-edge intersections
    for i1 in range(6):
        p1 = v1[i1]
        p2 = v1[(i1+1)%6]
        for i2 in range(6):
            q1 = v2[i2]
            q2 = v2[(i2+1)%6]
            # Check if segments p1-p2 and q1-q2 intersect
            if _segments_intersect(p1, p2, q1, q2):
                penetration = 1.0
    
    return penetration

def _segments_intersect(p1, p2, q1, q2):
    """Check if line segments p1-p2 and q1-q2 intersect."""
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    
    d1 = cross(q1, q2, p1)
    d2 = cross(q1, q2, p2)
    d3 = cross(p1, p2, q1)
    d4 = cross(p1, p2, q2)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def _min_side_length(centers, angles):
    """Compute the minimum side length of a regular hexagon (centered at origin, unrotated for simplicity 
    but we should consider rotation) that contains all hexagons."""
    # For a regular hexagon with side length S centered at (0,0) with flat sides at angles 0,60,120,180,240,300,
    # a point (x,y) is inside if for all edge normals n_k at angles 30+60k, x*n_kx + y*n_ky <= S*cos(30)
    # Actually, let's just find the maximum "hexagonal radius" needed.
    
    # The constraint for a regular hexagon with circumradius R_out (side = R_out) centered at origin, 
    # rotated by angle theta_out, is:
    # For each vertex of each inner hexagon, it must be inside the outer hexagon.
    # To minimize S (side length = circumradius), we can try several outer rotations.
    
    min_S = float('inf')
    for theta_out_deg in [0, 30, 60, 90, 120, 150]:
        theta_out = math.radians(theta_out_deg)
        max_needed = 0.0
        for i in range(len(centers)):
            cx, cy = centers[i]
            ca = math.cos(theta_out)
            sa = math.sin(theta_out)
            # Transform each vertex of inner hexagon to outer frame
            v = _hex_vertices(cx, cy, angles[i])
            for vx, vy in v:
                # Rotate by -theta_out to align outer hexagon
                lx = ca * vx + sa * vy
                ly = -sa * vx + ca * vy
                # The constraint for a hexagon with circumradius S is:
                # max over the 6 directions of (lx * cos(phi_k) + ly * sin(phi_k)) where phi_k = 30+60k
                for k in range(6):
                    phi = math.pi/6 + k * math.pi/3
                    val = lx * math.cos(phi) + ly * math.sin(phi)
                    if val > max_needed:
                        max_needed = val
        # The side length S must satisfy S * cos(pi/6) >= max_needed
        # So S >= max_needed / cos(pi/6)
        S_needed = max_needed / math.cos(math.pi/6)
        if S_needed < min_S:
            min_S = S_needed
    return min_S

def _objective(params):
    """Objective function for optimization."""
    n = len(params) // 3
    centers = []
    angles = []
    for i in range(n):
        x = params[3*i]
        y = params[3*i+1]
        ang = params[3*i+2]
        centers.append((x, y))
        angles.append(ang)
    
    # Compute overlap penalty
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            if _hexagon_overlap(i, j, centers, angles):
                penalty += 10.0  # Large penalty for overlap
    
    # Compute minimum outer side length
    S = _min_side_length(centers, angles)
    
    # We want to minimize S, but with a hard constraint on overlap
    # Use a soft constraint approach: minimize S + large_penalty * overlap
    return S + penalty

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin
    
    # Initial configuration: hexagonal packing
    # For N=11, we can use a central hexagon surrounded by 6, and 4 more
    # Let's start with a reasonable initial guess
    centers0 = []
    angles0 = []
    
    # Central hexagon
    centers0.append((0.0, 0.0))
    angles0.append(0.0)
    
    # 6 surrounding hexagons
    for i in range(6):
        theta = i * math.pi / 3
        x = 2.0 * math.cos(theta)
        y = 2.0 * math.sin(theta)
        centers0.append((x, y))
        angles0.append(0.0)
    
    # 4 more hexagons - place them at second ring positions
    # Second ring positions are at distance ~3.464 from center
    # Let's use 4 of the 12 positions
    for i in range(4):
        theta = (i * 2 + 0.5) * math.pi / 6  # Offset to avoid overlap
        # Actually, second ring in hexagonal packing has 12 positions at distance 2*sqrt(3) from center
        # But we need to be careful about overlap
        r = 2.0 * math.sqrt(3)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        centers0.append((x, y))
        angles0.append(0.0)
    
    # Flatten to parameter vector
    params0 = []
    for i in range(N):
        params0.append(centers0[i][0])
        params0.append(centers0[i][1])
        params0.append(angles0[i])
    
    best_params = params0[:]
    best_S = _min_side_length(centers0, angles0)
    best_overlap = 0
    for i in range(N):
        for j in range(i+1, N):
            if _hexagon_overlap(i, j, centers0, angles0):
                best_overlap += 1
    
    # Try optimization with multiple restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Perturb initial guess
        if restart == 0:
            params = params0[:]
        else:
            params = [p + np.random.normal(0, 1.0) for p in params0]
        
        try:
            result = minimize(_objective, params, method='Nelder-Mead', 
                             options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if result.fun < _objective(best_params):
                # Check if this is valid (no overlap)
                n = N
                centers = [(result.x[3*i], result.x[3*i+1]) for i in range(n)]
                angles = [result.x[3*i+2] for i in range(n)]
                
                has_overlap = False
                for i in range(n):
                    for j in range(i+1, n):
                        if _hexagon_overlap(i, j, centers, angles):
                            has_overlap = True
                            break
                    if has_overlap:
                        break
                
                if not has_overlap:
                    S = _min_side_length(centers, angles)
                    if S < best_S:
                        best_S = S
                        best_params = result.x[:]
        
        except Exception:
            continue
    
    # Extract best result
    n = N
    inner = []
    centers = []
    angles = []
    for i in range(n):
        x = best_params[3*i]
        y = best_params[3*i+1]
        ang_deg = math.degrees(best_params[3*i+2])
        inner.append([x, y, ang_deg])
        centers.append((x, y))
        angles.append(best_params[3*i+2])
    
    # Compute outer hexagon parameters
    S = _min_side_length(centers, angles)
    
    # Find the best outer rotation (already done in _min_side_length, but we need to return it)
    # Let's recompute to get the best rotation
    best_theta_out = 0.0
    best_S2 = float('inf')
    for theta_out_deg in [0, 30, 60, 90, 120, 150]:
        theta_out = math.radians(theta_out_deg)
        max_needed = 0.0
        for i in range(n):
            cx, cy = centers[i]
            ca = math.cos(theta_out)
            sa = math.sin(theta_out)
            v = _hex_vertices(cx, cy, angles[i])
            for vx, vy in v:
                lx = ca * vx + sa * vy
                ly = -sa * vx + ca * vy
                for k in range(6):
                    phi = math.pi/6 + k * math.pi/3
                    val = lx * math.cos(phi) + ly * math.sin(phi)
                    if val > max_needed:
                        max_needed = val
        S_needed = max_needed / math.cos(math.pi/6)
        if S_needed < best_S2:
            best_S2 = S_needed
            best_theta_out = theta_out_deg
    
    return inner, (0.0, 0.0), best_S2, best_theta_out

# EVOLVE-BLOCK-END
