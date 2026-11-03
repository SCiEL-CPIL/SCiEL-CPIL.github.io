"""Hexagon packing: place N unit hexagons inside a regular hexagon, minimize outer side."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 11
DEADLINE = 115.0  # seconds, safety margin before 120s limit
START_TIME = None

def hexagon_vertices(cx, cy, angle_deg=0.0, side=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given angle."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return verts

def hexagon_vertices_array(cx, cy, angle_deg=0.0, side=1.0):
    """Vectorized version."""
    a = math.radians(angle_deg)
    thetas = a + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def point_in_hexagon(px, py, cx, cy, angle_deg, side=1.0):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    a = math.radians(angle_deg)
    # Rotate point into hexagon frame
    cos_a = math.cos(-a)
    sin_a = math.sin(-a)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    # Regular hexagon with flat-top orientation (vertex at angle 0):
    # The hexagon has vertices at angles 0, 60, 120, 180, 240, 300
    # In local frame, the hexagon is defined by |x| <= side, |y| <= side*sqrt(3)/2,
    # and |x|/side + |y|/(side*sqrt(3)/2) <= 2/sqrt(3) ... 
    # Actually for a regular hexagon with vertices at distance 'side' from center:
    # The apothem (distance from center to edge midpoint) = side * cos(30°) = side * sqrt(3)/2
    # The hexagon can be described as intersection of 6 half-planes.
    # For vertex at angle 0: the 6 edge normals are at angles 30, 90, 150, 210, 270, 330
    # Edge normal at angle phi: the constraint is x*cos(phi) + y*sin(phi) <= apothem
    apothem = side * math.sqrt(3) / 2.0
    for i in range(6):
        phi = math.radians(30 + i * 60)
        if lx * math.cos(phi) + ly * math.sin(phi) > apothem + 1e-9:
            return False
    return True

def segments_intersect(p1, p2, p3, p4):
    """Check if segment (p1,p2) intersects segment (p3,p4)."""
    def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) - (b[1]-a[1])*(c[0]-a[0])
    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    # Check collinear cases (touching)
    if d1 == 0 and _on_segment(p3, p4, p1): return True
    if d2 == 0 and _on_segment(p3, p4, p2): return True
    if d3 == 0 and _on_segment(p1, p2, p3): return True
    if d4 == 0 and _on_segment(p1, p2, p4): return True
    return False

def _on_segment(a, b, p):
    return (min(a[0], b[0]) - 1e-9 <= p[0] <= max(a[0], b[0]) + 1e-9 and
            min(a[1], b[1]) - 1e-9 <= p[1] <= max(a[1], b[1]) + 1e-9)

def hexagons_overlap(v1, v2, tol=1e-7):
    """Check if two hexagons (as arrays of 6 vertices) overlap (interiors or edges)."""
    # Check if any vertex of v1 is inside v2 or vice versa
    for i in range(6):
        if point_in_hexagon(v1[i,0], v1[i,1], v2[0,0], v2[0,1], 0, 0):
            pass  # can't easily use this
    # Better: check edge intersections and point-in-polygon
    # Check if any vertex of hex1 is strictly inside hex2
    for i in range(6):
        if _point_in_convex_polygon(v1[i], v2):
            return True
    for i in range(6):
        if _point_in_convex_polygon(v2[i], v1):
            return True
    # Check edge intersections
    for i in range(6):
        j = (i + 1) % 6
        for k in range(6):
            l = (k + 1) % 6
            if segments_intersect(tuple(v1[i]), tuple(v1[j]), tuple(v2[k]), tuple(v2[l])):
                return True
    return False

def _point_in_convex_polygon(p, poly):
    """Check if point p is inside convex polygon poly (array of 6 vertices)."""
    # For convex polygon, check if p is on same side of all edges
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i+1) % n]
        cross = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
        if cross < -1e-9:
            return False
    return True

def outer_side_length(inner_positions, angles, outer_center, outer_angle, outer_side):
    """Compute the minimum outer side length needed to contain all inner hexagons.
    Returns the actual required side length."""
    cx, cy = outer_center
    # Get all vertices of all inner hexagons
    all_verts = []
    for (ix, iy), ang in zip(inner_positions, angles):
        verts = hexagon_vertices_array(ix, iy, ang, 1.0)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # The outer hexagon has center (cx, cy), side length S, rotation outer_angle
    # A regular hexagon with side S has apothem = S * sqrt(3)/2
    # The vertices are at distance S from center
    # The hexagon is the intersection of 6 half-planes
    # Edge normals are at angles: outer_angle + 30° + k*60° for k=0..5
    # Constraint: for each normal direction, max dot product of vertices <= apothem
    
    a = math.radians(outer_angle)
    max_proj = 0.0
    for k in range(6):
        phi = a + math.radians(30 + k * 60)
        nx, ny = math.cos(phi), math.sin(phi)
        projections = (all_verts[:, 0] - cx) * nx + (all_verts[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projections))
    
    # Required apothem = max_proj, so required side = max_proj / (sqrt(3)/2) = max_proj * 2/sqrt(3)
    required_side = max_proj * 2.0 / math.sqrt(3.0)
    return required_side

def soft_outer_side(inner_positions, angles, outer_center, outer_angle, beta=50.0):
    """Smooth approximation of outer side length using log-sum-exp."""
    cx, cy = outer_center
    all_verts = []
    for (ix, iy), ang in zip(inner_positions, angles):
        verts = hexagon_vertices_array(ix, iy, ang, 1.0)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    a = math.radians(outer_angle)
    max_logsum = 0.0
    for k in range(6):
        phi = a + math.radians(30 + k * 60)
        nx, ny = math.cos(phi), math.sin(phi)
        projections = (all_verts[:, 0] - cx) * nx + (all_verts[:, 1] - cy) * ny
        # log-sum-exp of projections
        m = np.max(projections)
        lse = m + np.log(np.sum(np.exp(projections - m)))
        max_logsum = max(max_logsum, lse)
    
    # The "max" over 6 directions is approximated by log-sum-exp too
    # Actually, let's just take the max of the 6 log-sum-exps (still smooth-ish)
    # Better: compute all 6 log-sum-exps and take a soft max
    lses = []
    for k in range(6):
        phi = a + math.radians(30 + k * 60)
        nx, ny = math.cos(phi), math.sin(phi)
        projections = (all_verts[:, 0] - cx) * nx + (all_verts[:, 1] - cy) * ny
        m = np.max(projections)
        lse = m + np.log(np.sum(np.exp(projections - m)))
        lses.append(lse)
    
    lses = np.array(lses)
    m2 = np.max(lses)
    soft_max = m2 + np.log(np.sum(np.exp(lses - m2)))
    
    required_side = soft_max * 2.0 / math.sqrt(3.0)
    return required_side

def build_lattice_initialization(mode=0):
    """Build a hexagonal lattice arrangement of N hexagons."""
    if mode == 0:
        # Linear chain
        positions = []
        angles = []
        for i in range(N):
            positions.append((2.0 * i, 0.0))
            angles.append(0.0)
        return positions, angles
    elif mode == 1:
        # Hexagonal cluster (center + rings)
        positions = []
        angles = []
        positions.append((0.0, 0.0))
        angles.append(0.0)
        # Ring 1: 6 hexagons
        for i in range(6):
            theta = i * math.pi / 3.0
            # For hexagonal tiling, neighbors are at distance 2*apothem = sqrt(3)
            # Actually for unit hexagons in a hex lattice, center-to-center distance is 2*apothem = sqrt(3)
            # Wait: apothem = sqrt(3)/2, so distance between adjacent hex centers = 2 * sqrt(3)/2 = sqrt(3)
            # But that's for the flat-top orientation where they share edges
            # Let me use the standard hex grid spacing
            dist = math.sqrt(3)  # distance between adjacent hex centers
            positions.append((dist * math.cos(theta), dist * math.sin(theta)))
            angles.append(0.0)
        # Ring 2: 12 hexagons
        for i in range(12):
            theta = i * math.pi / 6.0
            dist = 2 * math.sqrt(3)
            positions.append((dist * math.cos(theta), dist * math.sin(theta)))
            angles.append(0.0)
        # Trim to N
        positions = positions[:N]
        angles = angles[:N]
        return positions, angles
    elif mode == 2:
        # Dense hexagonal packing
        positions = []
        angles = []
        # Hex grid: use axial coordinates
        # For a hex with flat top, neighbors are at:
        # (sqrt(3), 0), (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5), (-sqrt(3), 0), (-sqrt(3)/2, -1.5), (sqrt(3)/2, -1.5)
        dx = math.sqrt(3)
        dy = 1.5
        # Generate hex grid points
        for q in range(-3, 4):
            for r in range(-3, 4):
                x = dx * (q + r/2.0)
                y = dy * r
                positions.append((x, y))
                angles.append(0.0)
        # Sort by distance from origin and take N closest
        positions.sort(key=lambda p: p[0]**2 + p[1]**2)
        positions = positions[:N]
        angles = angles[:N]
        return positions, angles
    else:
        # Random
        positions = []
        angles = []
        for i in range(N):
            positions.append((np.random.uniform(-5, 5), np.random.uniform(-5, 5)))
            angles.append(np.random.uniform(0, 60))
        return positions, angles

def check_validity(inner, center, side, angle):
    """Check if the packing is valid (no overlaps, all inside outer hexagon)."""
    cx, cy = center
    # Check all inner hexagons are inside outer hexagon
    for (ix, iy, ang) in inner:
        verts = hexagon_vertices_array(ix, iy, ang, 1.0)
        for v in verts:
            if not point_in_hexagon(v[0], v[1], cx, cy, angle, side):
                return False
    
    # Check no overlaps between inner hexagons
    hex_verts = []
    for (ix, iy, ang) in inner:
        hex_verts.append(hexagon_vertices_array(ix, iy, ang, 1.0))
    
    for i in range(N):
        for j in range(i+1, N):
            if hexagons_overlap(hex_verts[i], hex_verts[j]):
                return False
    
    return True

def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    best_result = None
    best_side = float('inf')
    
    # Try multiple lattice initializations
    for mode in range(3):
        if time.time() - START_TIME > DEADLINE:
            break
        
        positions, angles = build_lattice_initialization(mode)
        
        # Center the cluster
        if positions:
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]
            cx = (min(xs) + max(xs)) / 2.0
            cy = (min(ys) + max(ys)) / 2.0
            positions = [(p[0] - cx, p[1] - cy) for p in positions]
        
        # Compute initial outer side
        outer_side = outer_side_length(positions, angles, (0.0, 0.0), 0.0, 10.0)
        
        if outer_side < best_side:
            best_side = outer_side
            best_result = ([(p[0], p[1], a) for p, a in zip(positions, angles)], (0.0, 0.0), outer_side, 0.0)
        
        # Try to optimize with L-BFGS-B
        if time.time() - START_TIME > DEADLINE - 30:
            continue
        
        # Parameters: positions (2N), angles (N), outer center (2), outer angle (1)
        # Total: 3N + 3
        x0 = []
        for p in positions:
            x0.extend([p[0], p[1]])
        x0.extend(angles)
        x0.extend([0.0, 0.0, 0.0])  # outer center x, y, angle
        
        def objective(x):
            if time.time() - START_TIME > DEADLINE - 5:
                return float('inf')
            pos = [(x[2*i], x[2*i+1]) for i in range(N)]
            angs = x[2*N:3*N]
            ocx, ocy, oang = x[3*N], x[3*N+1], x[3*N+2]
            
            # Compute outer side
            side = outer_side_length(pos, angs, (ocx, ocy), oang, 1.0)
            
            # Penalty for overlaps
            hex_verts = [hexagon_vertices_array(p[0], p[1], a, 1.0) for p, a in zip(pos, angs)]
            penalty = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    if hexagons_overlap(hex_verts[i], hex_verts[j]):
                        penalty += 100.0
            
            return side + penalty
        
        def gradient(x):
            # Use numerical gradient for simplicity (expensive but N is small)
            eps = 1e-6
            grad = np.zeros(len(x))
            f0 = objective(x)
            for i in range(len(x)):
                xp = x.copy()
                xp[i] += eps
                fp = objective(xp)
                xm = x.copy()
                xm[i] -= eps
                fm = objective(xm)
                grad[i] = (fp - fm) / (2 * eps)
            return grad
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                            options={'maxiter': 50, 'maxfun': 200, 'ftol': 1e-8})
            
            if result.success or result.fun < best_side:
                x_opt = result.x
                pos = [(x_opt[2*i], x_opt[2*i+1]) for i in range(N)]
                angs = x_opt[2*N:3*N]
                ocx, ocy, oang = x_opt[3*N], x_opt[3*N+1], x_opt[3*N+2]
                
                side = outer_side_length(pos, angs, (ocx, ocy), oang, 1.0)
                
                # Check validity
                inner = [(p[0], p[1], a) for p, a in zip(pos, angs)]
                if check_validity(inner, (ocx, ocy), side, oang):
                    if side < best_side:
                        best_side = side
                        best_result = (inner, (ocx, ocy), side, oang)
        except Exception:
            pass
    
    # If no valid result found, return the best we have or a fallback
    if best_result is None:
        # Fallback: linear arrangement
        inner = [(2.0 * i, 0.0, 0.0) for i in range(N)]
        side = outer_side_length([(2.0*i, 0.0) for i in range(N)], [0.0]*N, (0.0, 0.0), 0.0, 1.0)
        best_result = (inner, (0.0, 0.0), side, 0.0)
    
    return best_result

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
