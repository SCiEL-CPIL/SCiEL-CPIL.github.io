"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given angle."""
    angle_rad = np.radians(angle_deg)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees + rotation
    angles = angle_rad + np.radians(np.arange(6) * 60)
    vx = cx + side * np.cos(angles)
    vy = cy + side * np.sin(angles)
    return np.column_stack([vx, vy])

def _point_in_hex_dist_sq(px, py, cx, cy, side, angle_deg):
    """
    Compute squared distance from point (px,py) to the interior of hexagon.
    If point is inside, returns 0. If outside, returns squared distance to boundary.
    Uses the fact that a regular hexagon is the intersection of 6 half-planes.
    """
    angle_rad = np.radians(angle_deg)
    # Transform point to hexagon's local frame
    dx = px - cx
    dy = py - cy
    # Rotate by -angle
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # In local frame, hexagon with side 'side' has vertices at angles 0,60,...
    # The 6 sides are at apothem distance = side * sqrt(3)/2
    apothem = side * np.sqrt(3) / 2.0
    
    # The 6 normal directions (pointing inward) for a hexagon with flat top:
    # Normals at angles 30, 90, 150, 210, 270, 330 degrees
    normal_angles = np.radians(30 + np.arange(6) * 60)
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    
    # For each side, check if point violates the half-plane constraint
    # Point must satisfy: lx*nx_i + ly*ny_i <= apothem for all i
    projections = lx * nx + ly * ny
    violations = np.maximum(0, projections - apothem)
    
    if np.max(violations) < 1e-12:
        return 0.0  # Inside
    
    # Distance to boundary: for the most violated side, distance = violation
    # But for corners, we need to check distance to vertices
    max_viol = np.max(violations)
    idx = np.argmax(violations)
    
    # Check if point is near a vertex (corner region)
    # The two adjacent sides to side idx
    idx1 = (idx - 1) % 6
    idx2 = (idx + 1) % 6
    proj1 = lx * nx[idx1] + ly * ny[idx1]
    proj2 = lx * nx[idx2] + ly * ny[idx2]
    
    if proj1 > apothem or proj2 > apothem:
        # Near a vertex, compute distance to the vertex
        # Vertex between sides idx1 and idx2
        v_angle = np.radians(0 + (idx1 % 6) * 60)  # vertex angle
        # Actually vertices are at angles 0, 60, 120, ... 
        # Side idx has normal at angle 30+idx*60, so vertex is at angle idx*60
        v_angle = np.radians(idx * 60)
        vx = side * np.cos(v_angle)
        vy = side * np.sin(v_angle)
        dist_sq = (lx - vx)**2 + (ly - vy)**2
        return max(0, dist_sq)
    else:
        return max_viol**2

def _hexagon_contains_point(px, py, cx, cy, side, angle_deg):
    """Check if point is inside the hexagon."""
    return _point_in_hex_dist_sq(px, py, cx, cy, side, angle_deg) == 0.0

def _min_clearance_to_boundary(vertices, ocx, ocy, oside, oangle):
    """
    Compute the minimum distance from any vertex to the outer hexagon boundary.
    Positive means inside with clearance, negative means outside.
    Returns the minimum signed distance (negative if any vertex is outside).
    """
    min_dist = np.inf
    for v in vertices:
        px, py = v
        # Transform to outer hexagon local frame
        dx = px - ocx
        dy = py - ocy
        angle_rad = np.radians(oangle)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        
        apothem = oside * np.sqrt(3) / 2.0
        normal_angles = np.radians(30 + np.arange(6) * 60)
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        projections = lx * nx + ly * ny
        # Signed distance: apothem - max(projections) if inside, else negative
        max_proj = np.max(projections)
        dist = apothem - max_proj
        
        # But we also need to check vertices for corner regions
        # For a more accurate distance, check against vertices too
        # The actual distance to boundary is min over all edges and vertices
        # For now, use the half-plane distance as approximation
        if dist < min_dist:
            min_dist = dist
    return min_dist

def _pairwise_min_dist(v1, v2, v3, v4):
    """Minimum distance between two segments (v1,v2) and (v3,v4)."""
    # For simplicity, use minimum of endpoint-to-segment distances
    def point_to_seg_dist(p, a, b):
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        proj = a + t * ab
        return np.linalg.norm(p - proj)
    
    d = min(
        point_to_seg_dist(v1, v3, v4),
        point_to_seg_dist(v2, v3, v4),
        point_to_seg_dist(v3, v1, v2),
        point_to_seg_dist(v4, v1, v2)
    )
    return d

def _hexagons_disjoint_penalty(hexes, outer_side):
    """
    Penalty for overlapping hexagons. Returns 0 if all disjoint, positive otherwise.
    hexes: list of (cx, cy, angle_deg)
    """
    penalty = 0.0
    n = len(hexes)
    for i in range(n):
        vi = _hex_vertices(hexes[i][0], hexes[i][1], hexes[i][2])
        for j in range(i+1, n):
            vj = _hex_vertices(hexes[j][0], hexes[j][1], hexes[j][2])
            # Check if any edge of hex i intersects any edge of hex j
            # Or if any vertex of one is inside the other
            # Simple check: min distance between any pair of edges < tolerance
            min_d = np.inf
            for k in range(6):
                e1 = (vi[k], vi[(k+1)%6])
                for m in range(6):
                    e2 = (vj[m], vj[(m+1)%6])
                    d = _pairwise_min_dist(*e1, *e2)
                    if d < min_d:
                        min_d = d
            # If min distance is very small, they're touching or overlapping
            # For disjointness, we need min_d > 0 (with some margin)
            # Use a penalty based on how much they overlap
            # If min_d < 0.01, consider them overlapping
            if min_d < 0.05:
                penalty += (0.05 - min_d)
    return penalty

def _objective(params):
    """
    Objective function to minimize.
    params: [cx1, cy1, a1, cx2, cy2, a2, ..., cxN, cyN, aN, ocx, ocy, oside, oangle]
    Total: 3*N + 4
    """
    N_local = 11
    inner_params = params[:3*N_local]
    ocx, ocy, oside, oangle = params[3*N_local:]
    
    # Extract inner hexagons
    hexes = []
    for i in range(N_local):
        cx = inner_params[3*i]
        cy = inner_params[3*i+1]
        angle = inner_params[3*i+2]
        hexes.append((cx, cy, angle))
    
    # Objective: minimize outer side length + penalties for violations
    # Penalty 1: inner hexagons must be inside outer hexagon
    boundary_penalty = 0.0
    for h in hexes:
        vertices = _hex_vertices(h[0], h[1], h[2])
        for v in vertices:
            px, py = v
            # Transform to outer hex local frame
            dx = px - ocx
            dy = py - ocy
            angle_rad = np.radians(oangle)
            cos_a = np.cos(-angle_rad)
            sin_a = np.sin(-angle_rad)
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            
            apothem = oside * np.sqrt(3) / 2.0
            normal_angles = np.radians(30 + np.arange(6) * 60)
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            
            projections = lx * nx + ly * ny
            max_proj = np.max(projections)
            # If max_proj > apothem, vertex is outside
            violation = max(0, max_proj - apothem)
            boundary_penalty += violation**2
    
    # Penalty 2: inner hexagons must be disjoint
    overlap_penalty = _hexagons_disjoint_penalty(hexes, oside)
    
    # Total objective
    obj = oside + 10.0 * boundary_penalty + 10.0 * overlap_penalty
    return obj

def _initial_guess():
    """Generate initial guess for hexagonal lattice arrangement."""
    # Hexagonal lattice with spacing sqrt(3) between centers
    spacing = np.sqrt(3)
    # Arrange in rows: 3, 4, 3, 1 = 11
    # Row y-coordinates: 0, spacing*sqrt(3)/2, spacing*sqrt(3), spacing*3*sqrt(3)/2
    # Actually for hex lattice, rows are offset
    # Let me use a simpler arrangement:
    # Center at origin, 6 around it at distance sqrt(3), then 4 more
    
    hexes = []
    # Center
    hexes.append((0, 0, 0))
    # First ring: 6 hexagons at distance sqrt(3) from center
    for i in range(6):
        angle = np.radians(i * 60)
        hexes.append((spacing * np.cos(angle), spacing * np.sin(angle), 0))
    # Second ring: 4 more hexagons (at positions 0, 60, 120, 180 of second ring)
    # Second ring distance = 2 * sqrt(3)
    for i in range(4):
        angle = np.radians(i * 120)  # 0, 120, 240, 360=0... need 4 distinct
        # Actually use angles 0, 60, 180, 240 or similar
        pass
    
    # Let me use a different arrangement: rows
    # Row 0: 3 hexagons at y=0
    # Row 1: 4 hexagons at y=spacing*sqrt(3)/2, offset by spacing/2
    # Row 2: 3 hexagons at y=spacing*sqrt(3)
    # Row 3: 1 hexagon at y=spacing*3*sqrt(3)/2
    hexes = []
    # Row 0 (y=0): 3 hexagons
    for i in range(3):
        hexes.append(((i-1)*spacing, 0, 0))
    # Row 1 (y=spacing*sqrt(3)/2): 4 hexagons, offset
    y1 = spacing * np.sqrt(3) / 2
    for i in range(4):
        hexes.append(((i-1.5)*spacing, y1, 0))
    # Row 2 (y=spacing*sqrt(3)): 3 hexagons
    y2 = spacing * np.sqrt(3)
    for i in range(3):
        hexes.append(((i-1)*spacing, y2, 0))
    # Row 3 (y=spacing*3*sqrt(3)/2): 1 hexagon
    y3 = spacing * 3 * np.sqrt(3) / 2
    hexes.append((0, y3, 0))
    
    # Center the arrangement
    xs = [h[0] for h in hexes]
    ys = [h[1] for h in hexes]
    cx_mean = (min(xs) + max(xs)) / 2
    cy_mean = (min(ys) + max(ys)) / 2
    hexes = [(h[0]-cx_mean, h[1]-cy_mean, h[2]) for h in hexes]
    
    # Build params
    params = []
    for h in hexes:
        params.extend([h[0], h[1], h[2]])
    # Outer hexagon: center at origin, side length estimate, angle 0
    # Estimate outer side: the arrangement spans about 3*spacing in x and 3*spacing*sqrt(3)/2 in y
    # Outer hex side should be at least max span / sqrt(3) for width
    est_side = 4.0
    params.extend([0.0, 0.0, est_side, 0.0])
    return np.array(params)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_params = None
    best_score = np.inf
    
    # Initial guess
    x0 = _initial_guess()
    
    # Try Nelder-Mead optimization
    try:
        res = minimize(_objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 10000})
        if res.success or res.fun < best_score:
            best_params = res.x
            best_score = res.fun
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        # Not enough time for more iterations
        pass
    else:
        # Try a few more starts with perturbed initial guesses
        for trial in range(3):
            if time.time() > deadline - 20:
                break
            try:
                x0_trial = x0 + np.random.normal(0, 0.1, size=len(x0))
                # Keep outer side positive
                x0_trial[3*11+2] = max(1.0, x0_trial[3*11+2])
                res = minimize(_objective, x0_trial, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 4000})
                if res.fun < best_score:
                    best_params = res.x
                    best_score = res.fun
            except Exception:
                pass
    
    if best_params is None:
        best_params = x0
    
    # Extract results
    inner = []
    for i in range(11):
        cx = best_params[3*i]
        cy = best_params[3*i+1]
        angle = best_params[3*i+2]
        inner.append([cx, cy, angle])
    
    ocx = best_params[3*11]
    ocy = best_params[3*11+1]
    oside = max(1.0, best_params[3*11+2])
    oangle = best_params[3*11+3]
    
    # Verify: check if all inner hexagons are inside outer and disjoint
    # If not, fall back to a known valid configuration
    valid = True
    hexes = [(h[0], h[1], h[2]) for h in inner]
    for h in hexes:
        vertices = _hex_vertices(h[0], h[1], h[2])
        for v in vertices:
            px, py = v
            dx = px - ocx
            dy = py - ocy
            angle_rad = np.radians(oangle)
            cos_a = np.cos(-angle_rad)
            sin_a = np.sin(-angle_rad)
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            apothem = oside * np.sqrt(3) / 2.0
            normal_angles = np.radians(30 + np.arange(6) * 60)
            nx = np.cos(normal_angles)
            ny = np.sin(normal_angles)
            projections = lx * nx + ly * ny
            if np.max(projections) > apothem + 1e-6:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fall back to a simple valid configuration
        # Use a larger outer hexagon with lattice arrangement
        spacing = np.sqrt(3)
        hexes_fallback = []
        # Row 0: 3
        for i in range(3):
            hexes_fallback.append(((i-1)*spacing, 0, 0))
        # Row 1: 4
        y1 = spacing * np.sqrt(3) / 2
        for i in range(4):
            hexes_fallback.append(((i-1.5)*spacing, y1, 0))
        # Row 2: 3
        y2 = spacing * np.sqrt(3)
        for i in range(3):
            hexes_fallback.append(((i-1)*spacing, y2, 0))
        # Row 3: 1
        y3 = spacing * 3 * np.sqrt(3) / 2
        hexes_fallback.append((0, y3, 0))
        
        xs = [h[0] for h in hexes_fallback]
        ys = [h[1] for h in hexes_fallback]
        cx_mean = (min(xs) + max(xs)) / 2
        cy_mean = (min(ys) + max(ys)) / 2
        hexes_fallback = [(h[0]-cx_mean, h[1]-cy_mean, h[2]) for h in hexes_fallback]
        
        inner = [[h[0], h[1], h[2]] for h in hexes_fallback]
        ocx, ocy = 0.0, 0.0
        # Outer side: need to contain all vertices
        # Max extent in x: about 1.5*spacing + 1 (hex radius) = 1.5*sqrt(3)+1
        # Max extent in y: about 1.5*spacing*sqrt(3)/2 + 1 = 1.5*3/2 + 1 = 3.25
        # For a hexagon, width = side*sqrt(3), so side >= max_extent / (sqrt(3)/2) for flat direction
        # Let's just use a generous estimate
        oside = 4.5
        oangle = 0.0
    
    return inner, (ocx, ocy), oside, oangle
# EVOLVE-BLOCK-END

