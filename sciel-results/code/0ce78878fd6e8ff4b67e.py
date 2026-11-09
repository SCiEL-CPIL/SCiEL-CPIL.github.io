"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution
import time

N = 11
_DEADLINE = 120.0  # seconds wall-clock limit

def _hex_vertices(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side length, rotation."""
    a = np.radians(angle_deg)
    verts = np.array([[np.cos(a + np.pi/3 * k), np.sin(a + np.pi/3 * k)] for k in range(6)]) * side
    return verts + np.array([cx, cy])

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    # Transform point to hexagon's local frame
    a = np.radians(angle_deg)
    dx, dy = px - cx, py - cy
    # Rotate point by -a
    rx = dx * np.cos(-a) - dy * np.sin(-a)
    ry = dx * np.sin(-a) + dy * np.cos(-a)
    # For a regular hexagon with flat-top orientation (angle=0), vertices at 0,60,120,180,240,300
    # The hexagon in local coords: |x| <= side, |y| <= side*sqrt(3)/2, and |x| + |y|/sqrt(3) <= side
    # Actually for pointy-top: let me use the standard form
    # A regular hexagon with vertices at angles 0,60,...,300 from center:
    # In local frame, the hexagon is defined by 6 half-planes
    # For a hexagon with vertex at angle theta_k = k*pi/3, the supporting line is at distance side*sqrt(3)/2 from center
    # Normal direction: theta_k + pi/2 ... actually let me just check against the 6 edges
    # Simpler: the hexagon with circumradius R=side has inradius r = side*sqrt(3)/2
    # In local frame (after rotation), check: for each of 6 directions at angles pi/6 + k*pi/3 (normals to edges)
    # Actually for a hexagon with vertices at angles 0, pi/3, ..., the edge normals are at pi/6, pi/2, ...
    # Edge normal k is at angle pi/6 + k*pi/3, and the edge is at distance inradius = side*sqrt(3)/2
    inradius = side * np.sqrt(3) / 2
    for k in range(6):
        theta = np.pi / 6 + k * np.pi / 3
        if rx * np.cos(theta) + ry * np.sin(theta) > inradius + 1e-9:
            return False
    return True

def _hex_overlap_penalty(verts1, verts2):
    """Check if two hexagons (as vertex arrays) overlap. Returns penalty > 0 if overlapping."""
    # Use a simple approach: check if any vertex of one is inside the other, or edges intersect
    # For robustness, check all vertices of poly1 inside poly2 and vice versa
    def point_in_poly(px, py, verts):
        # Ray casting
        n = len(verts)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = verts[i]
            xj, yj = verts[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
    
    for vx, vy in verts1:
        if point_in_poly(vx, vy, verts2):
            return 1.0
    for vx, vy in verts2:
        if point_in_poly(vx, vy, verts1):
            return 1.0
    # Check edge intersections
    def seg_intersect(p1, p2, p3, p4):
        def ccw(a, b, c):
            return (c[1]-a[1])*(b[0]-a[0]) - (b[1]-a[1])*(c[0]-a[0])
        d1 = ccw(p3, p4, p1)
        d2 = ccw(p3, p4, p2)
        d3 = ccw(p1, p2, p3)
        d4 = ccw(p1, p2, p4)
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False
    
    n1 = len(verts1)
    n2 = len(verts2)
    for i in range(n1):
        for j in range(n2):
            if seg_intersect(verts1[i], verts1[(i+1)%n1], verts2[j], verts2[(j+1)%n2]):
                return 1.0
    return 0.0

def _objective(params):
    """Objective: outer side length + penalty for violations."""
    # Parse parameters:
    # [outer_cx, outer_cy, outer_side, outer_angle, inner_1_x, inner_1_y, inner_1_angle, ...]
    ocx, ocy, oside, oangle = params[0], params[1], params[2], params[3]
    
    # Ensure reasonable bounds
    oside = max(oside, 1.0)
    
    inner_hexes = []
    for i in range(N):
        ix = params[4 + i*3]
        iy = params[5 + i*3]
        iang = params[6 + i*3]
        inner_hexes.append((ix, iy, iang))
    
    outer_verts = _hex_vertices(ocx, ocy, oside, oangle)
    
    penalty = 0.0
    
    # Check containment: all inner hex vertices must be inside outer hex
    for ix, iy, iang in inner_hexes:
        inner_verts = _hex_vertices(ix, iy, 1.0, iang)
        for vx, vy in inner_verts:
            if not _point_in_hex(vx, vy, ocx, ocy, oside, oangle):
                # Compute distance outside
                # Simple penalty: how far outside
                a = np.radians(oangle)
                dx, dy = vx - ocx, vy - ocy
                rx = dx * np.cos(-a) - dy * np.sin(-a)
                ry = dx * np.sin(-a) + dy * np.cos(-a)
                inradius = oside * np.sqrt(3) / 2
                for k in range(6):
                    theta = np.pi / 6 + k * np.pi / 3
                    dist = rx * np.cos(theta) + ry * np.sin(theta)
                    if dist > inradius:
                        penalty += (dist - inradius) ** 2
    
    # Check pairwise non-overlap
    for i in range(N):
        for j in range(i+1, N):
            ix1, iy1, ia1 = inner_hexes[i]
            ix2, iy2, ia2 = inner_hexes[j]
            v1 = _hex_vertices(ix1, iy1, 1.0, ia1)
            v2 = _hex_vertices(ix2, iy2, 1.0, ia2)
            penalty += _hex_overlap_penalty(v1, v2)
    
    return oside + 10.0 * penalty

def construct():
    t_start = time.time()
    
    # Initial guess: hexagonal packing
    # For N=11, use a hexagonal arrangement: center + 6 around + 4 more
    # Hexagonal lattice spacing for unit hexagons: 2*sqrt(3) for flat-top, or 2*sqrt(3) for pointy
    # Actually, two unit hexagons side by side (flat-top) have centers 2*sqrt(3) apart horizontally? 
    # No: for flat-top hexagons, horizontal spacing is 2*sqrt(3)*side? Let me think.
    # A flat-top hex with side 1: width = 2, height = sqrt(3). 
    # Actually vertices at (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5,-sqrt(3)/2), (0.5,-sqrt(3)/2)
    # So width = 2, height = sqrt(3)
    # For tiling: horizontal spacing = 1.5 (3/2), vertical spacing = sqrt(3), offset rows by 0.75?
    # Actually for flat-top hex tiling: dx = 3/2, dy = sqrt(3), odd rows offset by 3/4 in x
    # Let me use a simpler initial: place them in a grid-like pattern
    
    # Hexagonal packing positions for 11 hexagons:
    # Row 0: 3 hexagons
    # Row 1: 3 hexagons (offset)
    # Row 2: 2 hexagons
    # Row 3: 2 hexagons (offset)
    # Or just use a compact hex arrangement: 1 + 6 + 4 = 11
    # Center at (0,0), ring of 6 at distance 2*sqrt(3) (for flat-top, touching)
    # Actually for flat-top unit hexagons, two adjacent in same row: center distance = 3 (since width=2, gap=1? No)
    # Two flat-top hexes side by side: they share an edge. The shared edge is vertical, length sqrt(3).
    # Center-to-center = 2 * inradius = 2 * sqrt(3)/2 = sqrt(3)? No.
    # Inradius = sqrt(3)/2. Two hexes sharing a vertical edge: centers are 2*inradius = sqrt(3) apart horizontally.
    # Wait no. For flat-top hex, the left and right vertices are at x=±1. The vertical edges are at x=±0.5.
    # Two hexes sharing the right edge of left hex and left edge of right hex: 
    # Left hex right edge at x=0.5, right hex left edge at x=0.5+cx2-1... 
    # If right hex center is at (d, 0), its left edge is at d-0.5. For sharing: d-0.5 = 0.5, so d=1.
    # Hmm, that gives center distance 1? That can't be right for unit hexagons.
    # Let me reconsider. Flat-top hex with side 1: vertices at angles 0,60,120,180,240,300 degrees from center.
    # Vertex at 0°: (1, 0). Vertex at 60°: (0.5, sqrt(3)/2). 
    # The edge between (1,0) and (0.5, sqrt(3)/2) has length 1.
    # The rightmost point is at x=1. The "width" is 2.
    # Two such hexes placed side by side (sharing the edge from (1,0) to (0.5,sqrt(3)/2) and its mirror):
    # Actually they share a vertical edge? No, the edges are slanted.
    # For flat-top hex tiling, adjacent hexes in the same row are offset by (3/2, 0)? 
    # No wait. Let me just use the known fact: for regular hexagonal tiling with unit side,
    # the lattice vectors are (3/2, sqrt(3)/2) and (0, sqrt(3)) for pointy-top,
    # or equivalently for flat-top: (sqrt(3), 1/2) and (sqrt(3), -1/2)? 
    # I'll just use a reasonable initial guess and let the optimizer figure it out.
    
    # Simple initial: place hexagons in a hexagonal pattern
    # Using flat-top orientation (angle=0)
    # Spacing: for two flat-top hexes to be tangent, center distance = 2*inradius = sqrt(3) ~ 1.732
    # Actually no. Two flat-top hexes sharing an edge: the edge is between two adjacent vertices.
    # The perpendicular distance from center to edge = inradius = sqrt(3)/2.
    # Two hexes sharing an edge have centers 2*inradius = sqrt(3) apart, perpendicular to the shared edge.
    # For flat-top, the right edge goes from (0.5, sqrt(3)/2) to (0.5, -sqrt(3)/2). This is vertical at x=0.5.
    # So two hexes sharing this edge: left center at (0,0), right center at (sqrt(3), 0)? 
    # No, the right edge of left hex is at x=0.5 (from y=-sqrt(3)/2 to y=sqrt(3)/2).
    # The left edge of right hex (centered at (d,0)) is at x=d-0.5.
    # For sharing: d-0.5 = 0.5 => d=1. But then they'd overlap!
    # I think I'm confusing myself. Let me just use a generous initial spacing.
    
    # Use spacing of 2*sqrt(3) ≈ 3.464 between centers in a hex pattern (this is definitely non-overlapping)
    # Actually for safety, use spacing 4.0 as in the seed, then let optimizer compress.
    
    positions = []
    # Hexagonal arrangement: center + 2 rings
    # Ring 0: 1 at center
    positions.append((0, 0))
    # Ring 1: 6 at distance d
    d1 = 4.0
    for k in range(6):
        angle = k * np.pi / 3
        positions.append((d1 * np.cos(angle), d1 * np.sin(angle)))
    # Ring 2: add 4 more
    d2 = 8.0
    for k in range(4):
        angle = (k * np.pi / 3) + np.pi / 6
        positions.append((d2 * np.cos(angle), d2 * np.sin(angle)))
    
    # Build initial params
    init_params = [0.0, 0.0, 20.0, 0.0]
    for px, py in positions:
        init_params.extend([px, py, 0.0])
    
    # Define bounds
    bounds = [(-15, 15), (-15, 15), (3.0, 30.0), (0, 60)]
    for i in range(N):
        bounds.append((-15, 15))  # x
        bounds.append((-15, 15))  # y
        bounds.append((0, 60))    # angle
    
    # Run differential evolution with time budget
    t_remaining = _DEADLINE - (time.time() - t_start) - 5.0  # 5s safety margin
    if t_remaining <= 0:
        # Just return initial
        inner = [[init_params[4+i*3], init_params[5+i*3], init_params[6+i*3]] for i in range(N)]
        return inner, (init_params[0], init_params[1]), init_params[2], init_params[3]
    
    # Use differential evolution with maxiter scaled by time
    # DE is slow; let's try a limited number of iterations
    max_iters = 50
    popsize = 15
    
    try:
        result = differential_evolution(
            _objective, bounds,
            maxiter=max_iters,
            popsize=popsize,
            tol=1e-6,
            mutation=(0.3, 1.0),
            recombination=0.7,
            seed=42,
            polish=True,
            updating='immediate',
            callback=lambda x, convergence: (time.time() - t_start) > _DEADLINE - 10
        )
        
        best_params = result.x
        best_score = result.fun
    except Exception:
        best_params = np.array(init_params)
        best_score = _objective(best_params)
    
    # Extract results
    ocx, ocy, oside, oangle = best_params[0], best_params[1], best_params[2], best_params[3]
    inner = []
    for i in range(N):
        ix = best_params[4 + i*3]
        iy = best_params[5 + i*3]
        iang = best_params[6 + i*3]
        inner.append([float(ix), float(iy), float(iang)])
    
    return inner, (float(ocx), float(ocy)), float(oside), float(oangle)
# EVOLVE-BLOCK-END
