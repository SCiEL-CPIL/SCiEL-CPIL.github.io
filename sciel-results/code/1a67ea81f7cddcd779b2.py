"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def _hexagon_vertices(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon with center (cx,cy), side length, rotation."""
    angle_rad = np.radians(angle_deg)
    # Flat-top hexagon: vertices at angles 0, 60, 120, ...
    angles = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    # Transform to hexagon local coords
    angle_rad = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate back
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    # For a regular hexagon with side s, the condition is:
    # |lx| <= s and |ly| <= s * sqrt(3) and |lx*sqrt(3) + ly| <= 2*s*sqrt(3)/2 ... 
    # Actually for pointy-top (vertices at 30, 90, 150...): 
    # Let's use the standard: for flat-top hex with circumradius R=side:
    # vertices at 0,60,120,180,240,300 degrees
    # The hexagon is the intersection of 6 half-planes
    # Normal directions at 30, 90, 150, 210, 270, 330 degrees (for flat-top)
    # For our hexagon with angle rotation, normals are at angle+30, angle+90, ...
    normals = angle_rad + np.pi/6 + np.arange(6) * np.pi / 3
    # The distance from center to each side (apothem) = side * sqrt(3)/2
    apothem = side * np.sqrt(3) / 2
    for i in range(6):
        nx = np.cos(normals[i])
        ny = np.sin(normals[i])
        if lx * nx + ly * ny > apothem:
            return False
    return True

def _hexagon_to_hexagon_distance(c1, a1, c2, a2, side1=1.0, side2=1.0):
    """Approximate minimum distance between two hexagons (center-to-center minus max reach)."""
    # This is approximate; for exact we'd need more work
    # For now, use circumradius as a proxy for collision detection
    dist = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
    # Two unit hexagons: if centers are too close, they overlap
    # The minimum center distance for non-overlap depends on relative orientation
    # For identical orientation, min distance = 2 * inradius = sqrt(3) for side-by-side
    # But for arbitrary rotation, it's complex. Use 2*sqrt(3)/2 * something
    # Conservative: if dist < sqrt(3) - epsilon, definitely overlap
    return dist

def _required_outer_side(centers, angles, outer_angle=0.0, outer_center=(0.0, 0.0)):
    """Compute the minimum side length of a regular hexagon (with given center and rotation) 
    that contains all hexagon vertices."""
    # Get all vertices of all inner hexagons
    all_verts = []
    for i in range(len(centers)):
        cx, cy = centers[i]
        ang = angles[i]
        verts = _hexagon_vertices(cx, cy, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # The outer hexagon has center outer_center, rotation outer_angle, side S
    # A point (px,py) is inside if for all 6 normals: dot(p - center, normal) <= S * sqrt(3)/2
    # We need S such that max over all points and all normals of dot(p-center, normal) <= S * sqrt(3)/2
    # So S = max(dot) / (sqrt(3)/2)
    
    ox, oy = outer_center
    angle_rad = np.radians(outer_angle)
    normals = angle_rad + np.pi/6 + np.arange(6) * np.pi / 3
    
    max_proj = 0
    for i in range(6):
        nx = np.cos(normals[i])
        ny = np.sin(normals[i])
        # Project all vertices
        projections = (all_verts[:, 0] - ox) * nx + (all_verts[:, 1] - oy) * ny
        max_proj = max(max_proj, np.max(projections))
    
    side = max_proj / (np.sqrt(3) / 2)
    return side

def _check_overlap(centers, angles, eps=1e-4):
    """Check if any two hexagons overlap. Returns True if overlap detected."""
    n = len(centers)
    for i in range(n):
        for j in range(i+1, n):
            c1 = centers[i]
            c2 = centers[j]
            a1 = angles[i]
            a2 = angles[j]
            # Get vertices
            v1 = _hexagon_vertices(c1[0], c1[1], 1.0, a1)
            v2 = _hexagon_vertices(c2[0], c2[1], 1.0, a2)
            # Check if any vertex of v1 is inside v2 or vice versa
            # Also check edge intersections (simplified: just vertex-in-polygon)
            for vx, vy in v1:
                if _point_in_hexagon(vx, vy, c2[0], c2[1], 1.0, a2):
                    return True
            for vx, vy in v2:
                if _point_in_hexagon(vx, vy, c1[0], c1[1], 1.0, a1):
                    return True
    return False

def _objective(params):
    """Objective function for optimization."""
    # params: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11, ox, oy, S, outer_angle]
    n = N
    centers = []
    angles = []
    for i in range(n):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        centers.append((x, y))
        angles.append(a)
    ox = params[3*n]
    oy = params[3*n+1]
    S = params[3*n+2]
    outer_angle = params[3*n+3]
    
    # Penalty for overlap
    penalty = 0
    for i in range(n):
        for j in range(i+1, n):
            c1 = centers[i]
            c2 = centers[j]
            dist = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
            # Approximate: hexagons of side 1 have "radius" ~1
            # If dist < sqrt(3) - small, likely overlap
            min_dist = np.sqrt(3) - 0.1  # conservative
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2 * 100
    
    # Compute required side
    req_side = _required_outer_side(centers, angles, outer_angle, (ox, oy))
    
    # Return S + penalty, but we want to minimize S subject to S >= req_side
    # Actually, let's just minimize req_side + penalty
    return req_side + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s out of 150s budget
    
    # Initial honeycomb configuration
    # Hexagonal lattice: centers at distance sqrt(3) apart for non-overlapping unit hexagons
    # Flat-top hexagons: neighbors at (sqrt(3), 0), (sqrt(3)/2, 3/2), etc.
    # Let's use a compact arrangement
    
    # Configuration 1: 3-4-3-2 or similar compact cluster
    # Let's try: center at origin, ring of 6, then 5 more
    # Actually, let's use a known good packing: 2x6 rectangle or 3x4
    
    # Honeycomb positions (flat-top, side 1):
    # Row spacing: 1.5 (for pointy-top) or sqrt(3) for flat-top horizontal
    # Let's use pointy-top hexagons for easier honeycomb
    # Pointy-top: vertices at 30, 90, 150, 210, 270, 330
    # Neighbor offsets: (0, 2), (sqrt(3), 1), (sqrt(3), -1), (0, -2), (-sqrt(3), 1), (-sqrt(3), -1)
    
    sqrt3 = np.sqrt(3)
    
    # Try a few initial topologies
    topologies = []
    
    # Topology 1: 3 rows of 4
    top1 = []
    for row in range(3):
        for col in range(4):
            x = col * sqrt3 + (row % 2) * sqrt3/2
            y = row * 1.5
            top1.append((x, y, 0.0))
    topologies.append(top1)
    
    # Topology 2: 2 rows of 6
    top2 = []
    for row in range(2):
        for col in range(6):
            x = col * sqrt3 + (row % 2) * sqrt3/2
            y = row * 1.5
            top2.append((x, y, 0.0))
    topologies.append(top2)
    
    # Topology 3: 4 rows of 3
    top3 = []
    for row in range(4):
        for col in range(3):
            x = col * sqrt3 + (row % 2) * sqrt3/2
            y = row * 1.5
            top3.append((x, y, 0.0))
    topologies.append(top3)
    
    # Topology 4: Hexagonal cluster (1 + 6 + 5)
    top4 = [(0, 0, 0.0)]
    # Ring of 6
    for i in range(6):
        angle = i * np.pi / 3
        x = 2 * np.cos(angle)  # distance 2 for pointy-top neighbors? No, 2 is for circumradius
        # For pointy-top unit hex, neighbor center distance is 2 (circumradius) in some directions
        # Actually, for pointy-top: vertical neighbors at distance 2, diagonal at distance sqrt(3+1)=2? 
        # Let me recalculate: pointy-top hex with side 1, circumradius 1
        # Vertices at (cos(30°), sin(30°)) = (sqrt(3)/2, 1/2), etc.
        # The distance between centers of adjacent hexagons in a honeycomb is 2 * inradius = 2 * sqrt(3)/2 = sqrt(3) for horizontal
        # and 2 for vertical? No...
        # For pointy-top: the hexagon has width sqrt(3) (horizontal) and height 2 (vertical)
        # Adjacent hexagons share an edge. The center-to-center distance is 2 * apothem = 2 * sqrt(3)/2 = sqrt(3)
        # But this is the perpendicular distance. The actual center-to-center vector depends on direction.
        # For pointy-top, the 6 neighbor directions are at angles 0, 60, 120, 180, 240, 300 from horizontal? 
        # No, for pointy-top (vertex at top), the flat sides are on left/right, so neighbors are at 0°, 60°, 120°, 180°, 240°, 300°
        # Distance to each neighbor = 2 * apothem = sqrt(3)
        x = sqrt3 * np.cos(angle)
        y = sqrt3 * np.sin(angle)
        top4.append((x, y, 0.0))
    # Add 5 more to make 12: extend the ring
    # Add 5 more around the perimeter
    for i in range(5):
        angle = i * 2 * np.pi / 5 + np.pi / 5
        # Place them further out
        r = 2 * sqrt3
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        top4.append((x, y, 0.0))
    topologies.append(top4)
    
    best_result = None
    best_score = float('inf')
    
    for top_idx, top in enumerate(topologies):
        if time.time() > deadline - 30:
            break
        
        # Center the topology
        xs = [p[0] for p in top]
        ys = [p[1] for p in top]
        cx = np.mean(xs)
        cy = np.mean(ys)
        top_centered = [(x - cx, y - cy, a) for x, y, a in top]
        
        # Try different global rotations
        best_rot_score = float('inf')
        best_rot = 0.0
        best_config = None
        
        for rot_deg in np.arange(0, 60, 1.0):
            if time.time() > deadline - 20:
                break
            
            # Apply global rotation to all centers
            rot_rad = np.radians(rot_deg)
            rotated = []
            for x, y, a in top_centered:
                rx = x * np.cos(rot_rad) - y * np.sin(rot_rad)
                ry = x * np.sin(rot_rad) + y * np.cos(rot_rad)
                rotated.append((rx, ry, a + rot_deg))
            
            # Check for overlaps
            if _check_overlap([p[:2] for p in rotated], [p[2] for p in rotated]):
                continue
            
            # Compute required outer side
            side = _required_outer_side([p[:2] for p in rotated], [p[2] for p in rotated], 0.0, (0.0, 0.0))
            
            if side < best_rot_score:
                best_rot_score = side
                best_rot = rot_deg
                best_config = rotated
        
        if best_config is not None and best_rot_score < best_score:
            best_score = best_rot_score
            best_result = best_config
    
    if best_result is None:
        # Fallback: return the initial topology
        return [[p[0], p[1], p[2]] for p in topologies[0]], (0.0, 0.0), 10.0, 0.0
    
    # Polish with coordinate descent on the best configuration
    if time.time() < deadline - 30:
        centers = [p[:2] for p in best_result]
        angles = [p[2] for p in best_result]
        
        # Coordinate descent: try small perturbations
        for iteration in range(200):
            if time.time() > deadline - 25:
                break
            
            improved = False
            for i in range(N):
                for dx in [-0.1, 0.1]:
                    for dy in [-0.1, 0.1]:
                        new_centers = centers.copy()
                        new_centers[i] = (centers[i][0] + dx, centers[i][1] + dy)
                        
                        if _check_overlap(new_centers, angles):
                            continue
                        
                        side = _required_outer_side(new_centers, angles, 0.0, (0.0, 0.0))
                        if side < best_score - 1e-6:
                            best_score = side
                            centers = new_centers
                            improved = True
                
                for da in [-5, 5]:
                    new_angles = angles.copy()
                    new_angles[i] = angles[i] + da
                    if _check_overlap(centers, new_angles):
                        continue
                    side = _required_outer_side(centers, new_angles, 0.0, (0.0, 0.0))
                    if side < best_score - 1e-6:
                        best_score = side
                        angles = new_angles
                        improved = True
            
            if not improved:
                break
        
        best_result = [(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
    
    # Return the result
    inner = [[p[0], p[1], p[2]] for p in best_result]
    return inner, (0.0, 0.0), best_score, 0.0
# EVOLVE-BLOCK-END

