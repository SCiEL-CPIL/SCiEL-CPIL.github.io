import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate the 11 hexagons
    inner = []
    
    # 1. Center hexagon
    inner.append([0.0, 0.0, 0.0])
    
    # 2. First ring: 6 hexagons at distance sqrt(3)
    dist1 = math.sqrt(3)
    for i in range(6):
        angle_rad = i * math.pi / 3.0
        x = dist1 * math.cos(angle_rad)
        y = dist1 * math.sin(angle_rad)
        # Rotate so that the flat side faces outward (towards the outer hexagon boundary)
        # If the outer hexagon is pointy-top (vertices at 0, 60, ...), 
        # and the inner hex is at angle 0, its vertex points to 0.
        # We want the flat to point to 0? No, we want to minimize the radius.
        # The max extent of a hexagon is 1 (vertex) or sqrt(3)/2 (flat).
        # We want the direction from origin to be a flat direction.
        # For a hexagon centered at (dist, 0), the outward direction is 0.
        # If we rotate the hex by 30 degrees, the flat normal is at 0?
        # Standard hex (angle 0): Vertices at 0, 60... Flats at 30, 90...
        # So if we rotate by 30, vertices are at 30, 90... Flats at 0, 60...
        # Yes, rotate by 30 degrees relative to the position angle?
        # Actually, if position angle is theta, the outward vector is theta.
        # We want a flat normal at theta.
        # Flat normals are at 30 + k*60.
        # So we need rotation r such that 30 + k*60 == theta.
        # r = theta - 30 (mod 60).
        # Let's just use theta + 30 and see.
        angle_deg = math.degrees(angle_rad)
        rot = angle_deg + 30.0
        inner.append([x, y, rot])
        
    # 3. Second ring: 4 hexagons at distance 3
    # We pick 4 positions from the 12 available in the lattice.
    # To keep symmetry, we pick 4 that are roughly evenly distributed.
    # Available angles for dist 3: 30, 90, 150, 210, 270, 330.
    # Let's pick 30, 90, 210, 270.
    dist2 = 3.0
    angles_deg = [30.0, 90.0, 210.0, 270.0]
    for ang in angles_deg:
        angle_rad = math.radians(ang)
        x = dist2 * math.cos(angle_rad)
        y = dist2 * math.sin(angle_rad)
        rot = ang + 30.0
        inner.append([x, y, rot])
        
    # Calculate the required outer side length S
    # The outer hexagon is centered at (0,0).
    # We assume the outer hexagon is pointy-top (vertices at 0, 60, ...).
    # A point (x,y) is inside if it satisfies the inequalities.
    # For a regular hexagon of side S, the distance from center to a vertex is S.
    # The distance to the midpoint of a side is S * sqrt(3) / 2.
    
    # We need to find the minimum S such that all vertices of all inner hexagons are inside.
    # Since we rotated the inner hexagons to have flats facing out,
    # the "radius" of each inner hexagon in the radial direction is sqrt(3)/2.
    # However, the vertices of the inner hexagons might stick out in angular directions.
    
    # Let's compute the max "effective radius" required.
    # For each inner hexagon, we can compute the max distance of its vertices from the origin.
    # But the outer hexagon is not a circle.
    # We need to check the constraint:
    # |x| <= S/2 ? No.
    # For a pointy-top hexagon with side S:
    # The vertices are at (S, 0), (S/2, S*sqrt(3)/2), ...
    # The equation of the line connecting (S,0) and (S/2, S*sqrt(3)/2) is:
    # y = -sqrt(3) * (x - S)  => sqrt(3)x + y = S*sqrt(3)
    # Generally, for a point (x,y) to be inside, it must satisfy:
    # max( |x|, |x/2 + y*sqrt(3)/2|, |-x/2 + y*sqrt(3)/2| ) <= S ?
    # Actually, the support function of a regular hexagon of side S (pointy top) is:
    # h(theta) = S * max( cos(theta), cos(theta - pi/3), ... )?
    # Simpler: The distance from the origin to the boundary in direction theta is:
    # r(theta) = S * (sqrt(3)/2) / cos( (theta mod pi/3) - pi/6 )
    
    max_r = 0.0
    for x, y, rot_deg in inner:
        rot_rad = math.radians(rot_deg)
        # Vertices of the unit hexagon
        # Standard vertices for angle 0 (pointy top):
        # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
        # Wait, standard "pointy top" usually has a vertex at 90?
        # Let's stick to: Vertex at angle 0 is (1,0).
        # Vertices at 0, 60, 120, 180, 240, 300.
        verts = []
        for k in range(6):
            va = rot_rad + k * math.pi / 3.0
            vx = math.cos(va)
            vy = math.sin(va)
            px = x + vx
            py = y + vy
            verts.append((px, py))
            
        for px, py in verts:
            # Check distance to boundary in direction of (px, py)
            # If (px, py) is inside hex of side S, then S >= r_required(px, py)
            # r_required is the distance from origin to the boundary of the hex in the direction of (px,py).
            # If the point is at distance d and angle phi, it is inside if d <= r(phi).
            # So we need S such that d <= r_S(phi).
            # r_S(phi) = S * (sqrt(3)/2) / cos( (phi mod pi/3) - pi/6 )
            # So S >= d * cos( (phi mod pi/3) - pi/6 ) / (sqrt(3)/2)
            
            d = math.sqrt(px*px + py*py)
            phi = math.atan2(py, px)
            # Normalize phi to [0, pi/3)
            phi_mod = phi % (math.pi / 3.0)
            # The term cos(phi_mod - pi/6)
            denom = math.cos(phi_mod - math.pi / 6.0)
            if denom < 1e-9:
                # This shouldn't happen for a hexagon
                s_req = d * 2.0
            else:
                s_req = d * denom / (math.sqrt(3.0) / 2.0)
            
            if s_req > max_r:
                max_r = s_req

    # Add a small margin to ensure validity due to floating point errors
    S = max_r * 1.0001
    
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END
