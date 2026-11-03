import numpy as np

N = 12

def construct():
    # Generate axial coordinates for a hex grid
    # We need enough points to pick the 12 closest
    # Radius 3 gives 37 points, which is enough
    max_radius = 5
    points = []
    for q in range(-max_radius, max_radius + 1):
        for r in range(-max_radius, max_radius + 1):
            # Hex grid constraint: |q+r| <= max_radius is not the right constraint for a square loop
            # We just generate all and filter by distance later, or generate hexagonally
            # Simpler: generate all in a box, convert, sort by distance
            # Axial to Cartesian (pointy-top)
            # x = sqrt(3) * (q + r/2)
            # y = 1.5 * r
            x = np.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            points.append((x, y))
            
    # Sort by distance from origin
    points.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take the first N points
    inner_points = points[:N]
    
    # Calculate the minimal side length S of the outer hexagon
    # The outer hexagon is centered at (0,0) with angle 0
    # Side length S equals the circumradius (distance from center to vertex)
    # We need S such that all vertices of the inner hexagons are inside or on the boundary
    
    # For each inner hexagon center (x, y), calculate the max distance to its vertices
    # A unit hexagon (side 1) has vertices at distance 1 from its center
    # The vertices of a hexagon at (x,y) with angle 0 are:
    # (x + cos(theta), y + sin(theta)) for theta in [0, 60, 120, 180, 240, 300] degrees
    # Wait, standard "pointy-top" hexagon vertices are at 30, 90, 150...?
    # Let's check the orientation.
    # If x = sqrt(3)(q+r/2), y=1.5r, this is pointy-top.
    # Vertices of a pointy-top unit hexagon centered at origin:
    # (0, 1), (sqrt(3)/2, 0.5), (sqrt(3)/2, -0.5), (0, -1), (-sqrt(3)/2, -0.5), (-sqrt(3)/2, 0.5)
    # Max distance from origin for a hex at (x,y) is max over vertices v of || (x,y) + v ||
    
    max_dist = 0.0
    for x, y in inner_points:
        # Vertices of the unit hexagon
        # Angle 0 usually means flat-top or pointy-top?
        # The task says "angle_degrees" in the output.
        # If we return 0.0, we must ensure the geometry matches.
        # Let's assume the evaluator calculates distance based on the returned angle.
        # If we return angle 0, and our coordinates are for angle 0, it's consistent.
        # Let's use the vertices for angle 0 (pointy-top as derived above)
        angles = np.deg2rad([30, 90, 150, 210, 270, 330]) # Pointy top vertices
        # Wait, if y=1.5r, the "up" direction is y.
        # Vertex at (0,1) is at 90 deg.
        # So vertices are at 30, 90, 150, 210, 270, 330.
        for a in angles:
            vx = x + np.cos(a)
            vy = y + np.sin(a)
            dist = np.sqrt(vx**2 + vy**2)
            if dist > max_dist:
                max_dist = dist
                
    # The side length of the outer hexagon must be at least max_dist
    # To be safe and account for floating point, we can add a tiny epsilon
    # But usually exact fit is fine.
    side_length = max_dist
    
    # Construct the return values
    inner = []
    for x, y in inner_points:
        # Angle 0
        inner.append([x, y, 0.0])
        
    center = (0.0, 0.0)
    angle = 0.0
    
    return inner, center, side_length, angle
