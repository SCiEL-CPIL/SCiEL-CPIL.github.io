import math
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate centers for N=11 in a hexagonal lattice
    # Axial coordinates (q, r)
    # We want the 11 closest points to (0,0)
    # Norm squared: q^2 + qr + r^2
    points = []
    for q in range(-5, 6):
        for r in range(-5, 6):
            norm_sq = q*q + q*r + r*r
            points.append((norm_sq, q, r))
    
    points.sort()
    # Take the first 11
    selected = points[:11]
    
    centers = []
    for _, q, r in selected:
        # Convert axial to cartesian
        # x = sqrt(3) * (q + r/2)
        # y = 1.5 * r
        x = math.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        centers.append((x, y))
        
    # Generate vertices for each hexagon
    # Hexagon side length 1
    # "Pointy top" orientation (consistent with lattice spacing)
    # Vertices at angles 30, 90, 150, 210, 270, 330 degrees
    # Relative to center
    angles = [math.radians(30 + 60*i) for i in range(6)]
    vertices = []
    for cx, cy in centers:
        for ang in angles:
            vx = cx + math.cos(ang)
            vy = cy + math.sin(ang)
            vertices.append((vx, vy))
            
    # Find min enclosing hexagon side length
    best_S = float('inf')
    best_theta = 0.0
    
    # Try rotating the outer hexagon
    # Due to symmetry, we only need to check 0 to 30 degrees
    for deg in range(0, 31, 1):
        theta = math.radians(deg)
        # Normals of the outer hexagon faces
        # If outer hexagon is "flat top", normals are at 0, 60, ...
        # Let's assume outer hexagon is "flat top" (flat sides horizontal)
        # Normals: 0, 60, 120, 180, 240, 300
        # Rotated by theta: theta, theta+60, ...
        normals = [theta + math.radians(60*i) for i in range(6)]
        
        max_required_A = 0.0
        for vx, vy in vertices:
            dist = math.sqrt(vx*vx + vy*vy)
            if dist == 0:
                continue
            angle_v = math.atan2(vy, vx)
            
            # Find nearest normal
            min_diff = float('inf')
            for n in normals:
                diff = abs((angle_v - n) % (2*math.pi))
                if diff > math.pi:
                    diff = 2*math.pi - diff
                if diff < min_diff:
                    min_diff = diff
                    
            # Required apothem A such that dist <= A / cos(min_diff)
            # A >= dist * cos(min_diff)
            required_A = dist * math.cos(min_diff)
            if required_A > max_required_A:
                max_required_A = required_A
                
        # S = 2 * A / sqrt(3)
        S = 2.0 * max_required_A / math.sqrt(3)
        
        if S < best_S:
            best_S = S
            best_theta = deg
            
    # Construct the return values
    # inner: list of [x, y, angle_degrees]
    # All inner hexagons have 0 rotation relative to the lattice
    inner = [[x, y, 0.0] for x, y in centers]
    
    # Outer center (0,0)
    center = (0.0, 0.0)
    
    # Outer side length
    side = best_S
    
    # Outer rotation
    angle = float(best_theta)
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
