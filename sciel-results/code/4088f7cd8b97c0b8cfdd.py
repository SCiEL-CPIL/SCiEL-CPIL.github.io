import numpy as np
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate 11 unit hexagons in a compact cluster
    # 1 center, 6 surrounding, 4 extending outwards
    
    # Lattice vectors for hexagon centers
    # Distance between centers of adjacent hexagons is sqrt(3)
    d = math.sqrt(3)
    
    # Center
    centers = [(0.0, 0.0)]
    
    # 6 neighbors
    for i in range(6):
        angle = i * math.pi / 3
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        centers.append((x, y))
        
    # 4 more extending from 4 of the neighbors
    # To maintain symmetry and minimize bounding box, extend from 0, 60, 120, 180
    # These are indices 0, 1, 2, 3 in the neighbor list (if we started at 0)
    # Neighbor 0 is at angle 0. Next is at 2*angle 0 distance.
    # Neighbor 1 is at angle 60. Next is at 2*angle 60 distance.
    # Neighbor 2 is at angle 120. Next is at 2*angle 120 distance.
    # Neighbor 3 is at angle 180. Next is at 2*angle 180 distance.
    
    for i in [0, 1, 2, 3]:
        angle = i * math.pi / 3
        x = 2 * d * math.cos(angle)
        y = 2 * d * math.sin(angle)
        centers.append((x, y))
        
    # Now we have 11 centers.
    # We need to find the minimal side length L of a regular hexagon
    # centered at (0,0) that contains all these hexagons.
    # The hexagons have side length 1.
    # The vertices of a hexagon centered at (cx, cy) with rotation 0 are at:
    # (cx + cos(k*pi/3), cy + sin(k*pi/3)) for k=0..5
    
    all_vertices = []
    for cx, cy in centers:
        for k in range(6):
            angle = k * math.pi / 3
            vx = cx + math.cos(angle)
            vy = cy + math.sin(angle)
            all_vertices.append((vx, vy))
            
    # Find minimal L for a regular hexagon centered at (0,0)
    # A regular hexagon of side L can be defined by 6 inequalities.
    # If the hexagon is rotated by angle phi, the constraints are:
    # |dot(v, n_i)| <= L * sqrt(3) / 2 for normals n_i?
    # Actually, the distance from center to a side is L * sqrt(3) / 2.
    # The distance from center to a vertex is L.
    # The width in the direction of a normal is L * sqrt(3).
    # The width in the direction of a vertex is 2 * L.
    
    # We can iterate over possible rotations of the container hexagon.
    # Due to symmetry of the point set (approx), we can check a range.
    # The point set has spikes at 0, 60, 120, 180.
    # The container should probably align its vertices with the spikes.
    # If container vertices are at 0, 60, 120, 180, 240, 300.
    # Then the constraint for the spike at 0 is:
    # The vertex of the inner hexagon at 0 is at distance R_max.
    # The container vertex at 0 is at distance L.
    # So L >= R_max.
    
    # Let's compute R_max for the set of vertices.
    # R_max = max(sqrt(x^2+y^2))
    r_max = max(math.sqrt(x*x + y*y) for x, y in all_vertices)
    
    # If we align the container such that a vertex points to the max radius direction,
    # then L must be at least r_max.
    # However, we also need to satisfy the other directions.
    # Let's check if L = r_max works.
    # If we set L = r_max, and rotate the container so that a vertex is at angle 0 (where the max radius is),
    # does it contain everything?
    
    # The container vertices are at angles phi, phi+60, ...
    # The container sides are at angles phi+30, phi+90, ...
    # The condition for a point (x,y) to be inside is:
    # For each of the 3 axes (0, 60, 120 relative to container), the projection must be <= L * sqrt(3) / 2?
    # No, the inradius is L * sqrt(3) / 2.
    # So the max projection on the normal to the sides must be <= L * sqrt(3) / 2.
    
    # Let's try to optimize phi.
    # We want to minimize L such that:
    # max_{v in V} |dot(v, n_i(phi))| <= L * sqrt(3) / 2 for i=0,1,2
    # where n_i are the normals to the sides.
    
    # Since the set is symmetric (spikes at 0, 60, 120, 180),
    # the optimal phi is likely 0 or 30.
    
    # Case 1: Container vertex at 0. Normals at 30, 90, 150.
    # Case 2: Container flat at 0. Normals at 0, 60, 120.
    
    # Let's compute the required L for both cases.
    
    def required_L(phi):
        # phi is the angle of the first vertex of the container.
        # Normals are at phi + 30, phi + 90, phi + 150.
        normals = []
        for i in range(3):
            n_angle = phi + math.pi/6 + i * math.pi/3
            normals.append((math.cos(n_angle), math.sin(n_angle)))
        
        max_proj = 0
        for x, y in all_vertices:
            for nx, ny in normals:
                proj = abs(x*nx + y*ny)
                if proj > max_proj:
                    max_proj = proj
        # L * sqrt(3) / 2 >= max_proj
        return max_proj * 2 / math.sqrt(3)
    
    # Check phi = 0 (vertex at 0)
    L1 = required_L(0)
    # Check phi = 30 (flat at 0, vertex at 30)
    L2 = required_L(math.pi/6)
    
    # Pick the better one
    L = min(L1, L2)
    
    # Construct the inner list
    # (x, y, angle_degrees)
    # All inner hexagons are aligned with the lattice, so angle is 0.
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, 0.0])
        
    return inner, (0.0, 0.0), L, 0.0
# EVOLVE-BLOCK-END
