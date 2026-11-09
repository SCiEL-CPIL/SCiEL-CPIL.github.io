import numpy as np
import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate 11 unit hexagons in a compact cluster
    # Using axial coordinates (q, r)
    # Center at (0,0)
    # 3x3 grid is 9 hexagons: q in {-1, 0, 1}, r in {-1, 0, 1}
    # Add 2 more to make 11.
    # Let's add (2, 0) and (-2, 0) to extend the width?
    # Or (0, 2) and (0, -2)?
    # A 3x3 block is compact. Adding 2 on the sides might increase bounding box.
    # Let's try a "flower" of 7 + 4 more.
    # 7: (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # 4 more: (2,0), (0,2), (-2,2), (-2,0)?
    # Let's just use the first 11 of a hexagonal spiral.
    
    centers_axial = []
    # Spiral generation
    centers_axial.append((0,0))
    # Ring 1: 6 hexagons
    ring1 = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    centers_axial.extend(ring1)
    # We have 7. Need 4 more from Ring 2.
    # Ring 2 starts at (2,0)
    # (2,0), (2,-1), (2,-2), (1,-2)
    ring2_part = [(2,0), (2,-1), (2,-2), (1,-2)]
    centers_axial.extend(ring2_part)
    
    # Convert axial to cartesian
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    sqrt3 = math.sqrt(3)
    centers_cart = []
    for q, r in centers_axial:
        x = sqrt3 * (q + r/2.0)
        y = 1.5 * r
        centers_cart.append((x, y))
        
    # Generate vertices for each hexagon
    # Unit hexagon, flat-top (vertices at 0, 60, 120...) or pointy-top?
    # Standard tiling usually assumes pointy-top for this axial formula?
    # Let's check.
    # If q=1, r=0 -> x=sqrt(3), y=0.
    # If pointy-top, right neighbor is at (sqrt(3), 0). Correct.
    # So inner hexagons are pointy-top.
    # Vertices of a pointy-top unit hexagon at (cx, cy):
    # (cx, cy+1), (cx+sqrt(3)/2, cy+0.5), (cx+sqrt(3)/2, cy-0.5), 
    # (cx, cy-1), (cx-sqrt(3)/2, cy-0.5), (cx-sqrt(3)/2, cy+0.5)
    
    all_vertices = []
    for cx, cy in centers_cart:
        verts = [
            (cx, cy + 1.0),
            (cx + sqrt3/2.0, cy + 0.5),
            (cx + sqrt3/2.0, cy - 0.5),
            (cx, cy - 1.0),
            (cx - sqrt3/2.0, cy - 0.5),
            (cx - sqrt3/2.0, cy + 0.5)
        ]
        all_vertices.extend(verts)
        
    all_vertices = np.array(all_vertices)
    
    # Find minimal bounding hexagon
    # Outer hexagon is also regular.
    # We can rotate the outer hexagon.
    # Due to symmetry of the point set (roughly), optimal rotation is likely near 0 or 30.
    # We search for theta in [0, pi/6)
    
    min_L = float('inf')
    best_theta = 0.0
    
    # The condition for a point (x,y) to be inside a regular hexagon of side L,
    # centered at origin, rotated by theta (flat-top at theta=0? No, let's define carefully).
    # Let's use the apothem method.
    # For a regular hexagon with side L, apothem A = L * sqrt(3) / 2.
    # The hexagon is the intersection of 6 strips.
    # The normals to the sides are at angles theta + k * 60.
    # For a point p, the projection on normal n_k must be <= A.
    # So A >= max_k max_p (p . n_k)
    # L = 2 * A / sqrt(3)
    
    for theta_deg in np.linspace(0, 30, 61):
        theta = np.radians(theta_deg)
        # Normals
        angles = theta + np.arange(6) * np.pi / 3.0
        normals = np.column_stack((np.cos(angles), np.sin(angles)))
        
        # Projections
        # (612 points x 6 normals)
        projections = all_vertices @ normals.T
        
        # Max projection for each normal
        max_projs = np.max(projections, axis=0)
        
        # Required apothem
        A = np.max(max_projs)
        
        # Required side length
        L = 2.0 * A / sqrt3
        
        if L < min_L:
            min_L = L
            best_theta = theta_deg
            
    # The inner hexagons are at 0 degrees (pointy-top).
    # The outer hexagon is at best_theta.
    # We need to return the inner hexagons' positions, angles, and the outer hexagon's center, side, angle.
    
    inner = []
    for cx, cy in centers_cart:
        inner.append([cx, cy, 0.0])
        
    center = [0.0, 0.0]
    side = min_L
    angle = best_theta
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
