"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3.0)
    
    # Generate all 19 centers of the side-3 hexagon cluster
    # Axial coordinates (q, r) with max(|q|, |r|, |q+r|) <= 2
    # Using Flat-Top hexagon grid coordinates
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                # Convert axial to cartesian (flat-top hexagons)
                # x = 1.5 * q, y = sqrt(3) * (r + q/2)
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y, q, r))
    
    # We have 19 centers, need to choose 12
    # The outer hexagon side length L must satisfy:
    # For each inner hex center (cx, cy), the hexagon's vertices must be inside outer hex
    # A unit hexagon centered at (cx,cy) with same orientation as outer has vertices at
    # distance 1 from (cx,cy) in 6 directions
    # The outer hexagon (side L, center at origin, pointy-top) contains point (px,py) iff:
    # max(|px|/sqrt3, |py - px/sqrt3|/1.5, |py + px/sqrt3|/1.5) ... 
    # Actually for a regular hexagon with side L, pointy-top, centered at origin:
    # It's the intersection of 6 half-planes. The inradius is L*sqrt3/2.
    # A point (x,y) is inside iff for all 6 directions, the projection is within inradius.
    # Directions: 0, 60, 120, 180, 240, 300 degrees
    # For pointy-top hex, the flat sides are at angles 30, 90, 150, 210, 270, 330
    # Let me use: hexagon with vertices at angles 0,60,120,180,240,300 (flat-top)
    # or vertices at 30,90,150,210,270,330 (pointy-top)
    
    # Let's use flat-top: vertices at (L*cos(60k), L*sin(60k)) for k=0..5
    # The hexagon contains (x,y) iff max over k of |x*cos(60k) + y*sin(60k)| <= L*sqrt3/2
    # No wait, that's for the inscribed circle. Let me think again.
    
    # For a regular hexagon with circumradius R (distance center to vertex = R):
    # Vertices at R*(cos(60k), sin(60k))
    # The hexagon is the intersection of 6 half-planes:
    # For each edge, the half-plane containing the center
    # Edge between vertices at angles 60k and 60(k+1): midpoint at angle 60k+30, distance R*cos(30)=R*sqrt3/2
    # Half-plane: x*cos(60k+30) + y*sin(60k+30) <= R*sqrt3/2
    # Side length s = R (for regular hexagon, side = circumradius)
    # So R = L (side length)
    # Inradius = L*sqrt3/2
    
    def hex_contains(L, px, py):
        """Check if point (px,py) is inside regular hexagon of side L, flat-top, centered at origin."""
        R = L  # circumradius = side length for regular hexagon
        inradius = R * math.sqrt(3.0) / 2.0
        for k in range(6):
            angle = math.radians(60 * k + 30)
            proj = px * math.cos(angle) + py * math.sin(angle)
            if proj > inradius + 1e-10:
                return False
        return True
    
    def required_L(centers_subset):
        """Find minimum L such that all unit hexagons at given centers fit inside outer hex of side L."""
        # For each inner hex center, check all 6 vertices
        # Unit hexagon vertices (flat-top, side 1): at angles 0,60,120,180,240,300, distance 1
        min_L = 0.0
        for (cx, cy, _, _) in centers_subset:
            for k in range(6):
                angle = math.radians(60 * k)
                vx = cx + math.cos(angle)
                vy = cy + math.sin(angle)
                # Find minimum L such that (vx,vy) is in hex of side L
                # Outer hex is flat-top (normals at 30,90,150,210,270,330)
                # Condition: proj onto normal <= inradius = L * sqrt(3) / 2
                max_proj = 0.0
                for j in range(6):
                    ang = math.radians(60 * j + 30)
                    proj = vx * math.cos(ang) + vy * math.sin(ang)
                    if proj > max_proj:
                        max_proj = proj
                L_needed = max_proj / (math.sqrt(3.0) / 2.0)
                if L_needed > min_L:
                    min_L = L_needed
        return min_L
    
    # Try all subsets of size 12 from 19 centers
    # C(19,12) = C(19,7) = 50388 - manageable
    from itertools import combinations
    
    best_L_ideal = float('inf')
    best_subset = None
    
    # Search for the subset that minimizes the outer hexagon size assuming touching is allowed
    for subset in combinations(range(19), 12):
        if time.time() > deadline:
            break
        centers_subset = []
        for i in subset:
            x, y, _, _ = all_centers[i]
            centers_subset.append((x, y, 0, 0))
        
        L = required_L(centers_subset)
        if L < best_L_ideal:
            best_L_ideal = L
            best_subset = subset
    
    if best_subset is None:
        # Fallback: use first 12
        best_subset = tuple(range(12))
        centers_subset = []
        for i in best_subset:
            x, y, _, _ = all_centers[i]
            centers_subset.append((x, y, 0, 0))
        best_L_ideal = required_L(centers_subset)
    
    # To ensure strict disjointness, we scale the positions slightly.
    # This increases the distance between centers from sqrt(3) to S*sqrt(3).
    # We must recalculate the required outer L for these scaled positions.
    SCALE = 1.001
    
    inner = []
    scaled_centers_final = []
    for i in best_subset:
        x, y, _, _ = all_centers[i]
        sx, sy = x * SCALE, y * SCALE
        inner.append([sx, sy, 0.0])
        scaled_centers_final.append((sx, sy, 0, 0))
    
    final_L = required_L(scaled_centers_final)
    
    return inner, (0.0, 0.0), final_L, 0.0
# EVOLVE-BLOCK-END
