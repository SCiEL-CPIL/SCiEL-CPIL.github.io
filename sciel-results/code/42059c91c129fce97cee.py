"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 145.0  # 5s safety margin

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Helper: Generate vertices of a regular hexagon
    def hex_vertices(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = angle_rad + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            verts.append((x, y))
        return verts

    # Helper: Check if two convex polygons intersect
    def polygons_intersect(poly1, poly2):
        # Simple check: if any vertex of poly1 is inside poly2 or vice versa
        # For robustness in optimization, we use a distance-based penalty or 
        # a more robust intersection test. 
        # Given the small N and time limit, a vertex-in-polygon check for 
        # strict non-overlap is often sufficient if we penalize penetration.
        # However, for a hard constraint, we need to ensure they are disjoint.
        # Let's use a simple check: min distance between edges > 0.
        # Or simpler: if they overlap, the area of intersection > 0.
        # For speed in Nelder-Mead, we will use a penalty function based on 
        # the maximum penetration depth or just a boolean check converted to penalty.
        
        # A robust and fast check for convex polygons:
        # Separating Axis Theorem (SAT)
        def get_edges(poly):
            edges = []
            for i in range(len(poly)):
                x1, y1 = poly[i]
                x2, y2 = poly[(i + 1) % len(poly)]
                dx = x2 - x1
                dy = y2 - y1
                # Normal vector
                length = math.sqrt(dx*dx + dy*dy)
                if length == 0: continue
                nx = -dy / length
                ny = dx / length
                edges.append((nx, ny, poly[i]))
            return edges

        axes = []
        for edge in get_edges(poly1):
            axes.append((edge[0], edge[1]))
        for edge in get_edges(poly2):
            axes.append((edge[0], edge[1]))

        for nx, ny in axes:
            # Project both polygons onto the normal
            proj1 = [x*nx + y*ny for x, y in poly1]
            proj2 = [x*nx + y*ny for x, y in poly2]
            min1, max1 = min(proj1), max(proj1)
            min2, max2 = min(proj2), max(proj2)
            
            # Check for gap
            if max1 <= min2 or max2 <= min1:
                return False # Disjoint
        
        return True # Intersecting (or touching)

    def objective(params):
        if time.time() > DEADLINE:
            return 1000.0 # Large penalty to stop search
        
        # Unpack params: x, y, angle for the first hexagon
        x, y, angle = params
        
        # Generate all 12 hexagon centers and angles using 6-fold symmetry
        # The outer hexagon is centered at (0,0) with side L.
        # We need to determine L. 
        # L is determined by the furthest point of any inner hexagon from the origin.
        # Actually, the outer hexagon must contain all inner hexagons.
        # The "side length" of the outer hexagon is the parameter we minimize.
        # But the outer hexagon shape is fixed (regular).
        # So L is max over all inner hexagon vertices of the "hexagonal radius" distance?
        # No, the outer hexagon is a container. 
        # If we fix the inner hexagons, the minimum L is the smallest L such that
        # all inner hexagons are inside the outer hexagon of side L.
        
        # How to compute required L for a given set of inner hexagons?
        # The outer hexagon is defined by 6 half-planes.
        # A point (px, py) is inside a regular hexagon of side L centered at origin (aligned)
        # if:
        # |px| <= L * sqrt(3)/2 * ? No, let's use the support function.
        # For a regular hexagon of side L, the distance from center to side (apothem) is a = L * sqrt(3)/2.
        # The distance from center to vertex is R = L.
        # The hexagon is the intersection of 6 strips.
        # For an axis-aligned regular hexagon (flat top/bottom or pointy top?):
        # Standard orientation: vertices at 0, 60, 120... degrees?
        # Let's assume the outer hexagon is aligned with the x-axis (flat top/bottom? No, usually pointy top/bottom or flat).
        # The problem says "regular hexagon". The angle parameter for the outer hexagon is also part of the output, 
        # but we can fix it to 0 without loss of generality due to rotational symmetry of the space?
        # Actually, the inner hexagons can rotate. The outer hexagon rotation is a free parameter too.
        # However, since the inner hexagons can rotate freely, we can fix the outer hexagon to angle 0 
        # and let the inner ones adjust.
        
        # Let's fix outer hexagon at angle 0.
        # Vertices of outer hexagon (side L):
        # (L, 0), (L/2, L*sqrt(3)/2), (-L/2, L*sqrt(3)/2), (-L, 0), (-L/2, -L*sqrt(3)/2), (L/2, -L*sqrt(3)/2)
        # Wait, this is a "pointy" hexagon?
        # Distance from center to vertex is L.
        # Distance from center to edge (apothem) is L * sqrt(3)/2.
        
        # To check if a point (px, py) is inside this hexagon:
        # The hexagon is defined by |x| <= L, |y| <= L*sqrt(3)/2, and |sqrt(3)x + y| <= L*sqrt(3)?
        # Let's derive the inequalities.
        # Normals to the sides are at angles 30, 90, 150, 210, 270, 330?
        # If vertices are at 0, 60..., sides are perpendicular to 30, 90...
        # Side 1: normal at 90 deg (up). Equation: y <= L*sqrt(3)/2.
        # Side 2: normal at 30 deg. Equation: x*cos(30) + y*sin(30) <= L*sqrt(3)/2.
        # Side 3: normal at 150 deg. Equation: -x*cos(30) + y*sin(30) <= L*sqrt(3)/2.
        # And their negatives.
        
        # So a point (px, py) is inside if:
        # 1. |py| <= L * sqrt(3)/2
        # 2. |px * sqrt(3)/2 + py * 1/2| <= L * sqrt(3)/2
        # 3. |-px * sqrt(3)/2 + py * 1/2| <= L * sqrt(3)/2
        
        # We need to find the minimum L such that ALL vertices of ALL inner hexagons satisfy these.
        # L must be >= max over all vertices (px, py) of:
        # L_req = max( 2*|py|/sqrt(3), 2*|px*sqrt(3)/2 + py/2|/sqrt(3), 2*|-px*sqrt(3)/2 + py/2|/sqrt(3) )
        
        # Let's calculate L_required for the current configuration.
        
        # Generate the 12 inner hexagons
        centers = []
        angles = []
        for k in range(6):
            # Rotate the base center (x,y) by 60*k degrees
            theta = k * math.pi / 3
            rx = x * math.cos(theta) - y * math.sin(theta)
            ry = x * math.sin(theta) + y * math.cos(theta)
            centers.append((rx, ry))
            angles.append(angle + k * 60.0)
            
            # Also, we might need a second set? 
            # N=12. 6-fold symmetry gives 6 copies. 
            # We need 12. 
            # Perhaps we place 2 hexagons in the first sector?
            # Or maybe the symmetry is 12-fold? No, hexagon is 6-fold.
            # If we place 2 hexagons in the fundamental sector (60 degrees), we get 12 total.
            # Let's assume we place 2 hexagons in the first sector.
            # Params: x1, y1, a1, x2, y2, a2
            # This doubles the dimensionality to 12.
            
        # Let's stick to 2 hexagons per sector for better packing potential.
        # Params: x1, y1, a1, x2, y2, a2
        # Total 12 hexagons.
        
        # Wait, the objective function above only used 1 hexagon. 
        # I need to update the objective to handle 2 hexagons per sector.
        
        # Let's redefine the objective to accept 6 parameters: x1, y1, a1, x2, y2, a2
        pass

    # Refined Objective
    def objective_2(params):
        if time.time() > DEADLINE:
            return 1000.0
            
        x1, y1, a1, x2, y2, a2 = params
        
        # Generate 12 hexagons
        all_hexes = []
        for k in range(6):
            theta = k * math.pi / 3
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            
            # Hex 1 in sector k
            cx1 = x1 * cos_t - y1 * sin_t
            cy1 = x1 * sin_t + y1 * cos_t
            ang1 = a1 + k * 60.0
            
            # Hex 2 in sector k
            cx2 = x2 * cos_t - y2 * sin_t
            cy2 = x2 * sin_t + y2 * cos_t
            ang2 = a2 + k * 60.0
            
            all_hexes.append((cx1, cy1, ang1))
            all_hexes.append((cx2, cy2, ang2))
            
        # Check overlaps
        # We only need to check overlaps between hexagons in the same sector and adjacent sectors?
        # Due to symmetry, it's sufficient to check:
        # 1. Overlap between Hex1_k and Hex2_k (same sector)
        # 2. Overlap between Hex1_k and Hex1_{k+1} (adjacent)
        # 3. Overlap between Hex1_k and Hex2_{k+1} (adjacent)
        # 4. Overlap between Hex2_k and Hex1_{k+1} (adjacent)
        # 5. Overlap between Hex2_k and Hex2_{k+1} (adjacent)
        # Actually, just checking all pairs in the first two sectors (k=0 and k=1) is sufficient 
        # if we check all pairs within k=0 and between k=0 and k=1.
        
        penalty = 0.0
        
        # Get vertices for hexes in sector 0 and 1
        # Sector 0: (x1, y1, a1), (x2, y2, a2)
        # Sector 1: rotated by 60 deg
        
        hexes = []
        # k=0
        hexes.append(hex_vertices(x1, y1, 1.0, a1))
        hexes.append(hex_vertices(x2, y2, 1.0, a2))
        
        # k=1
        theta = math.pi / 3
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        
        cx1_1 = x1 * cos_t - y1 * sin_t
        cy1_1 = x1 * sin_t + y1 * cos_t
        ang1_1 = a1 + 60.0
        hexes.append(hex_vertices(cx1_1, cy1_1, 1.0, ang1_1))
        
        cx2_1 = x2 * cos_t - y2 * sin_t
        cy2_1 = x2 * sin_t + y2 * cos_t
        ang2_1 = a2 + 60.0
        hexes.append(hex_vertices(cx2_1, cy2_1, 1.0, ang2_1))
        
        # Check all pairs among these 4 hexagons
        for i in range(4):
            for j in range(i+1, 4):
                if polygons_intersect(hexes[i], hexes[j]):
                    # Calculate penetration depth or just add a large penalty
                    # Simple penalty: 10.0
                    penalty += 10.0
        
        if penalty > 0:
            return penalty
        
        # Calculate required L
        # We need to check all 12 hexagons' vertices against the outer hexagon constraints
        # But due to symmetry, checking the vertices of the 4 hexagons in sectors 0 and 1 
        # is sufficient? 
        # The outer hexagon is symmetric. The constraints are linear in x,y.
        # The maximum "L_req" will be determined by the furthest point.
        # Due to symmetry, the furthest point in sector k is the same as in sector 0 rotated.
        # So we just need to check the 4 hexagons in sectors 0 and 1? 
        # Actually, we need to check all vertices of the 12 hexagons.
        # But we can just check the 4 hexagons in sectors 0 and 1, 
        # because the set of vertices is just the rotation of the set in sector 0.
        # Wait, the outer hexagon is fixed. The vertices in sector 1 are rotated versions of sector 0.
        # The condition for a point p to be inside the outer hexagon is invariant under rotation 
        # if the outer hexagon is also rotated? No, the outer hexagon is fixed.
        # However, the outer hexagon is symmetric under 60 deg rotation.
        # So if a point p is inside, its rotation by 60 deg is also inside.
        # Therefore, we only need to check the vertices of the hexagons in sector 0 (and maybe 1 for boundary cases?).
        # Actually, just checking the 2 hexagons in sector 0 is sufficient for the L calculation?
        # Let's check all 4 to be safe, it's cheap.
        
        max_l_req = 0.0
        for h in hexes:
            for px, py in h:
                # Constraints:
                # 1. |py| <= L * sqrt(3)/2  => L >= 2*|py|/sqrt(3)
                # 2. |px*sqrt(3)/2 + py/2| <= L * sqrt(3)/2 => L >= 2*|px*sqrt(3)/2 + py/2|/sqrt(3)
                # 3. |-px*sqrt(3)/2 + py/2| <= L * sqrt(3)/2 => L >= 2*|-px*sqrt(3)/2 + py/2|/sqrt(3)
                
                l1 = 2.0 * abs(py) / math.sqrt(3)
                l2 = 2.0 * abs(px * math.sqrt(3)/2 + py * 0.5) / math.sqrt(3)
                l3 = 2.0 * abs(-px * math.sqrt(3)/2 + py * 0.5) / math.sqrt(3)
                
                l_req = max(l1, l2, l3)
                if l_req > max_l_req:
                    max_l_req = l_req
                    
        return max_l_req

    # Initial guess
    # Place hexagons in a ring?
    # Sector 0 is 0 to 60 deg.
    # Let's try placing them at distance 2.0 from center.
    # Hex 1: angle 30 deg (middle of sector), dist 2.0
    # Hex 2: angle 30 deg, dist 3.0? Or angle 10, dist 2.0?
    
    # Let's try a simple grid-like start
    # x1 = 2.0, y1 = 0.0, a1 = 0.0
    # x2 = 2.0, y2 = 1.0, a2 = 0.0
    
    x0 = [2.0, 0.0, 0.0, 2.0, 1.0, 0.0]
    
    # Use Nelder-Mead
    res = minimize(objective_2, x0, method='Nelder-Mead', 
                   options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
    
    if time.time() > DEADLINE:
         # Return a valid default if timed out
         inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
         return inner, (0.0, 0.0), 40.0, 0.0

    best_params = res.x
    best_L = res.fun
    
    # Reconstruct the 12 inner hexagons
    inner = []
    for k in range(6):
        theta = k * math.pi / 3
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        
        cx1 = best_params[0] * cos_t - best_params[1] * sin_t
        cy1 = best_params[0] * sin_t + best_params[1] * cos_t
        ang1 = best_params[2] + k * 60.0
        
        cx2 = best_params[3] * cos_t - best_params[4] * sin_t
        cy2 = best_params[3] * sin_t + best_params[4] * cos_t
        ang2 = best_params[5] + k * 60.0
        
        inner.append([cx1, cy1, ang1])
        inner.append([cx2, cy2, ang2])
        
    # The outer hexagon center is (0,0), side is best_L, angle is 0.0
    return inner, (0.0, 0.0), best_L, 0.0
# EVOLVE-BLOCK-END
