"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill

    # 1. Generate positions for N unit hexagons on a hexagonal lattice
    #    Spacing is sqrt(3) (width of a unit hexagon) to allow tiling.
    positions = []
    # Hexagonal numbers: 1, 7, 19, 37, 61...
    # For N=11, we use a central hexagon (1) + 6 neighbors + 4 more from the next ring.
    # This forms a compact cluster.
    
    # Generate coordinates in axial system (q, r) for a hex grid
    # Then convert to Cartesian
    # Axial to Cartesian (pointy-top, but we use flat-top for simplicity of width/height? 
    # Actually, let's use standard flat-top hexagons where width = 2, height = sqrt(3).
    # Wait, the problem says "Unit regular hexagons". Usually side length 1.
    # Width of flat-top hex (side 1) is 2. Height is sqrt(3).
    # If we tile them, the horizontal distance between centers is 1.5 (for columns) 
    # and vertical is sqrt(3).
    # Let's stick to the triangular lattice spacing of sqrt(3) mentioned in the prompt?
    # "spacing equal to the width of a unit hexagon (sqrt(3))"
    # If side length is 1, the distance between centers in a tiling is:
    # Horizontal neighbors: 1.5
    # Vertical neighbors: sqrt(3)
    # Diagonal neighbors: sqrt(3)
    # The prompt says "spacing equal to the width of a unit hexagon (sqrt(3))".
    # This implies a specific orientation or a simplified model. 
    # If we place them on a triangular grid with side length sqrt(3), 
    # the distance between centers is sqrt(3).
    # This corresponds to hexagons touching at vertices? No, that would be distance 2.
    # If distance is sqrt(3) ~ 1.732, and width is 2, they overlap.
    # Wait, "width of a unit hexagon" usually refers to the distance between parallel sides, which is sqrt(3).
    # If we place centers at distance sqrt(3), and align them, do they fit?
    # If hexagon A is at (0,0) and B at (sqrt(3), 0).
    # Width of A is 2 (from -1 to 1). Right edge of A is at x=1.
    # Left edge of B is at sqrt(3)-1 ~ 0.732.
    # They overlap.
    # So the prompt's "spacing equal to the width" might refer to the distance between *parallel sides* being the spacing?
    # In a perfect tiling of flat-top hexagons:
    # x-spacing = 1.5
    # y-spacing = sqrt(3)
    # The distance between centers is sqrt(1.5^2 + (sqrt(3)/2)^2) = sqrt(2.25 + 0.75) = sqrt(3).
    # Ah, the distance between centers in a hexagonal tiling IS sqrt(3).
    # So the prompt is describing the center-to-center distance in a tiling.
    
    # So we generate a cluster of N points on a hex grid with center-to-center distance sqrt(3).
    
    # Hex grid coordinates (q, r) to Cartesian (x, y) for flat-top hexagons:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    # Distance between (0,0) and (1,0) is 1.5.
    # Distance between (0,0) and (1,1) is sqrt(1.5^2 + (sqrt(3)*1.5)^2) ... wait.
    # Let's check the distance between (0,0) and (1,0) in axial coords for flat top:
    # x1=0, y1=0
    # x2=1.5, y2=sqrt(3)*0.5
    # dist = sqrt(1.5^2 + 0.75^2) = sqrt(2.25+0.75) = sqrt(3). Correct.
    
    # We need N=11 points.
    # Ring 0: (0,0) -> 1 point
    # Ring 1: 6 points
    # Ring 2: 12 points
    # Total 1 + 6 = 7. Need 4 more from Ring 2.
    
    def axial_to_cartesian(q, r):
        x = 1.5 * q
        y = math.sqrt(3) * (r + q / 2.0)
        return x, y

    # Get points in order of distance from center
    candidates = []
    # Ring 0
    candidates.append((0, 0))
    # Ring 1
    for q, r in [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]:
        candidates.append((q, r))
    # Ring 2 (partial)
    # The 12 points of ring 2 are:
    # (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,1)
    # We just need 4 more. Let's pick them to be symmetric if possible, or just take the first 4.
    # To keep it compact, any 4 from ring 2 will work, but symmetry is nice.
    # Let's take (2,0), (0,2), (-2,0), (0,-2) ? No, (0,-2) is in ring 2?
    # (0,-2): q=0, r=-2. Dist is 2.
    # Let's just iterate through ring 2 points and add until we have 11.
    ring2_points = [
        (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -1), 
        (-2, 0), (-2, 1), (-2, 2), (-1, 2), (0, 2), (1, 1)
    ]
    for p in ring2_points:
        if len(candidates) >= N:
            break
        candidates.append(p)
        
    # Ensure we have exactly N
    candidates = candidates[:N]
    
    # Convert to Cartesian
    inner_points = []
    for q, r in candidates:
        x, y = axial_to_cartesian(q, r)
        inner_points.append((x, y))
        
    # 2. Determine the minimal enclosing regular hexagon
    # The inner hexagons are rotated by 0 degrees (flat-top).
    # The outer hexagon should also be rotated by 0 degrees to align with the lattice.
    # We need to find the side length S of the outer hexagon (centered at 0,0) 
    # such that all inner hexagons (which are also flat-top, side 1) are inside.
    
    # A flat-top regular hexagon with side length S centered at (0,0) has vertices at:
    # (S, 0), (S/2, S*sqrt(3)/2), (-S/2, S*sqrt(3)/2), (-S, 0), (-S/2, -S*sqrt(3)/2), (S/2, -S*sqrt(3)/2)
    # The condition for a point (x,y) to be inside a flat-top hexagon of side S is:
    # |x| <= S
    # |y| <= S * sqrt(3) / 2
    # |x|/2 + |y|/sqrt(3) <= S/2  (Wait, this is for the slanted sides)
    # Let's derive the inequalities for a flat-top hexagon with "radius" S (distance to vertex).
    # Vertices: (S,0), (S/2, S*sqrt(3)/2), ...
    # The sides are:
    # 1. Right vertical? No, flat-top has vertical sides?
    # Flat-top: Top and bottom are flat. Sides are slanted?
    # No, "flat-top" means the top edge is horizontal.
    # Vertices for flat-top (side 1):
    # (1, 0), (1/2, sqrt(3)/2), (-1/2, sqrt(3)/2), (-1, 0), (-1/2, -sqrt(3)/2), (1/2, -sqrt(3)/2)
    # Yes, this is flat-top. The top edge connects (1/2, sqrt(3)/2) and (-1/2, sqrt(3)/2).
    # The inequalities for a point (x,y) to be inside a flat-top hexagon of side S:
    # 1. |y| <= S * sqrt(3) / 2  (Top and bottom flat edges)
    # 2. |x| <= S  (Left and right vertices... wait, the vertical extent at x=0 is S*sqrt(3)/2. The horizontal extent is S.)
    # 3. The slanted edges:
    #    Top-Right edge connects (S, 0) and (S/2, S*sqrt(3)/2).
    #    Equation: y = -sqrt(3)(x - S) => sqrt(3)x + y = S*sqrt(3)
    #    So: sqrt(3)x + y <= S*sqrt(3)
    #    And by symmetry:
    #    -sqrt(3)x + y <= S*sqrt(3)
    #    sqrt(3)x - y <= S*sqrt(3)
    #    -sqrt(3)x - y <= S*sqrt(3)
    
    # However, the INNER hexagons also have size. We must ensure the ENTIRE inner hexagon is inside the outer one.
    # This is equivalent to checking the 6 vertices of each inner hexagon.
    # Since the inner hexagons are also flat-top and aligned, we can just check their vertices.
    
    # Vertices of a unit flat-top hexagon centered at (cx, cy):
    # (cx+1, cy), (cx+0.5, cy+sqrt(3)/2), (cx-0.5, cy+sqrt(3)/2), (cx-1, cy), (cx-0.5, cy-sqrt(3)/2), (cx+0.5, cy-sqrt(3)/2)
    
    max_s = 0.0
    for cx, cy in inner_points:
        verts = [
            (cx + 1.0, cy),
            (cx + 0.5, cy + math.sqrt(3)/2),
            (cx - 0.5, cy + math.sqrt(3)/2),
            (cx - 1.0, cy),
            (cx - 0.5, cy - math.sqrt(3)/2),
            (cx + 0.5, cy - math.sqrt(3)/2)
        ]
        
        for vx, vy in verts:
            # Check constraints for outer hexagon of side S
            # 1. |vy| <= S * sqrt(3) / 2  => S >= 2 * |vy| / sqrt(3)
            s1 = 2.0 * abs(vy) / math.sqrt(3)
            
            # 2. |vx| <= S => S >= |vx|
            s2 = abs(vx)
            
            # 3. sqrt(3)*vx + vy <= S*sqrt(3) => S >= (sqrt(3)*vx + vy) / sqrt(3) = vx + vy/sqrt(3)
            # Also need to handle the negative side? The constraint is for the specific edge.
            # The 4 slanted constraints are:
            # sqrt(3)*vx + vy <= S*sqrt(3)  => S >= vx + vy/sqrt(3)
            # -sqrt(3)*vx + vy <= S*sqrt(3) => S >= -vx + vy/sqrt(3)
            # sqrt(3)*vx - vy <= S*sqrt(3)  => S >= vx - vy/sqrt(3)
            # -sqrt(3)*vx - vy <= S*sqrt(3) => S >= -vx - vy/sqrt(3)
            
            s3 = vx + vy / math.sqrt(3)
            s4 = -vx + vy / math.sqrt(3)
            s5 = vx - vy / math.sqrt(3)
            s6 = -vx - vy / math.sqrt(3)
            
            # The required S for this vertex is the max of all these lower bounds
            req_s = max(s1, s2, s3, s4, s5, s6)
            if req_s > max_s:
                max_s = req_s
                
    # Add a tiny epsilon to avoid floating point boundary issues
    outer_side = max_s + 1e-6
    
    # Return format: inner rows (x, y, angle_degrees)
    inner_list = []
    for x, y in inner_points:
        inner_list.append([x, y, 0.0])
        
    return inner_list, (0.0, 0.0), outer_side, 0.0
# EVOLVE-BLOCK-END
