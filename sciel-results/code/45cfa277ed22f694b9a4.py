import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define 12 unit hexagons in axial coordinates (q, r)
    # Converted to Cartesian (x, y) for a flat-topped hexagon grid
    # x = 1.5 * q + 0.75 * r
    # y = sqrt(3)/2 * r
    
    axial_coords = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (1, -1), (-1, 1),
        (2, 0), (-2, 0),
        (0, 2),
        (1, 1)
    ]
    
    # Ensure we have exactly 12
    assert len(axial_coords) == 12
    
    inner = []
    for q, r in axial_coords:
        x = 1.5 * q + 0.75 * r
        y = math.sqrt(3) / 2 * r
        inner.append([x, y, 0.0])
        
    # Calculate the minimum side length L of the outer hexagon
    # Outer hexagon is centered at (0,0) and aligned (angle 0)
    # Constraints for a flat-topped hexagon of side L:
    # 1. |x| <= L
    # 2. |sqrt(3)*x + y| <= sqrt(3)*L
    # 3. |sqrt(3)*x - y| <= sqrt(3)*L
    
    # For each inner hexagon, we must check all its vertices.
    # Vertices of a unit flat-topped hexagon at (0,0):
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    
    verts = [
        (1.0, 0.0),
        (0.5, math.sqrt(3)/2),
        (-0.5, math.sqrt(3)/2),
        (-1.0, 0.0),
        (-0.5, -math.sqrt(3)/2),
        (0.5, -math.sqrt(3)/2)
    ]
    
    max_L = 0.0
    for cx, cy, _ in inner:
        for vx, vy in verts:
            px = cx + vx
            py = cy + vy
            
            # Calculate L required for this point
            L1 = abs(px)
            L2 = abs(math.sqrt(3)*px + py) / math.sqrt(3)
            L3 = abs(math.sqrt(3)*px - py) / math.sqrt(3)
            
            required_L = max(L1, L2, L3)
            if required_L > max_L:
                max_L = required_L
                
    # Add a tiny epsilon to avoid floating point boundary issues
    outer_side = max_L + 1e-6
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
