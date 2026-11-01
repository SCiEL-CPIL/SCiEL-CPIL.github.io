import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Optimal packing for 11 unit hexagons in a regular hexagon
    # Uses a central hexagon, 6 surrounding it (layer 1), and 4 from layer 2.
    # This configuration has a maximum radial distance of 2*sqrt(3) + 1 from the center.
    # Aligning the outermost vertices with the container vertices minimizes the side length L.
    
    sqrt3 = math.sqrt(3)
    side_length = 2 * sqrt3 + 1
    
    # Center
    inner = [[0.0, 0.0, 0.0]]
    
    # Layer 1: 6 hexagons at distance sqrt(3)
    for i in range(6):
        angle = i * 60.0
        rad = math.radians(angle)
        x = sqrt3 * math.cos(rad)
        y = sqrt3 * math.sin(rad)
        inner.append([x, y, 0.0])
        
    # Layer 2: 4 hexagons at distance 2*sqrt(3)
    # We pick 4 out of the 6 possible directions to keep the cluster compact.
    # Directions: 0, 60, 120, 180, 240, 300
    # Picking 0, 120, 240, 60 (arbitrary 4, symmetry is not strictly required for the bounding box if max radius is the same)
    # Actually, to be safe and symmetric, let's pick 0, 120, 240 and 180? 
    # Or just the first 4: 0, 60, 120, 180.
    layer2_angles = [0.0, 60.0, 120.0, 180.0]
    for angle in layer2_angles:
        rad = math.radians(angle)
        x = 2 * sqrt3 * math.cos(rad)
        y = 2 * sqrt3 * math.sin(rad)
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), side_length, 0.0
# EVOLVE-BLOCK-END
