import math
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Define the 11 hexagon positions
    # Using a compact arrangement that fits in a side-5 hexagon
    # 3x3 grid (8 hexes) + 3 more
    
    positions = []
    # Row 1 (y = 0)
    positions.append((-2.0, 0.0))
    positions.append((0.0, 0.0))
    positions.append((2.0, 0.0))
    
    # Row 2 (y = sqrt(3))
    positions.append((-1.0, math.sqrt(3)))
    positions.append((1.0, math.sqrt(3)))
    
    # Row 3 (y = 2*sqrt(3))
    positions.append((-2.0, 2.0 * math.sqrt(3)))
    positions.append((0.0, 2.0 * math.sqrt(3)))
    positions.append((2.0, 2.0 * math.sqrt(3)))
    
    # Add 3 more to reach 11
    # Place them to stay within radius 5
    positions.append((4.0, 0.0))
    positions.append((-4.0, 0.0))
    positions.append((0.0, 4.0 * math.sqrt(3) * 0.5)) # Approx 3.46, dist 3.46
    
    # Verify count
    assert len(positions) == N, f"Expected {N} positions, got {len(positions)}"
    
    # Create inner list: [x, y, angle]
    inner = [[x, y, 0.0] for x, y in positions]
    
    # Outer hexagon center, side length, rotation
    center = (0.0, 0.0)
    side = 5.0
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
