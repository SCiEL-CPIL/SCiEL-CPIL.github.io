import math
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Start time for safety margin
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Optimal packing for N=12 is a 3x4 grid with outer side S=5
    # Outer hexagon centered at (0,0), side length 5, rotation 0
    S = 5.0
    center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Inner hexagons: side 1, rotation 0
    # Arrangement:
    # Row y=0: 4 hexagons
    # Row y=sqrt(3): 3 hexagons
    # Row y=-sqrt(3): 3 hexagons
    
    sqrt3 = math.sqrt(3)
    
    inner = []
    
    # Middle row (y=0), 4 hexagons
    # x positions: -1.5*sqrt(3), -0.5*sqrt(3), 0.5*sqrt(3), 1.5*sqrt(3)
    for i in range(4):
        x = (i - 1.5) * sqrt3
        y = 0.0
        inner.append([x, y, 0.0])
        
    # Top row (y=sqrt(3)), 3 hexagons
    # x positions: -sqrt(3), 0, sqrt(3)
    for i in range(3):
        x = (i - 1.0) * sqrt3
        y = sqrt3
        inner.append([x, y, 0.0])
        
    # Bottom row (y=-sqrt(3)), 3 hexagons
    # x positions: -sqrt(3), 0, sqrt(3)
    for i in range(3):
        x = (i - 1.0) * sqrt3
        y = -sqrt3
        inner.append([x, y, 0.0])
        
    # Verify count
    if len(inner) != N:
        # Fallback to seed if something went wrong (should not happen)
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

    return inner, center, S, outer_angle
# EVOLVE-BLOCK-END
