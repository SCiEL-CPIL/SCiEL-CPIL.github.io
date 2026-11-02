import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # 3-row packing: 5, 3, 3
    # Row 0 (y=0): 5 hexagons
    # Row 1 (y=1.5): 3 hexagons
    # Row 2 (y=3.0): 3 hexagons
    
    sqrt3 = math.sqrt(3)
    inner = []
    
    # Row 0: 5 hexagons, centered
    # Centers: -2, -1, 0, 1, 2 (times sqrt3)
    for i in range(-2, 3):
        inner.append([i * sqrt3, 0.0, 0.0])
        
    # Row 1: 3 hexagons, staggered
    # Centers: -1.5, -0.5, 0.5 (times sqrt3)
    for i in [-1.5, -0.5, 0.5]:
        inner.append([i * sqrt3, 1.5, 0.0])
        
    # Row 2: 3 hexagons, staggered back
    # Centers: -1, 0, 1 (times sqrt3)
    for i in range(-1, 2):
        inner.append([i * sqrt3, 3.0, 0.0])
        
    return inner, (0.0, 0.0), 5.0, 0.0
# EVOLVE-BLOCK-END
