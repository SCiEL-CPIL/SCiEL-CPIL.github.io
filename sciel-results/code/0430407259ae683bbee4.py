import numpy as np

def construct():
    # 12 unit hexagons
    # Center
    centers = [(0.0, 0.0)]
    
    # Ring 1 (6 hexagons)
    # Distance 2 to touch center hexagon (aligned)
    r1 = 2.0
    for i in range(6):
        angle = i * np.pi / 3
        centers.append((r1 * np.cos(angle), r1 * np.sin(angle)))
        
    # Ring 2 (5 hexagons)
    # Distance 2*sqrt(3) to touch Ring 1 hexagons
    r2 = 2.0 * np.sqrt(3)
    # Angles: 30, 90, 150, 210, 270 degrees (skip 330)
    # These positions touch the Ring 1 hexagons at 0, 60, 120, 180, 240 degrees
    angles = [np.pi/6, np.pi/2, 5*np.pi/6, 7*np.pi/6, 3*np.pi/2]
    for angle in angles:
        centers.append((r2 * np.cos(angle), r2 * np.sin(angle)))
        
    inner = []
    for x, y in centers:
        inner.append([x, y, 0.0])
        
    # Outer hexagon
    # To minimize side length, rotate outer hexagon by 30 degrees
    # Side length calculation:
    # Max x-projection is 4 (from hex at 30 deg position, vertex at x=4)
    # Apothem A = S * sqrt(3) / 2
    # 4 <= S * sqrt(3) / 2 => S >= 8 / sqrt(3)
    side = 8.0 / np.sqrt(3)
    
    return inner, (0.0, 0.0), side, 30.0
