import math

def construct():
    # Generate points in a spiral order
    # We need 32 points.
    # Hexagon of radius 3 has 37 points.
    # So we generate up to radius 3.
    
    # Simpler: Just generate all points in a box, sort by graph distance, then by angle.
    
    R = 10
    candidates = []
    sqrt3_2 = math.sqrt(3) / 2
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            d = (abs(a) + abs(b) + abs(a + b)) // 2
            # Angle approximation
            # x = a + b/2, y = b * sqrt(3)/2
            # angle = atan2(y, x)
            x = a + 0.5 * b
            y = sqrt3_2 * b
            angle = math.atan2(y, x)
            candidates.append((d, angle, a, b))
            
    candidates.sort(key=lambda x: (x[0], x[1]))
    
    selected = []
    for d, angle, a, b in candidates:
        selected.append([a, b])
        if len(selected) == 32:
            break
            
    return selected
