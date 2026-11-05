import math

def construct():
    n = 32
    # Generate candidate points
    # A hexagon of radius 3 has 37 points. We need 32.
    # So we need points with hex distance <= 3.
    candidates = []
    for a in range(-10, 11):
        for b in range(-10, 11):
            # Hex distance formula for axial coordinates (a, b)
            # d = (|a| + |b| + |a+b|) / 2
            d = (abs(a) + abs(b) + abs(a + b)) // 2
            if d <= 3:
                # Calculate a secondary sorting key to order points in the ring
                # This helps in creating a connected, spiral-like shape
                # Using angle for ordering
                angle = math.atan2(b, a)
                candidates.append((d, angle, a, b))
    
    # Sort by distance, then by angle
    candidates.sort(key=lambda x: (x[0], x[1]))
    
    # Select the first n points
    selected = [ [a, b] for d, ang, a, b in candidates[:n] ]
    
    # Ensure we have exactly n points
    if len(selected) < n:
        # Fallback: expand range if somehow not enough (should not happen for n=32)
        # But let's add a safety check
        pass
        
    return selected
