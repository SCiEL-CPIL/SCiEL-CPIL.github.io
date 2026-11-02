import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Calculate the smallest integer k such that 3k(k-1) + 1 >= N
    # This k represents the "radius" of the hexagonal cluster in terms of hexagons per side.
    # The side length of the outer hexagon will be 2k - 1.
    
    # Solve 3k^2 - 3k + 1 - N >= 0
    # k = (3 + sqrt(9 - 12(1-N))) / 6 = (3 + sqrt(12N - 3)) / 6
    k = math.ceil((3 + math.sqrt(12 * N - 3)) / 6)
    
    # Ensure k is at least 1
    k = max(1, k)
    
    # Generate axial coordinates (q, r) for a hexagon of radius k-1
    # The condition for a hexagon of radius R (center at 0,0) is max(|q|, |r|, |q+r|) <= R
    R = k - 1
    
    coords = []
    # Iterate through a bounding box of the axial coordinates
    # q ranges from -R to R
    # r ranges from -R to R
    for q in range(-R, R + 1):
        for r in range(-R, R + 1):
            if max(abs(q), abs(r), abs(q + r)) <= R:
                coords.append((q, r))
    
    # Sort coordinates to make the selection deterministic (e.g., by distance from center, then angle)
    # This ensures we pick the most central hexagons first if N < total capacity
    def sort_key(coord):
        q, r = coord
        # Distance squared in axial coords is roughly q^2 + r^2 + qr
        dist = q*q + r*r + q*r
        # Angle for tie-breaking
        x = 1.5 * q
        y = math.sqrt(3) * (r + 0.5 * q)
        angle = math.atan2(y, x)
        return (dist, angle)
        
    coords.sort(key=sort_key)
    
    # Take the first N coordinates
    selected_coords = coords[:N]
    
    # Convert axial (q, r) to Cartesian (x, y) for flat-top hexagons
    # x = 1.5 * q
    # y = sqrt(3) * (r + 0.5 * q)
    inner = []
    for q, r in selected_coords:
        x = 1.5 * q
        y = math.sqrt(3) * (r + 0.5 * q)
        # Angle is 0 for all, as they are aligned
        inner.append([x, y, 0.0])
        
    # Outer hexagon parameters
    center = (0.0, 0.0)
    side = 2 * k - 1
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
