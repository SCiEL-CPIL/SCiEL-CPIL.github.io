import math

def construct():
    # Generate candidate points in a sufficiently large area
    # The 32nd closest point will be at distance squared <= 9 (Ring 3)
    # Ring 3 has max coordinate 3, so a range of -5 to 5 is safe
    candidates = []
    for a in range(-5, 6):
        for b in range(-5, 6):
            # Squared distance in triangular lattice: a^2 + ab + b^2
            d2 = a*a + a*b + b*b
            candidates.append((d2, a, b))
    
    # Sort by distance, then by a, then by b to ensure determinism
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    
    # Take the first 32 points
    selected = [ (a, b) for d2, a, b in candidates[:32] ]
    
    # Convert to list of lists as required
    return [[a, b] for a, b in selected]
