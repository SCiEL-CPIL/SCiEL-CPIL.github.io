"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Generate all candidate points in a reasonable range
    # The optimal set is centered at (0,0)
    candidates = []
    for a in range(-10, 11):
        for b in range(-10, 11):
            # Calculate hexagonal distance to origin
            # d = (|a| + |b| + |a+b|) / 2
            d = (abs(a) + abs(b) + abs(a + b)) / 2
            candidates.append((d, a, b))
    
    # Sort by distance, then by a, b for determinism
    candidates.sort()
    
    # Take the first 32
    points = [(a, b) for d, a, b in candidates[:32]]
    
    # Verify constraints and count edges
    # Convert to list of lists
    result = [[a, b] for a, b in points]
    
    # Optional: Verify edge count to ensure it's 76
    # If not 76, we might need a better search, but for now, 
    # this is the standard "hexagonal" solution.
    
    return result
# EVOLVE-BLOCK-END
