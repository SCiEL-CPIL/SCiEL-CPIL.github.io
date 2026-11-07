"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Start time guard
    start_time = time.time()
    
    n = 24
    # Base path edges are (i, i+1) for i in 0..22
    # Vertices 0..23
    # Evens: 0, 2, ..., 22
    # Odds: 1, 3, ..., 23
    
    # Strategy: Connect central even hub (12) to all odd vertices 
    # that are not already adjacent to it in the path.
    # This ensures bipartiteness (triangle-free) and reduces eccentricity 
    # of hub to 1 for odds and 2 for evens, resulting in diameter <= 4.
    
    hub = 12
    added_edges = []
    
    for v in range(n):
        if v == hub:
            continue
        # Check bipartite compatibility: hub is Even, so v must be Odd
        if v % 2 == 1:
            # Check if edge already exists in base path
            # Path edges are (u, u+1). So 12 is connected to 11 and 13.
            if abs(v - hub) != 1:
                added_edges.append([hub, v])
                
    # Validate time limit just in case (should be instantaneous)
    if time.time() - start_time > 50:
         # Fallback if something went wrong (should never happen)
         return []
         
    # Return list of lists of ints
    return added_edges

# EVOLVE-BLOCK-END
