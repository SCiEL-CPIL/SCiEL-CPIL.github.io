import numpy as np

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    
    # The conflict graph is the tensor product of G_5 and G_41.
    # G_5 is a 5-cycle (edges for diff 1, 4). Alpha(G_5) = 2.
    # G_41 is the Paley graph of order 41. Alpha(G_41) ~ 6.
    # We can pick an independent set U in G_5 and take ALL vertices in G_41 for each u in U.
    # This yields an independent set of size |U| * 41 = 2 * 41 = 82.
    
    # Independent set in G_5 (mod 5): {0, 2}
    # Check: 0-2 = -2 = 3 (not in {1,4}), 2-0 = 2 (not in {1,4}).
    # So {0, 2} is independent.
    
    U = {0, 2}
    
    result = []
    for x in range(m):
        if x % 5 in U:
            result.append(x)
            
    return np.array(result, dtype=np.int64)
