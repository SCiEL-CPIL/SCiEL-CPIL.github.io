"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    """Return an edge-attractiveness matrix for ACO TSP.
    
    Uses inverse-fifth distance as the heuristic. This provides an extremely
    strong guidance toward short edges. Empirical results show that increasing
    the exponent (sharpening the heuristic) from 1 (seed) to 2 to 3 has
    consistently improved the score (1.0 -> 1.25 -> 1.34). This suggests that
    for this specific ACO implementation and instance set, a very deterministic
    preference for short edges helps the ants converge to better tours, likely
    by mimicking a strong Nearest Neighbor search which is a good baseline for
    uniform random points. We push this further with exponent 5.
    """
    start_time = time.time()
    
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    # Safety check for deadline
    if time.time() - start_time > 290:
        return 1.0 / (d + 1e-9)

    # Replace diagonal (self-loops) with a large value to make their heuristic ~0
    d_safe = d.copy()
    np.fill_diagonal(d_safe, 1e10)
    
    # Inverse fifth power distance: even sharper contrast
    # This makes short edges overwhelmingly more attractive than long ones.
    eta = 1.0 / (d_safe ** 5 + 1e-18)
    
    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
