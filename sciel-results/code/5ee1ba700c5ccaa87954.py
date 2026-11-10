# EVOLVE-BLOCK-START
import numpy as np
import time

def kissing_number11() -> np.ndarray:
    """
    Returns a set of 11-dimensional integer vectors with minimum pairwise distance >= max norm.
    Uses the 220 minimal vectors of the D11 lattice.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    d = 11
    
    # Construct the 220 minimal vectors of the D11 lattice
    # These are vectors in Z^11 with sum of coordinates even and L2 norm squared 2.
    # There are 220 such vectors.
    
    points = []
    
    # Type 1: Vectors with two non-zero entries (+1, +1) or (+1, -1) or (-1, +1) or (-1, -1)
    # There are C(11, 2) = 55 pairs of indices.
    # For each pair, there are 4 sign combinations.
    # Total = 55 * 4 = 220.
    # Wait, let's check the norm.
    # (1, 1) -> norm^2 = 2. Sum = 2 (even).
    # (1, -1) -> norm^2 = 2. Sum = 0 (even).
    # (-1, 1) -> norm^2 = 2. Sum = 0 (even).
    # (-1, -1) -> norm^2 = 2. Sum = -2 (even).
    
    # All these vectors have pairwise distance squared:
    # If they share an index, e.g., (1,1,0..0) and (1,-1,0..0) (at indices 0,1):
    # Diff: (0, 2, 0..) -> norm^2 = 4.
    # If they don't share an index, e.g., (1,1,0..) and (1,1,0..) at different indices? No, distinct pairs.
    # e.g., (1,1,0..) at (0,1) and (1,1,0..) at (2,3).
    # Diff: (-1,-1,1,1,0..) -> norm^2 = 4.
    
    # So min distance squared is 4.
    # Max norm squared is 2.
    # Condition: min_dist >= max_norm. 4 >= 2. True.
    
    for i in range(d):
        for j in range(i+1, d):
            # Generate 4 vectors for indices i, j
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    points.append(v)
    
    # Total points should be 220
    # Let's double check if there are other vectors in D11 with norm^2 2.
    # D11 = {x in Z^11 | sum x_i is even}.
    # Norm^2 = 2 implies exactly two non-zero entries, each +/-1.
    # So yes, this is the complete set.
    
    points_array = np.array(points, dtype=np.int64)
    
    # Validate
    k = points_array.shape[0]
    if k < 2:
        return points_array
        
    # Check no zero vector
    # All vectors have norm^2 2, so no zero vector.
    
    # Check min distance >= max norm
    # Max norm squared is 2.
    # Min distance squared is 4.
    # 4 >= 2.
    
    return points_array

# EVOLVE-BLOCK-END
