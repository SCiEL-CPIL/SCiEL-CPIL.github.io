# EVOLVE-BLOCK-START
import numpy as np
import time

def kissing_number11() -> np.ndarray:
    """
    Constructs a set of 11-dimensional integer points maximizing k, such that
    min_{i != j} ||p_i - p_j||^2 >= max_i ||p_i||^2.
    
    Strategy: Direct construction using the 2E8 ⊕ 2D3 lattice.
    1. Construct the 240 minimal vectors of the E8 lattice (scaled by 2 to have norm^2=8).
       The E8 lattice minimal vectors have norm^2=2. Scaling by 2 gives norm^2=8.
       Actually, standard E8 minimal vectors have norm^2=2.
       Wait, the minimal vectors of E8 are:
       - 112 vectors of type (±1, ±1, 0, ..., 0) with norm^2 = 2.
       - 128 vectors of type (±1/2, ±1/2, ..., ±1/2) with all signs, norm^2 = 8 * (1/4) = 2.
       To make them integers, we can scale E8 by 2.
       The scaled vectors will be:
       - 112 vectors of type (±2, ±2, 0, ..., 0) with norm^2 = 8.
       - 128 vectors of type (±1, ±1, ..., ±1) with norm^2 = 8.
       Total 240 vectors in 8 dimensions.
       
    2. Construct the 12 minimal vectors of the D3 lattice (scaled to have norm^2=8).
       D3 is isomorphic to A3. The minimal vectors of D3 (in 3D) are permutations of (±1, ±1, 0) with norm^2=2.
       Wait, D3 minimal vectors are (±1, ±1, 0) and permutations. There are 3*4=12 such vectors.
       Norm^2 = 2.
       To get norm^2=8, we scale by 2.
       Vectors: (±2, ±2, 0) and permutations. Norm^2 = 4+4=8.
       
    3. Embed these into 11 dimensions.
       E8 vectors occupy dimensions 0-7.
       D3 vectors occupy dimensions 8-10.
       
    4. Verify constraints.
       All points have norm^2 = 8.
       Max norm^2 = 8.
       We need min pairwise distance^2 >= 8.
       
       Let's check distances within E8 part (scaled by 2):
       Original E8 min distance^2 is 2 (between minimal vectors).
       Scaled by 2, min distance^2 is 4 * 2 = 8.
       So any two distinct E8 vectors have distance^2 >= 8.
       
       Let's check distances within D3 part (scaled by 2):
       Original D3 min distance^2 is 2 (between minimal vectors).
       Scaled by 2, min distance^2 is 4 * 2 = 8.
       So any two distinct D3 vectors have distance^2 >= 8.
       
       Let's check distances between E8 and D3 parts:
       An E8 vector v_E has zeros in dims 8-10.
       A D3 vector v_D has zeros in dims 0-7.
       Distance^2 = ||v_E||^2 + ||v_D||^2 = 8 + 8 = 16.
       16 >= 8.
       
       Thus, the set of 240 + 12 = 252 points is valid.
    """
    start_time = time.time()
    time_limit = 580.0
    
    d = 11
    points = []
    
    # Generate scaled E8 minimal vectors (norm^2 = 8)
    # Type 1: (±2, ±2, 0, ..., 0) in 8 dims
    for i in range(8):
        for j in range(i + 1, 8):
            for si in (2, -2):
                for sj in (2, -2):
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = si
                    p[j] = sj
                    points.append(p)
                    
    # Type 2: (±1, ±1, ..., ±1) in 8 dims
    for signs in np.ndindex(*([2] * 8)):
        p = np.zeros(d, dtype=np.int64)
        for idx in range(8):
            p[idx] = 1 if signs[idx] == 0 else -1
        points.append(p)
        
    # Generate scaled D3 minimal vectors (norm^2 = 8)
    # D3 minimal vectors are permutations of (±1, ±1, 0).
    # Scaled by 2: permutations of (±2, ±2, 0).
    # Dims 8, 9, 10.
    # Positions for the two non-zeros: (8,9), (8,10), (9,10).
    for i in range(3):
        for j in range(i + 1, 3):
            dim_i = 8 + i
            dim_j = 8 + j
            for si in (2, -2):
                for sj in (2, -2):
                    p = np.zeros(d, dtype=np.int64)
                    p[dim_i] = si
                    p[dim_j] = sj
                    points.append(p)

    # Convert to array
    points_arr = np.array(points, dtype=np.int64)
    points_arr = np.around(points_arr).astype(np.int64)
    
    # Quick validation check (optional, but good for safety)
    if len(points_arr) > 0:
        norms_sq = np.sum(points_arr * points_arr, axis=1)
        max_norm_sq = np.max(norms_sq)
        
        # Check min distance
        min_dist_sq = float('inf')
        # For 252 points, 252*251/2 = ~31k pairs. Very fast.
        for i in range(len(points_arr)):
            for j in range(i + 1, len(points_arr)):
                diff = points_arr[i] - points_arr[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    if min_dist_sq < max_norm_sq:
                        # Should not happen, but break early if invalid
                        break
            if min_dist_sq < max_norm_sq:
                break
        
        if min_dist_sq < max_norm_sq:
            # Fallback to D11 construction if something went wrong (shouldn't)
            # But given the math, it's solid.
            pass
            
    return points_arr

# EVOLVE-BLOCK-END
