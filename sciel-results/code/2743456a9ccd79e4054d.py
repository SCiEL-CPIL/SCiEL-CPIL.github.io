# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates.
    
    Strategy: Use the 2E8 ⊕ 2D3 lattice construction.
    - E8 lattice (8 dims): 240 minimal vectors (roots), scaled by 2 to have norm^2 = 8.
    - D3 lattice (3 dims): 12 minimal vectors (roots), scaled by 2 to have norm^2 = 8.
    
    The E8 roots in Z^8 are:
    1. (±1, ±1, 0, 0, 0, 0, 0, 0) : 112 vectors. Scaled by 2: (±2, ±2, 0, ...) norm^2=8.
    2. (±1/2, ..., ±1/2) with even minus signs : 128 vectors. Scaled by 4 to be integer: (±2, ..., ±2) norm^2=64? No.
       Wait. The standard E8 roots have norm^2=2.
       Type 1: (±1, ±1, 0, ...) -> norm^2 = 2.
       Type 2: (±1/2, ..., ±1/2) -> norm^2 = 8*(1/4) = 2.
       
       To get integer coordinates, we can scale the entire E8 lattice by 2.
       Then:
       Type 1 becomes (±2, ±2, 0, ...) -> norm^2 = 8.
       Type 2 becomes (±1, ±1, ..., ±1) -> norm^2 = 8.
       
       So the 240 scaled E8 vectors are:
       A. (±2, ±2, 0, 0, 0, 0, 0, 0) : 112 vectors.
       B. (±1, ±1, ±1, ±1, ±1, ±1, ±1, ±1) : 128 vectors (even number of minus signs).
       
    - D3 lattice (3 dims): 12 minimal vectors.
      D3 roots are (±1, ±1, 0) permutations.
      Scaled by 2: (±2, ±2, 0) permutations.
      Count: 3 * 4 = 12 vectors. Norm^2 = 4+4 = 8.
      
    Embedding in 11D:
    First 8 dims: E8 part.
    Last 3 dims: D3 part.
    
    Points are either:
    1. (v_e8, 0) where v_e8 is a scaled E8 vector.
    2. (0, v_d3) where v_d3 is a scaled D3 vector.
    
    Total points: 240 + 12 = 252.
    
    Distance Checks:
    - All points have norm^2 = 8. So max_norm^2 = 8.
    - We need min_dist^2 >= 8.
    
    Case 1: Two points from E8 part.
    dist^2 = ||v1 - v2||^2.
    For E8 lattice scaled by 2, the minimum distance between distinct roots is 2*sqrt(2) ?
    Original E8 roots have min dist^2 = 2 (e.g., (1,1,0..) and (1,-1,0..) -> dist^2 = 4? No.
    (1,1,0..) and (-1,1,0..) -> dist^2 = 4.
    (1,1,0..) and (1,0,1..) -> dist^2 = 2.
    So min dist^2 in E8 is 2.
    Scaled by 2, min dist^2 becomes 4.
    Wait. If min dist^2 = 4, then 4 < 8. Condition fails.
    
    Let's re-evaluate.
    Condition: min_dist^2 >= max_norm^2.
    If we use norm^2 = 8, we need min_dist^2 >= 8.
    In E8, min distance between distinct roots is sqrt(2). Squared is 2.
    If we scale by factor S, norm^2 = 2*S^2, min_dist^2 = 2*S^2.
    So min_dist^2 = norm^2. The condition min_dist^2 >= max_norm^2 holds with equality.
    
    So, for E8 scaled by 2:
    Norm^2 = 2 * 4 = 8.
    Min dist^2 = 2 * 4 = 8.
    This works!
    
    Let's verify the integer coordinates for scaled E8 roots.
    Original Roots:
    R1: (±1, ±1, 0, 0, 0, 0, 0, 0)
    R2: (±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2) with even minus signs.
    
    Scale by 2:
    S1: (±2, ±2, 0, 0, 0, 0, 0, 0) -> Integer. Norm^2 = 8.
    S2: (±1, ±1, ±1, ±1, ±1, ±1, ±1, ±1) with even minus signs. -> Integer. Norm^2 = 8.
    
    Are there any other roots? No, these are all 240.
    
    For D3 scaled by 2:
    Original D3 roots: (±1, ±1, 0) perms. Norm^2 = 2.
    Scale by 2: (±2, ±2, 0) perms. Norm^2 = 8.
    Min dist in D3?
    (1,1,0) and (1,-1,0) -> dist^2 = 4.
    (1,1,0) and (-1,1,0) -> dist^2 = 4.
    (1,1,0) and (1,0,1) -> dist^2 = 2.
    So min dist^2 = 2.
    Scaled by 2, min dist^2 = 8.
    Norm^2 = 8.
    Condition holds.
    
    Cross distances:
    Point p1 = (v_e8, 0)
    Point p2 = (0, v_d3)
    dist^2 = ||v_e8||^2 + ||v_d3||^2 = 8 + 8 = 16.
    16 >= 8. Holds.
    
    So the set of 252 points is valid.
    """
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    d = 11
    points_list = []
    
    # --- Generate 240 Scaled E8 Vectors (8 dims) ---
    # Type 1: (±2, ±2, 0, 0, 0, 0, 0, 0)
    for i, j in combinations(range(8), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                v = np.zeros(8, dtype=np.int64)
                v[i] = 2 * si
                v[j] = 2 * sj
                # Embed in 11D: first 8 dims, last 3 are 0
                p = np.zeros(d, dtype=np.int64)
                p[:8] = v
                points_list.append(p)
                
    # Type 2: (±1, ±1, ±1, ±1, ±1, ±1, ±1, ±1) with even number of minus signs
    # 128 such vectors.
    # Iterate over all 2^8 sign combinations, keep those with even number of -1s.
    for signs in product([1, -1], repeat=8):
        if signs.count(-1) % 2 == 0:
            v = np.array(signs, dtype=np.int64)
            p = np.zeros(d, dtype=np.int64)
            p[:8] = v
            points_list.append(p)
            
    # --- Generate 12 Scaled D3 Vectors (3 dims) ---
    # (±2, ±2, 0) permutations in last 3 dims (indices 8, 9, 10)
    # Indices: 8, 9, 10
    for i, j in combinations([8, 9, 10], 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = 2 * si
                p[j] = 2 * sj
                points_list.append(p)
                
    if time.time() > deadline:
         return np.array(points_list[:2], dtype=np.int64)

    points = np.array(points_list, dtype=np.int64)
    
    # Optional: Quick validation to be safe (though logic is sound)
    # k = len(points)
    # min_d2 = float('inf')
    # max_n2 = 0
    # for p in points:
    #     n2 = np.dot(p, p)
    #     if n2 > max_n2: max_n2 = n2
    #     if n2 == 0: return np.array(points[:2]) # Should not happen
    # for i in range(k):
    #     for j in range(i+1, k):
    #         diff = points[i] - points[j]
    #         d2 = np.dot(diff, diff)
    #         if d2 < min_d2: min_d2 = d2
    #     if time.time() > deadline: break
    # if min_d2 < max_n2:
    #     # Fallback to D11 if somehow invalid (should not happen)
    #     pass
    
    return points

# EVOLVE-BLOCK-END
