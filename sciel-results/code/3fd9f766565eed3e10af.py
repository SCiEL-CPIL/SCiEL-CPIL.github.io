# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
import time

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates such that
    the minimum pairwise squared distance is >= the maximum squared norm.
    
    Strategy:
    Decompose R^11 into R^8 (E8 lattice) and R^3 (A3 lattice).
    Use the roots of E8 (240 vectors) and roots of A3 (12 vectors).
    Standard roots have squared norm 2.
    We scale all roots by 2 to achieve a uniform squared norm of 8.
    
    Let S be the set of points.
    Points from E8 part: p = (2 * r_E8, 0, 0, 0) where r_E8 is a root of E8.
    Points from A3 part: q = (0, 0, 0, 0, 2 * r_A3) where r_A3 is a root of A3.
    
    Norms:
    ||p||^2 = 4 * ||r_E8||^2 = 4 * 2 = 8.
    ||q||^2 = 4 * ||r_A3||^2 = 4 * 2 = 8.
    Max squared norm = 8.
    
    Distances:
    1. Within E8: ||p1 - p2||^2 = 4 * ||r1 - r2||^2.
       For E8 roots, min ||r1 - r2||^2 is 2 (when angle is 90 deg, dot=0, dist^2=2+2=4? No.
       Roots have norm 2. ||r1-r2||^2 = 2+2-2 r1.r2.
       Possible dot products for E8 roots: -1, 0, 1.
       If dot=1 (same root or opposite? No, distinct roots).
       In E8, min distance between distinct roots is sqrt(2) if dot=0? 
       Wait. ||r1-r2||^2 = 4 - 2 r1.r2.
       If r1, r2 are distinct roots:
       If r1.r2 = 1, they are the same root (since norm is 2, cos theta = 1/2? No.
       In E8, roots have norm 2. Angle between roots:
       cos theta = (r1.r2) / (2*2) = r1.r2 / 4.
       Possible integer dot products: -1, 0, 1.
       If r1.r2 = 1, cos theta = 1/4. ||r1-r2||^2 = 4 - 2(1) = 2.
       If r1.r2 = 0, ||r1-r2||^2 = 4.
       If r1.r2 = -1, ||r1-r2||^2 = 6.
       So min ||r1-r2||^2 for distinct roots is 2.
       Scaled by 4: min dist^2 = 4 * 2 = 8.
       This is >= max norm^2 (8). OK.
       
    2. Within A3:
       A3 roots are of form (1, -1, 0) and perms, with signs.
       Norm^2 = 2.
       Distinct roots:
       If they share a coordinate with same sign, dot can be 1?
       Example: (1, -1, 0) and (1, 0, -1). Dot = 1.
       ||r1-r2||^2 = 2+2-2(1) = 2.
       Scaled by 4: min dist^2 = 8. OK.
       
    3. Cross terms (E8 vs A3):
       p = (2*r_E8, 0), q = (0, 2*r_A3).
       ||p-q||^2 = ||p||^2 + ||q||^2 - 2 p.q = 8 + 8 - 0 = 16.
       16 >= 8. OK.
       
    Total points: 240 (E8) + 12 (A3) = 252.
    """
    
    d = 11
    points_list = []
    
    # 1. Generate E8 roots (240 vectors in 8D)
    # E8 roots are:
    # a) All (±1, ±1, 0, 0, 0, 0, 0, 0) -> 2^2 * C(8,2) = 4 * 28 = 112
    # b) All (±1/2, ..., ±1/2) with even number of minus signs -> 2^7 = 128
    # To keep integers, we usually work with the dual or just generate the integer ones first.
    # The problem requires INTEGER coordinates.
    # The standard E8 lattice contains the half-integer vectors.
    # However, the "roots" of E8 include the half-integer ones.
    # If we use the half-integer roots, they are not integers.
    # The task requires: "Entries are rounded with np.around and cast to int64".
    # If we use half-integers, rounding will destroy the lattice structure or make them integers incorrectly.
    # Let's check the constraint: "Integer kissing configuration". "Entries are rounded... cast to int64".
    # This implies the input to the function should be such that rounding gives integers, OR the function outputs integers.
    # The current seed program uses integers (weight 2 vectors).
    # If I use half-integer E8 roots, e.g., (0.5, 0.5, ...), rounding gives (1, 1, ...) which changes the norm.
    # (1,1,0...0) has norm 2. (0.5, 0.5, 1, 1, 1, 1, 1, 1) has norm 8*0.25 = 2.
    # If I round (0.5, 0.5, 1, 1, 1, 1, 1, 1) -> (0 or 1, 0 or 1, 1, 1, 1, 1, 1, 1).
    # np.around(0.5) is 0.0 (banker's rounding) or 1? In numpy, around(0.5) is 0.0. around(1.5) is 2.0.
    # This is dangerous. The task says "Integer kissing configuration". It likely expects the final array to be integers.
    # The seed program uses only integer vectors.
    # The "240 roots of E8" includes 128 half-integer vectors.
    # Can we use ONLY the 112 integer roots of E8?
    # The 112 integer roots are of the form (±1, ±1, 0, ..., 0).
    # These are exactly the weight-2 vectors in 8 dimensions.
    # If we take the 112 E8 integer roots and embed them in 11D, we have 112 points.
    # We can also take the 12 A3 roots (which are integer: (1, -1, 0) etc) and embed them in the last 3 dims.
    # Total = 112 + 12 = 124. This is LESS than 220.
    
    # So, to beat 220, we MUST use the half-integer E8 roots somehow, or find a different lattice.
    # BUT the output MUST be integers.
    # If we use half-integers, we cannot just round them arbitrarily.
    # However, notice the scaling factor.
    # If we scale the E8 roots by 2, the half-integer roots become integers!
    # Root r_E8 (norm 2).
    # If r_E8 is half-integer, 2*r_E8 is integer.
    # If r_E8 is integer, 2*r_E8 is integer (even).
    # So, let's scale ALL E8 roots by 2.
    # New vectors v_E8 = 2 * r_E8.
    # Norm^2(v_E8) = 4 * Norm^2(r_E8) = 4 * 2 = 8.
    # These are all integers.
    # Count: 240 points.
    
    # Similarly for A3:
    # A3 roots are integer (±1, ±1, 0). Norm 2.
    # Scale by 2: v_A3 = 2 * r_A3.
    # Norm^2(v_A3) = 8.
    # These are all integers.
    # Count: 12 points.
    
    # Total points: 240 + 12 = 252.
    
    # Let's verify the integer nature.
    # E8 half-integer roots: (±0.5, ±0.5, ±0.5, ±0.5, ±0.5, ±0.5, ±0.5, ±0.5) with even minuses.
    # Multiplied by 2: (±1, ±1, ..., ±1) with even minuses.
    # These are integers. Norm^2 = 8 * 1 = 8.
    
    # E8 integer roots: (±1, ±1, 0, ..., 0).
    # Multiplied by 2: (±2, ±2, 0, ..., 0).
    # These are integers. Norm^2 = 4 + 4 = 8.
    
    # So the plan holds. We construct 240 scaled E8 roots and 12 scaled A3 roots.
    
    # Generate E8 roots
    e8_roots = []
    
    # Type 1: (±1, ±1, 0, 0, 0, 0, 0, 0)
    for i, j in combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(8, dtype=np.float64)
                v[i] = si
                v[j] = sj
                e8_roots.append(v)
                
    # Type 2: (±0.5, ±0.5, ±0.5, ±0.5, ±0.5, ±0.5, ±0.5, ±0.5) with even number of -0.5s
    # Number of such vectors is 2^7 = 128.
    # We can generate all 2^8 sign combinations and filter for even number of negatives.
    # Or iterate over all subsets of indices to flip.
    base = np.full(8, 0.5)
    for mask in range(256):
        # Count bits set in mask. If even, this corresponds to a valid vector?
        # Let's define the vector based on mask bits.
        # Bit i set -> sign is -1 (or 0.5 vs -0.5).
        # We need even number of negative signs.
        # Let's just generate all 256 and check.
        v = np.zeros(8, dtype=np.float64)
        neg_count = 0
        for i in range(8):
            if (mask >> i) & 1:
                v[i] = -0.5
                neg_count += 1
            else:
                v[i] = 0.5
        if neg_count % 2 == 0:
            e8_roots.append(v)
            
    # Scale E8 roots by 2 and embed in 11D (first 8 dims)
    for r in e8_roots:
        p = np.zeros(11, dtype=np.int64)
        # r is float, 2*r should be integer
        scaled = (2 * r).astype(np.int64)
        p[:8] = scaled
        points_list.append(p)
        
    # Generate A3 roots
    # A3 is the root system of A3, which is the same as the roots of the Lie algebra sl(4).
    # Roots are e_i - e_j for i != j.
    # In 3D orthogonal coordinates (standard embedding), they are:
    # (1, -1, 0), (1, 0, -1), (0, 1, -1) and their negatives.
    # Total 12 roots.
    a3_roots = []
    a3_base = [
        [1, -1, 0],
        [1, 0, -1],
        [0, 1, -1]
    ]
    for r in a3_base:
        a3_roots.append(np.array(r))
        a3_roots.append(-np.array(r))
        
    # Scale A3 roots by 2 and embed in 11D (last 3 dims)
    for r in a3_roots:
        p = np.zeros(11, dtype=np.int64)
        scaled = (2 * r).astype(np.int64)
        p[8:] = scaled
        points_list.append(p)
        
    points = np.array(points_list, dtype=np.int64)
    points = np.around(points).astype(np.int64)
    
    return points

# EVOLVE-BLOCK-END
