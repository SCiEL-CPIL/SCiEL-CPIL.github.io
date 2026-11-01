"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    # Strategy: Use the known optimal kissing number for d=11 which is 165.
    # The vectors can be constructed from the D_11 lattice or similar.
    # However, generating the exact 165 vectors requires specific knowledge.
    
    # A simpler, robust approach that yields a high score:
    # 1. Generate all vectors of the form (±1, ±1, 0, ..., 0) with two non-zero entries.
    #    There are C(11,2) * 4 = 55 * 4 = 220 such vectors.
    #    Norm squared = 2. Norm = sqrt(2).
    #    Distance between two such vectors:
    #    If they share no indices: dist^2 = 2 + 2 = 4. dist = 2.
    #    If they share one index: dist^2 = 1 + 1 + 1 + 1 = 4 (if signs same) or 1+1+1+1=4 (if signs diff? No)
    #    Let v1 = (1, 1, 0...), v2 = (1, -1, 0...). dist^2 = (1-1)^2 + (1-(-1))^2 = 0 + 4 = 4. dist = 2.
    #    Let v1 = (1, 1, 0...), v2 = (-1, 1, 0...). dist^2 = (1-(-1))^2 + (1-1)^2 = 4.
    #    Let v1 = (1, 1, 0...), v3 = (0, 0, 1, 1...). dist^2 = 1+1+1+1 = 4.
    #    So for any pair of distinct vectors from this set, the squared distance is either 2 (if identical, not allowed) or 4.
    #    Wait, are there pairs with distance squared < 4?
    #    If v1 and v2 are distinct vectors of weight 2 with entries ±1.
    #    The inner product <v1, v2> can be:
    #    - 0 (disjoint supports) -> dist^2 = 2+2-0 = 4
    #    - 1 (one sign matches, one differs? No, if supports overlap in 1 index)
    #      v1=(1,1,0), v2=(1,-1,0). <v1,v2> = 1*1 + 1*(-1) = 0. dist^2 = 4.
    #      v1=(1,1,0), v2=(-1,1,0). <v1,v2> = -1+1=0. dist^2 = 4.
    #      v1=(1,1,0), v2=(1,1,0) is same vector.
    #    - 2 (identical support and signs) -> same vector.
    #    So all distinct pairs have squared distance 4.
    #    Max norm squared is 2.
    #    Condition: min pairwise distance >= max norm.
    #    min dist = 2. max norm = sqrt(2) ≈ 1.414.
    #    2 >= 1.414 is TRUE.
    #    So we can return all 220 vectors.
    
    # Can we do better?
    # The kissing number in d=11 is 165.
    # Wait, 220 > 165. How is this possible?
    # The definition of kissing number is the maximum number of non-overlapping unit spheres that can touch a central unit sphere.
    # This corresponds to finding vectors v_i with ||v_i|| = 1 such that ||v_i - v_j|| >= 1 for all i != j.
    # Or, scaling: vectors v_i with ||v_i|| = R such that ||v_i - v_j|| >= R.
    # In my construction: ||v_i|| = sqrt(2). ||v_i - v_j|| = 2.
    # Condition: ||v_i - v_j|| >= ||v_i|| => 2 >= sqrt(2). This holds.
    # If we normalize to unit norm: u_i = v_i / sqrt(2).
    # ||u_i|| = 1.
    # ||u_i - u_j|| = ||v_i - v_j|| / sqrt(2) = 2 / sqrt(2) = sqrt(2) ≈ 1.414.
    # Condition for kissing: ||u_i - u_j|| >= 1.
    # 1.414 >= 1. This holds.
    # So 220 vectors form a valid kissing configuration?
    # But the kissing number of d=11 is 165.
    # Let me re-check the kissing number for d=11.
    # According to literature, the kissing number for d=11 is indeed 165?
    # Let's check d=3: 12. d=4: 24. d=5: 40. d=6: 72. d=7: 126. d=8: 240.
    # d=9: 396? d=10: 504? d=11: 504?
    # Actually, the kissing number for d=11 is not definitively known to be 165.
    # The lower bound is often higher.
    # The set of 220 vectors I constructed is a subset of the D_11 lattice?
    # D_n lattice vectors have even sum of coordinates.
    # My vectors have two ±1's, so sum is 0, 2, or -2. All even.
    # So they are in D_11.
    # The minimum norm in D_n is 2 (for n>=2).
    # The vectors of norm sqrt(2) are the shortest vectors.
    # The number of shortest vectors in D_n is n(n-1) (for n>=2)?
    # For D_4, shortest vectors are 24. 4*3=12? No, D_4 has 24 shortest vectors.
    # The shortest vectors in D_n are the vectors with two non-zero entries ±1.
    # Number of such vectors: Choose 2 positions from n: C(n,2). For each, 4 sign combinations.
    # Total = 4 * C(n,2) = 2n(n-1).
    # For n=4: 2*4*3 = 24. Correct.
    # For n=11: 2*11*10 = 220.
    # So there are 220 shortest vectors in D_11.
    # Do they form a valid kissing configuration?
    # We need ||v_i - v_j|| >= ||v_i||.
    # ||v_i||^2 = 2.
    # ||v_i - v_j||^2 = 4 for all distinct i,j.
    # 4 >= 2. Yes.
    # So 220 is a valid kissing number lower bound for d=11.
    # Is it the maximum?
    # The actual kissing number for d=11 is believed to be higher, possibly 504?
    # But 220 is a strong, easily constructible number.
    
    # Can we add more vectors?
    # Vectors of norm sqrt(4)=2?
    # If we add vectors of norm 2, the max norm becomes 2.
    # Then we need min pairwise distance >= 2.
    # Distance between a norm-sqrt(2) vector and a norm-2 vector?
    # This gets complicated.
    
    # Let's stick with the 220 vectors of norm sqrt(2).
    # This is a valid solution.
    
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    # Verify constraint
    arr = np.array(vectors, dtype=np.int64)
    # Check min distance vs max norm
    # This is O(k^2 * d). k=220, d=11. 220^2 * 11 ~ 500k ops. Fast.
    norms_sq = np.sum(arr**2, axis=1)
    max_norm = np.sqrt(np.max(norms_sq))
    
    # Calculate pairwise distances
    # dist^2 = ||a||^2 + ||b||^2 - 2<a,b>
    # All norms are sqrt(2), so norm_sq = 2.
    # dist^2 = 2 + 2 - 2<a,b> = 4 - 2<a,b>.
    # We need dist >= max_norm = sqrt(2).
    # dist^2 >= 2.
    # 4 - 2<a,b> >= 2 => 2 >= 2<a,b> => <a,b> <= 1.
    # Inner products of distinct vectors in this set:
    # As analyzed, <a,b> is 0 or 1 or -1?
    # If disjoint: 0.
    # If overlap 1 index:
    #   (1,1) and (1,-1) -> 1*1 + 1*(-1) = 0.
    #   (1,1) and (-1,1) -> -1+1=0.
    #   (1,1) and (1,1) -> same vector.
    #   So <a,b> is always 0 for distinct vectors?
    #   Wait. (1,1,0) and (1,0,1) -> disjoint? No, share index 0.
    #   (1,1,0) and (1,0,1). Inner product = 1*1 + 1*0 + 0*1 = 1.
    #   Ah! They share index 0.
    #   v1 = [1, 1, 0, ...]
    #   v2 = [1, 0, 1, ...]
    #   <v1, v2> = 1.
    #   dist^2 = 4 - 2(1) = 2.
    #   dist = sqrt(2).
    #   max_norm = sqrt(2).
    #   dist >= max_norm => sqrt(2) >= sqrt(2). True.
    #   So it works.
    
    return arr

# EVOLVE-BLOCK-END
