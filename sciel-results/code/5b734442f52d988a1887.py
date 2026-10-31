import numpy as np
import time
import random
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Strategy: 
    # The kissing number in dimension 11 is 224, achieved by the Leech lattice.
    # The minimal vectors of the Leech lattice have norm squared 4.
    # In R^11, we can look for integer vectors with norm squared 4 (weight 2, +/-1 entries).
    # There are C(11,2) * 4 = 220 such vectors.
    # The condition for a kissing configuration is min_{i!=j} ||v_i - v_j|| >= max_k ||v_k||.
    # If we pick a set S of these vectors, R = 2 (since all have norm 2).
    # We need ||v_i - v_j||^2 >= 4 for all i!=j in S.
    # ||v_i - v_j||^2 = 4 + 4 - 2 v_i . v_j = 8 - 2 v_i . v_j.
    # So we need 8 - 2 v_i . v_j >= 4 => 2 v_i . v_j <= 4 => v_i . v_j <= 2.
    # The dot product of two distinct weight-2 vectors in R^11 can be:
    # - 2, 0, -2 (if they share 1 or 2 indices with same/opposite signs)
    # Specifically:
    #   If they share 2 indices: v_i . v_j = sign1*sign2 + sign3*sign4. Can be 2, 0, -2.
    #   If they share 1 index: v_i . v_j = sign1*sign2. Can be 1 or -1.
    #   If they share 0 indices: v_i . v_j = 0.
    # We need v_i . v_j <= 2.
    # The only problematic case is when v_i . v_j > 2, which is impossible for integer vectors of weight 2?
    # Wait, max dot product is 2 (if they are identical). But we pick distinct vectors.
    # If they are distinct, can they have dot product 2?
    # v_i = (1, 1, 0, ...), v_j = (1, 1, 0, ...) -> identical.
    # If distinct, they must differ in at least one entry.
    # If they share 2 indices, and the signs are the same, dot product is 2. But then they are identical.
    # So for distinct vectors, max dot product is 1?
    # Let's check: v_i = (1, 1, 0, ...), v_j = (1, -1, 0, ...) -> dot = 0.
    # v_i = (1, 1, 0, ...), v_j = (1, 0, 1, ...) -> dot = 1.
    # So for any two distinct weight-2 integer vectors, v_i . v_j <= 1.
    # Thus, 8 - 2(1) = 6 >= 4.
    # So ANY set of distinct weight-2 integer vectors in R^11 satisfies the condition!
    # Let's verify.
    # v_i = e_1 + e_2. v_j = e_1 + e_3.
    # ||v_i||^2 = 2. ||v_j||^2 = 2. R = sqrt(2).
    # ||v_i - v_j||^2 = ||e_2 - e_3||^2 = 2.
    # Is 2 >= sqrt(2)? Yes, 2 >= 1.414.
    # Wait, the condition is min pairwise distance >= max vector norm.
    # Distance is sqrt(2). Norm is sqrt(2). So sqrt(2) >= sqrt(2) holds.
    # So the entire pool of 220 vectors is a valid kissing configuration?
    # Let's check the score. k = 220.
    # The current program gets ~1.168. The seed likely gets ~1.0.
    # If 220 is the answer, the score would be 220/seed_k.
    # If seed_k is around 100, score is 2.2.
    # Let's double check if there are any pairs with distance < norm.
    # Distance squared = ||v_i - v_j||^2.
    # Norm squared = 2.
    # We need ||v_i - v_j||^2 >= 2.
    # ||v_i - v_j||^2 = 4 - 2 v_i . v_j.
    # We need 4 - 2 v_i . v_j >= 2 => 2 v_i . v_j <= 2 => v_i . v_j <= 1.
    # As established, for distinct weight-2 vectors, max dot product is 1.
    # So the condition holds for all pairs.
    
    # So the solution is simply all 220 weight-2 vectors.
    
    pool = []
    for comb in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for idx, sign in zip(comb, signs):
                v[idx] = sign
            pool.append(v)
            
    return np.array(pool, dtype=np.int64)
