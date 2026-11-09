import numpy as np
import time
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 580
    
    # Strategy: Use all integer vectors in Z^11 with exactly two non-zero entries,
    # each being +1 or -1.
    # Norm squared is 1^2 + 1^2 = 2.
    # Maximum squared norm is 2.
    # Minimum squared distance between any two distinct such vectors is >= 2.
    # Proof:
    # Let u, v be two such vectors.
    # Case 1: Supports are disjoint. ||u-v||^2 = ||u||^2 + ||v||^2 = 2+2=4 >= 2.
    # Case 2: Supports share exactly 1 index i.
    #   u has non-zeros at i, j. v has non-zeros at i, k (j!=k).
    #   ||u-v||^2 = (u_i-v_i)^2 + u_j^2 + v_k^2.
    #   u_j^2=1, v_k^2=1.
    #   (u_i-v_i)^2 is 0 if u_i==v_i, else 4.
    #   So dist^2 is 2 or 6. Both >= 2.
    # Case 3: Supports are identical (share 2 indices i, j).
    #   u != v, so signs differ in at least one position.
    #   u.v = u_i v_i + u_j v_j.
    #   If signs differ in 1 pos: u.v = 1 - 1 = 0. dist^2 = 2+2-0=4 >= 2.
    #   If signs differ in 2 pos: u.v = -1 - 1 = -2. dist^2 = 2+2-(-4)=8 >= 2.
    #
    # Total number of such points:
    # Choose 2 indices from 11: C(11, 2) = 55.
    # For each pair, 4 sign combinations (+1,+1), (+1,-1), (-1,+1), (-1,-1).
    # Total = 55 * 4 = 220.
    
    points = []
    for pair in combinations(range(d), 2):
        if time.time() > deadline - 10:
            break
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = np.zeros(d, dtype=np.int64)
                v[pair[0]] = s1
                v[pair[1]] = s2
                points.append(v)
    
    points_array = np.array(points, dtype=np.int64)
    return points_array
