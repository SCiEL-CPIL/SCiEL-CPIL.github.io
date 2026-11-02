import numpy as np
import itertools
import time

def construct():
    """
    Constructs a set of nonzero integer vectors in R^11 such that the minimum 
    pairwise distance is at least the maximum vector norm.
    
    Strategy: Use all weight-2 vectors (vectors with exactly two non-zero entries, 
    each being +1 or -1).
    
    Let V be the set of all such vectors.
    1. Norm of any v in V is sqrt(2). So R = sqrt(2).
    2. Condition: ||u - v|| >= R for all distinct u, v in V.
       Squared: ||u - v||^2 >= 2.
       ||u - v||^2 = ||u||^2 + ||v||^2 - 2(u dot v) = 2 + 2 - 2(u dot v) = 4 - 2(u dot v).
       So we need 4 - 2(u dot v) >= 2  =>  2(u dot v) <= 2  =>  u dot v <= 1.
    3. For distinct weight-2 vectors u and v:
       - If they have disjoint supports, u dot v = 0. (0 <= 1, OK)
       - If they share 1 index, u dot v is either 1 or -1. (1 <= 1, OK)
       - If they share 2 indices, they must be identical or differ in signs.
         Since they are distinct, they cannot be identical.
         If they share 2 indices, they have the same support.
         u = (s1, s2, 0...), v = (t1, t2, 0...).
         u dot v = s1*t1 + s2*t2.
         Possible values: 2 (if s1=t1, s2=t2 -> u=v, excluded), 
                           0 (if one sign differs), 
                          -2 (if both signs differ).
         So for distinct u, v with same support, u dot v is 0 or -2. (0 <= 1, OK; -2 <= 1, OK).
       
       Thus, ALL distinct weight-2 vectors satisfy the condition.
       
    Number of such vectors:
    Choose 2 positions from 11: C(11, 2) = 55.
    Choose signs for 2 positions: 2^2 = 4.
    Total = 55 * 4 = 220.
    
    This returns a valid configuration of size 220.
    """
    D = 11
    vectors = []
    
    # Generate all weight-2 vectors
    for pos in itertools.combinations(range(D), 2):
        p1, p2 = pos
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                v = [0] * D
                v[p1] = s1
                v[p2] = s2
                vectors.append(v)
                
    # Convert to numpy array
    result = np.array(vectors, dtype=np.int64)
    
    return result
