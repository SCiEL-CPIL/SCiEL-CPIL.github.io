import numpy as np
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    # Generate all integer vectors with exactly two non-zero entries, each ±1
    # These form a valid kissing configuration in dimension 11
    vectors = []
    
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    # Check if we're still within time budget
    if time.time() > deadline:
        return np.array(vectors[:100], dtype=np.int64)
    
    # Try to extend the set by adding vectors with one non-zero entry (±1)
    # But first, check compatibility with existing vectors
    # A vector with a single 1 at index i has dot product 0 or 1 with any weight-2 vector
    # This is compatible since max norm^2 = 2, so min distance^2 = 2 - 1 = 1 >= ... 
    # Wait, we need min pairwise distance >= max norm
    # For weight-2 vectors: norm^2 = 2
    # Distance between two weight-2 vectors sharing one index: dot product = 1, dist^2 = 2+2-2*1 = 2
    # Distance between two weight-2 vectors with disjoint supports: dot product = 0, dist^2 = 4
    # So min distance^2 = 2, max norm^2 = 2. OK.
    
    # If we add a unit vector e_i (norm^2=1), distance to a weight-2 vector with e_i component:
    # dot product = 1, dist^2 = 1 + 2 - 2*1 = 1
    # But max norm^2 is now 2, so we need dist^2 >= 2. But we got 1. NOT compatible.
    
    # So we cannot simply add unit vectors. The weight-2 set is maximal in this family.
    
    # Let's try a different approach: use vectors with entries in {-1, 0, 1} 
    # but more systematically search for a larger set.
    
    # Actually, the kissing number in dimension 11 is known to be 254 (lower bound) 
    # and upper bound is 254 as well? Let me think...
    # Known kissing numbers: k(1)=2, k(2)=6, k(3)=12, k(4)=24, k(5)=40, k(6)=72, 
    # k(7)=126, k(8)=240, k(9)=460, k(10)=840, k(11) >= 1400? 
    # Actually k(11) is not exactly known. The best lower bound is around 1400-254?
    # Let me recall: k(9) >= 320, k(10) >= 552, k(11) >= 930? I'm not sure of exact values.
    
    # For now, let's stick with the weight-2 set which gives 220 vectors.
    # This should be a valid solution.
    
    return np.array(vectors, dtype=np.int64)
