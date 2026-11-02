import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Generate candidate vectors
    # We want vectors with small norms to pack densely.
    # Let's try squared norms up to 20.
    max_sq_norm = 20
    max_coord = int(np.sqrt(max_sq_norm)) + 1
    
    candidates = []
    # Generate all combinations of coordinates in range [-max_coord, max_coord]
    # This is (2*max_coord+1)^11 which is too large to iterate blindly.
    # Better: iterate recursively or use a smarter generation.
    # Since max_sq_norm is small, most coordinates are 0.
    
    # We can generate by number of non-zero coordinates.
    # k=1: 11 * (2*max_coord)
    # k=2: C(11,2) * (2*max_coord-1)^2 ...
    # This is complex. Let's just use a random sampling + specific small vectors?
    # Or, since D=11 is small, we can iterate over all vectors with small L2 norm
    # by iterating over possible sums of squares.
    
    # Let's try a different approach:
    # The set of shortest vectors is most dense.
    # Let's include all vectors with L2 norm <= 4.
    
    # Generate all vectors with |v_i| <= 3 (since 3^2=9, 4^2=16, 5^2=25>20)
    # Actually, if one coord is 4, others must be 0.
    # If one coord is 3, others can be up to 1 (since 3^2+1^2=10 <= 20).
    
    # Let's just generate all vectors in the box [-3,3]^11 and filter by norm.
    # 7^11 is approx 1.9 * 10^9. Too big.
    
    # We need a smarter generator.
    # We can generate vectors by their non-zero support.
    
    def gen_vectors(max_sq):
        vecs = []
        # k=0: zero vector (skip)
        # k=1
        for i in range(D):
            for val in range(1, int(np.sqrt(max_sq)) + 1):
                for sign in [1, -1]:
                    v = [0]*D
                    v[i] = sign * val
                    vecs.append(v)
        # k=2
        for i, j in itertools.combinations(range(D), 2):
            max_v1 = int(np.sqrt(max_sq))
            for v1 in range(1, max_v1 + 1):
                rem = max_sq - v1*v1
                max_v2 = int(np.sqrt(rem))
                for v2 in range(1, max_v2 + 1):
                    for s1 in [1, -1]:
                        for s2 in [1, -1]:
                            v = [0]*D
                            v[i] = s1 * v1
                            v[j] = s2 * v2
                            vecs.append(v)
        # k=3
        for i, j, k in itertools.combinations(range(D), 3):
            max_v1 = int(np.sqrt(max_sq))
            for v1 in range(1, max_v1 + 1):
                rem1 = max_sq - v1*v1
                max_v2 = int(np.sqrt(rem1))
                for v2 in range(1, max_v2 + 1):
                    rem2 = rem1 - v2*v2
                    max_v3 = int(np.sqrt(rem2))
                    for v3 in range(1, max_v3 + 1):
                        for s1 in [1, -1]:
                            for s2 in [1, -1]:
                                for s3 in [1, -1]:
                                    v = [0]*D
                                    v[i] = s1 * v1
                                    v[j] = s2 * v2
                                    v[k] = s3 * v3
                                    vecs.append(v)
        # k=4
        for i, j, k, l in itertools.combinations(range(D), 4):
             # Only possible if 1^2+1^2+1^2+1^2 = 4 <= max_sq
             # And other small combos
             # For max_sq=20, we can have 1,1,1,1 (norm 4)
             # 1,1,1,2 (norm 7)
             # 1,1,2,2 (norm 10)
             # 1,2,2,2 (norm 13)
             # 2,2,2,2 (norm 16)
             # 1,1,1,3 (norm 12)
             # etc.
             
             # Iterate values
             max_v1 = int(np.sqrt(max_sq))
             for v1 in range(1, max_v1 + 1):
                 rem1 = max_sq - v1*v1
                 max_v2 = int(np.sqrt(rem1))
                 for v2 in range(1, max_v2 + 1):
                     rem2 = rem1 - v2*v2
                     max_v3 = int(np.sqrt(rem2))
                     for v3 in range(1, max_v3 + 1):
                         rem3 = rem2 - v3*v3
                         max_v4 = int(np.sqrt(rem3))
                         for v4 in range(1, max_v4 + 1):
                             for s1 in [1, -1]:
                                 for s2 in [1, -1]:
                                     for s3 in [1, -1]:
                                         for s4 in [1, -1]:
                                             v = [0]*D
                                             v[i] = s1 * v1
                                             v[j] = s2 * v2
                                             v[k] = s3 * v3
                                             v[l] = s4 * v4
                                             vecs.append(v)
        return vecs

    candidates = gen_vectors(max_sq_norm)
    
    # Remove duplicates (shouldn't be any if generated carefully, but just in case)
    # And filter out zero vector (not generated)
    
    # Sort by norm squared
    candidates.sort(key=lambda x: sum(v*v for v in x))
    
    selected = []
    
    # Greedy selection
    for v in candidates:
        if time.time() > deadline:
            break
            
        v_arr = np.array(v)
        v_norm_sq = np.dot(v_arr, v_arr)
        
        can_add = True
        for u in selected:
            diff = v_arr - np.array(u)
            dist_sq = np.dot(diff, diff)
            # Condition: dist >= max(norm_v, norm_u)
            # Since sorted, norm_u <= norm_v
            # So we need dist_sq >= v_norm_sq
            if dist_sq < v_norm_sq:
                can_add = False
                break
        
        if can_add:
            selected.append(v)
            
    if not selected:
        # Fallback: just return the standard 220
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)
        
    return np.array(selected, dtype=np.int64)
# EVOLVE-BLOCK-END
