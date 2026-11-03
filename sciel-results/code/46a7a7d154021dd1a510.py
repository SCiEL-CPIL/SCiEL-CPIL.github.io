"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 135.0
    
    # Precompute all weight-2 vectors: e_i +/- e_j for i < j
    # There are C(11,2) * 4 = 55 * 4 = 220 such vectors
    # Seed uses exactly these with min dist sqrt(2), max norm sqrt(2).
    # Score = 220.
    
    # Strategy: Try to find a larger set by using weight-4 vectors (norm 2)
    # and mixing. But weight-4 vectors are many: C(11,4)*16 = 330*16 = 5280.
    # Constraint: min pairwise dist >= max norm.
    
    # Key insight: The seed has 220 vectors all with norm sqrt(2).
    # If we use vectors with larger norm, the distance threshold increases.
    # So we want vectors with small norm to allow many of them.
    
    # Actually, let's think differently. The score is k (number of vectors).
    # The constraint is: for all pairs u,v: ||u-v|| >= max_w ||w||.
    # Let M = max_w ||w||. Then ||u-v|| >= M for all u != v.
    # Also ||u|| >= 1 (nonzero integer vector).
    
    # If all vectors have the same norm r, then ||u-v||^2 = 2r^2 - 2<u,v> >= r^2
    # => r^2 >= 2<u,v> => <u,v> <= r^2/2.
    
    # For weight-2 vectors (norm sqrt(2)): <u,v> <= 1.
    # Two weight-2 vectors e_i+e_j and e_k+e_l:
    # If disjoint: <u,v>=0, dist^2=4, dist=2 >= sqrt(2). OK.
    # If share one index: <u,v>=1 (if same sign) or 0 (if opposite sign on shared, depends).
    # Actually e_i+e_j and e_i+e_k: <u,v> = 1 (from e_i.e_i). dist^2 = 2+2-2*1 = 2. dist=sqrt(2). OK.
    # e_i+e_j and e_i-e_k: <u,v> = 1. Same.
    # e_i+e_j and -e_i+e_k: <u,v> = -1. dist^2 = 2+2-2*(-1)=6. OK.
    # e_i+e_j and -e_i-e_k: <u,v> = -1. Same.
    # So all pairs of weight-2 vectors have dist >= sqrt(2). That's why the seed has 220.
    
    # Can we add more? What about weight-1 vectors (norm 1)?
    # e_i and e_j: dist^2 = 2, dist = sqrt(2) >= sqrt(2). OK.
    # e_i and e_i+e_j: dist^2 = 0^2+1^2+... = 1. dist=1 < sqrt(2). NO.
    # So we can't mix weight-1 and weight-2 if the max norm is sqrt(2).
    
    # What if we use only weight-1 vectors?
    # e_i, -e_i: dist^2 = 4, dist=2 >= 1. OK.
    # e_i, e_j: dist^2=2, dist=sqrt(2) >= 1. OK.
    # So all 22 weight-1 vectors (11 axes * 2 signs) work with max norm 1.
    # Score = 22. Much worse than 220.
    
    # What about a mix where max norm is 2 (weight-4 vectors)?
    # Then we need dist >= 2 between all pairs.
    # Weight-2 vector u and weight-2 vector v: dist >= 2 requires they be disjoint supports.
    # C(11,2)=55 pairs of indices. Max matching of disjoint pairs: floor(11/2)=5.
    # So at most 5 weight-2 vectors with pairwise dist >= 2.
    # Weight-4 vectors: many, but they constrain each other heavily.
    
    # So the seed's 220 is quite good for the "all same weight-2" approach.
    
    # Let's try a different angle: Can we use vectors of mixed weights but keep max norm small?
    # If max norm is sqrt(2), we can only use weight-1 and weight-2 vectors.
    # Weight-1: e_i. Weight-2: e_i+e_j.
    # e_i and e_i+e_j: dist^2 = 1. dist=1 < sqrt(2). Invalid.
    # So no mixing of weight-1 and weight-2.
    
    # Therefore, the seed's 220 is optimal for max norm sqrt(2).
    
    # Can we do better with a different construction?
    # What if we use a lattice-based approach?
    
    # Let me try a greedy local search to see if we can beat 220.
    # Start with the seed's 220 vectors, then try to add or swap.
    
    # Actually, let's try to construct a set using a different family.
    # Consider vectors with entries in {-1, 0, 1} with weight 3.
    # Norm = sqrt(3) ~ 1.732.
    # Two weight-3 vectors u, v: dist^2 = 3+3-2<u,v> = 6-2<u,v>.
    # Need dist >= sqrt(3): 6-2<u,v> >= 3 => <u,v> <= 1.5 => <u,v> <= 1.
    # <u,v> can be -3,-1,1,3 (since integer entries in {-1,0,1}).
    # So <u,v> <= 1 means we exclude pairs with <u,v>=3 (identical support, same signs).
    # How many weight-3 vectors are there? C(11,3)*8 = 165*8 = 1320.
    # Can we find a large subset with <u,v> <= 1 for all pairs?
    
    # This is a combinatorial problem. Let me try a greedy approach.
    
    # Generate all weight-3 vectors
    weight3_vectors = []
    for support in range(1 << D):
        if bin(support).count('1') != 3:
            continue
        for signs in range(1 << 3):
            v = np.zeros(D, dtype=np.int64)
            idx = 0
            for i in range(D):
                if support & (1 << i):
                    v[i] = 1 if (signs & (1 << idx)) else -1
                    idx += 1
            weight3_vectors.append(v)
    
    weight3_vectors = np.array(weight3_vectors)
    # 1320 vectors
    
    # Greedy: pick vectors such that for all pairs, <u,v> <= 1
    selected = []
    for v in weight3_vectors:
        if not selected:
            selected.append(v)
            continue
        # Check against all selected
        dots = weight3_vectors[:len(selected)] @ v
        if np.all(dots <= 1):
            selected.append(v)
    
    print(f"Weight-3 greedy: {len(selected)} vectors")
    
    # Compare with seed
    if len(selected) > 220:
        return np.array(selected, dtype=np.int64)
    
    # Try weight-2 but with a twist: maybe we can do better than 220?
    # The seed has all C(11,2)*4 = 220 weight-2 vectors.
    # Is 220 the maximum for weight-2? Yes, since there are exactly 220.
    
    # What about mixing weight-2 with weight-4?
    # If we include any weight-4 vector (norm 2), then max norm = 2.
    # Then all pairs need dist >= 2.
    # Weight-2 vectors with each other: need disjoint supports.
    # Weight-2 u and weight-4 v: need dist >= 2, i.e., <u,v> <= (2+4)/2 - 2 = 1.
    # Wait: ||u-v||^2 = ||u||^2 + ||v||^2 - 2<u,v> = 2+4-2<u,v> = 6-2<u,v> >= 4
    # => <u,v> <= 1.
    # Weight-4 u and weight-4 v: ||u-v||^2 = 4+4-2<u,v> = 8-2<u,v> >= 4 => <u,v> <= 2.
    
    # This is getting complex. Let me just try a random restart local search.
    
    # For now, return the seed if nothing better is found.
    if time.time() > deadline:
        vectors = []
        for i in range(D):
            for j in range(i+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = [0] * D
                        v[i], v[j] = si, sj
                        vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    # Generate seed vectors
    seed_vectors = []
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    seed_vectors.append(v)
    seed_vectors = np.array(seed_vectors, dtype=np.int64)
    
    return seed_vectors
# EVOLVE-BLOCK-END
