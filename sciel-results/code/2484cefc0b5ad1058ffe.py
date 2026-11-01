"""Kissing number in dimension 11 - construct() -> integer array of shape (k, d)"""
import time
import numpy as np

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 125  # safety margin within 150s wall-clock
    
    # Seed construction: all vectors e_i + s*e_j with i<j, s in {+1,-1}
    # These have norm sqrt(2), pairwise distances >= sqrt(2)
    seed_vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for s in (1, -1):
                v = [0] * D
                v[i] = 1
                v[j] = s
                seed_vectors.append(v)
    
    seed_array = np.array(seed_vectors, dtype=np.int64)
    k = len(seed_vectors)
    max_norm_sq = 2  # norm squared of each seed vector
    
    # Try to augment: add weight-3 vectors with norm^2 <= 2? No, weight-3 has norm^2=3 > 2
    # The constraint is: min pairwise distance >= max vector norm
    # If we add a vector of norm^2 > 2, then max norm increases, and all pairwise distances must be >= that new norm
    # This makes it harder. So we can only add vectors with norm^2 <= 2 (i.e., weight 1 or 2)
    # Weight 1 vectors: e_i. Distance from e_i to e_j+s*e_k: if i not in {j,k}, dist^2 = 2+1=3 >= 2 OK
    #   but if i=j, dist^2 = 0+1+1=2 >= 2 OK; if i=k, dist^2 = 1+1+0=2 >= 2 OK
    # Wait: dist(e_i, e_j + s*e_k)^2 = (delta_ij - 1)^2 + (delta_ik - s)^2 + ... 
    # Actually dist(e_i, v)^2 = ||e_i - v||^2 = 1 - 2*e_i.v + ||v||^2
    # For v = e_j + s*e_k: e_i.v = delta_ij + s*delta_ik
    # ||v||^2 = 2
    # dist^2 = 1 - 2(delta_ij + s*delta_ik) + 2
    # If i=j: 1 - 2(1 + 0) + 2 = 1. That's < 2! So e_i conflicts with e_i + s*e_k (which is in seed)
    # If i=k: 1 - 2(0 + s) + 2 = 3 - 2s. If s=1: 1 < 2. Conflict!
    # So weight-1 vectors conflict with seed. Can't add them.
    
    # What about adding more weight-2 vectors with different sign patterns?
    # Seed already has all e_i + s*e_j for i<j. That's C(11,2)*2 = 110*2 = 220.
    # Are there other weight-2 integer vectors? No, all are e_i + s*e_j or -e_i + s*e_j, etc.
    # Actually weight-2 vectors are: (a_i, a_j) with a_i^2 + a_j^2 = 2, so |a_i|=|a_j|=1.
    # All such are covered by the seed (up to sign, but we include all sign combos for ordered pairs i<j).
    # Wait: e_i - e_j is included (s=-1 when j is the second index). And -e_i + e_j would be i>j...
    # Actually for i<j, we have e_i + e_j and e_i - e_j. What about -e_i - e_j and -e_i + e_j?
    # Those are just negatives: -e_i - e_j = -(e_i + e_j), and -e_i + e_j = -(e_i - e_j).
    # The task says "nonzero integer vectors" - does it count both v and -v? Yes, they're distinct vectors.
    # But do they satisfy the constraint? dist(v, -v) = ||2v|| = 2*sqrt(2) = sqrt(8) > sqrt(2). OK.
    # But wait: the seed only generates i<j with s in {+1,-1} applied to the SECOND component.
    # So for pair (i,j) with i<j: vectors are e_i+e_j and e_i-e_j.
    # Missing: -e_i-e_j and -e_i+e_j. Let me check if these are valid to add.
    
    # dist(-e_i-e_j, e_k+e_l)^2 for various cases...
    # Actually, let's just generate ALL weight-2 vectors: all (a,b) with a^2+b^2=2, a,b in {-1,1}
    # That's 4 per pair of positions, and 2 positions = 2*C(11,2)*4 = 440 total weight-2 vectors.
    # Let me check if all 440 satisfy pairwise distance >= sqrt(2).
    
    all_w2 = []
    for i in range(D):
        for j in range(D):
            if i == j:
                continue
            for ai in (1, -1):
                for aj in (1, -1):
                    v = [0] * D
                    v[i] = ai
                    v[j] = aj
                    all_w2.append(v)
    
    all_w2_array = np.array(all_w2, dtype=np.int64)
    # Verify: compute all pairwise distances
    # This is 440*440/2 ~ 96800 pairs, fast
    n_w2 = len(all_w2_array)
    valid = True
    if n_w2 <= 2000:  # safety check
        dists = np.linalg.norm(all_w2_array[:, None, :] - all_w2_array[None, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        min_dist = dists.min()
        if min_dist < np.sqrt(2) - 1e-9:
            valid = False
            print(f"All weight-2 vectors have min pairwise dist {min_dist:.4f} < sqrt(2)")
    
    if valid:
        print(f"Using all {n_w2} weight-2 vectors")
        return all_w2_array
    
    # Fallback: use seed (220 vectors)
    print(f"Falling back to seed with {k} vectors")
    return seed_array
# EVOLVE-BLOCK-END
