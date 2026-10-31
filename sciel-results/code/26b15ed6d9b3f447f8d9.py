import numpy as np
import time

D = 11

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # 20s safety margin before 160s kill
    rng = np.random.default_rng(12345)
    
    # Generate all weight-2 vectors: C(11,2) * 4 = 55 * 4 = 220
    w2_vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    w2_vectors.append(v)
    w2_vectors = np.array(w2_vectors, dtype=np.int64)  # shape (220, 11)
    
    # Generate all weight-4 vectors: C(11,4) * 16 = 330 * 16 = 5280
    # A weight-4 vector has 4 non-zero entries, each in {-1, 1}
    from itertools import combinations
    w4_vectors = []
    for combo in combinations(range(D), 4):
        idx = list(combo)
        for mask in range(16):
            v = np.zeros(D, dtype=np.int64)
            for k, pos in enumerate(idx):
                v[pos] = 1 if (mask >> k) & 1 else -1
            w4_vectors.append(v)
    w4_vectors = np.array(w4_vectors, dtype=np.int64)  # shape (5280, 11)
    
    # Conflict check: two vectors u, v are compatible iff ||u-v||^2 >= max(||u||^2, ||v||^2)
    # For w2 vs w4: ||u-v||^2 = ||u||^2 + ||v||^2 - 2 u.v = 4 + 16 - 2 u.v = 20 - 2 u.v
    # Need 20 - 2 u.v >= 16 => u.v <= 2
    # For w4 vs w4: ||u-v||^2 = 16 + 16 - 2 u.v = 32 - 2 u.v >= 16 => u.v <= 8
    # But also need >= 16 (max norm squared is 16 for w4)
    
    # Actually, re-reading the task: "minimum pairwise distance is at least the maximum vector norm"
    # The maximum norm among all vectors in our set. If we include w4 vectors, max norm = 4.
    # So we need pairwise distance >= 4 for ALL pairs.
    # ||u - v|| >= 4 => ||u-v||^2 >= 16
    
    # w2 vs w2: ||u-v||^2 = 4+4-2u.v = 8-2u.v >= 16? No! 8-2u.v <= 8. 
    # So weight-2 vectors CANNOT satisfy distance >= 4 among themselves!
    # The seed program only uses weight-2 vectors and their max norm is 2, so min dist >= 2 works.
    # If we add weight-4 vectors, max norm becomes 4, and we need ALL pairwise distances >= 4.
    # But weight-2 vectors among themselves have distance at most sqrt(8) < 4.
    
    # So we MUST choose: either all weight-2 (max norm 2, need dist >= 2), 
    # or include some weight-4 (max norm 4, need dist >= 4 for all pairs).
    
    # Strategy: Try to find a large set of weight-4 vectors that are mutually at distance >= 4.
    # Among weight-4 vectors: need u.v <= 8. Since u and v each have 4 non-zero entries with values +-1,
    # u.v = (number of positions where both are +1) + (number where both are -1) - (number where u=+1,v=-1) - (number where u=-1,v=+1)
    # Max u.v = 4 (when they're identical). We need u.v <= 8, which is always true for distinct w4 vectors?
    # Wait: if u = v, u.v = 4. If u != v, u.v can be at most... they share at most 3 non-zero positions, 
    # so u.v <= 3 + something. Actually if they share k positions, u.v = sum of products over shared positions.
    # Each shared position contributes +-1. So |u.v| <= min(4,4) = 4, and if they differ in at least one position,
    # they can still have u.v = 4 only if they're identical.
    # So for distinct w4 vectors, u.v <= 3 < 8. Thus ALL distinct weight-4 vectors are mutually compatible!
    
    # Wait, that can't be right. Let me double-check: u and v are weight-4 vectors (4 non-zero entries each, values +-1).
    # u.v = sum over all 11 positions of u_k * v_k. Only positions where both are non-zero contribute.
    # If they share s positions, u.v = sum of s terms each +-1, so -s <= u.v <= s <= 4.
    # We need ||u-v||^2 = 32 - 2*u.v >= 16, i.e., u.v <= 8. Always true since u.v <= 4.
    
    # So ALL 5280 weight-4 vectors form a valid packing with max norm 4 and min distance >= 4!
    # That gives score 5280 / 220 = 24.0
    
    # Let's verify: the seed has 220 vectors. We can return all 5280 weight-4 vectors.
    # But wait - do we need nonzero vectors? Yes, they're all nonzero.
    # Are they integer vectors? Yes.
    
    # Actually, can we do even better? What about weight-3 vectors?
    # Weight-3: C(11,3)*8 = 165*8 = 1320 vectors, norm = sqrt(3)
    # If we mix weights, max norm determines the threshold.
    # If max norm is sqrt(16)=4, we need all distances >= 4.
    # w3 vs w4: ||u-v||^2 = 3+16-2u.v = 19-2u.v >= 16 => u.v <= 1.5 => u.v <= 1
    # w3 vs w3: ||u-v||^2 = 3+3-2u.v = 6-2u.v >= 16? No, 6-2u.v <= 6.
    # So we can't include weight-3 if max norm is 4.
    
    # What if we only use weight-4? 5280 vectors.
    # Can we add weight-5+ vectors? Weight-5: C(11,5)*32 = 462*32 = 14784, norm = sqrt(5)
    # But then max norm would be 5 (if we include weight-10 vectors) or sqrt(5) if only weight-5.
    # If we use only weight-5 vectors: need ||u-v||^2 >= 5 for all pairs? No, max norm is sqrt(5), 
    # so we need distance >= sqrt(5), i.e., ||u-v||^2 >= 5.
    # w5 vs w5: ||u-v||^2 = 10 - 2u.v >= 5 => u.v <= 2.5 => u.v <= 2.
    # Not all w5 vectors satisfy this.
    
    # The weight-4 set of 5280 is already great. Let's just return that.
    return w4_vectors
# EVOLVE-BLOCK-END
