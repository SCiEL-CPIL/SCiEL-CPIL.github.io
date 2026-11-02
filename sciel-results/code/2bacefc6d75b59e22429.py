import numpy as np
import time
import itertools
import random

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    # Base set: all vectors with two non-zero entries of value ±1
    # Norm squared = 2, norm = sqrt(2)
    base_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                base_vectors.append(v)

    base_arr = np.array(base_vectors)
    n_base = len(base_arr)  # 55 * 4 = 220

    # Generate candidate vectors with one entry of ±2 and others 0
    # Norm squared = 4, norm = 2
    # These are compatible with base vectors: dot product can be at most 1 (if same position and sign match)
    # Actually dot product with base vector: if base has ±1 at position i and candidate has ±2 at i, dot = ±2
    # We need distance >= max_norm. If we include norm-2 vectors, max_norm = 2.
    # Distance squared between norm-2 vector u and norm-sqrt(2) vector v:
    # |u-v|^2 = 4 + 2 - 2*u.v = 6 - 2*u.v
    # Need >= 4, so u.v <= 1.
    # But u.v can be ±2 if they share the same position with matching/opposite signs.
    # So we need u.v <= 1, which means they can't share the same position with same sign.
    # If u has +2 at i and v has +1 at i, dot = 2 > 1. Conflict.
    # If u has +2 at i and v has -1 at i, dot = -2 <= 1. OK.
    # If u has +2 at i and v has 0 at i, dot = 0. OK.
    
    # So for each ±2 vector, it conflicts with base vectors that have +1 at the same position (if the 2-vector is +2)
    # or -1 at the same position (if the 2-vector is -2).
    # Each position i has 2*(D-1) = 20 base vectors with ±1 at i.
    # Half have +1, half have -1. So 10 base vectors conflict with +2e_i, and 10 with -2e_i.
    
    # Let's try a different approach: use only base vectors and add norm-sqrt(3) vectors (three ±1 entries)
    # Norm squared = 3, norm = sqrt(3)
    # If we include these, max_norm = sqrt(3) (since sqrt(3) > sqrt(2))
    # Distance between two base vectors (norm-sqrt(2)): |u-v|^2 = 2+2-2u.v = 4-2u.v
    # Need >= 3, so u.v <= 0.5, i.e., u.v <= 0 for integer vectors.
    # Base vectors have dot product in {-2, -1, 0, 1, 2}. We need u.v <= 0.
    # Two base vectors conflict if they share a position with same sign (dot >= 1).
    
    # This is getting complex. Let me use the approach from the inspiration:
    # Use vectors with norm squared 4 (one ±2) as the main set, with max_norm = 2.
    
    # Generate all vectors with one entry ±2
    w1_vectors = []
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = 2 * s
            w1_vectors.append(v)
    
    w1_arr = np.array(w1_vectors)
    n_w1 = len(w1_arr)  # 22

    # Generate vectors with two entries ±2
    w2_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = 2 * si
                v[j] = 2 * sj
                w2_vectors.append(v)
    
    w2_arr = np.array(w2_vectors)
    n_w2 = len(w2_arr)  # 55 * 4 = 220

    # With max_norm = 2 (from w1 vectors), we need all pairwise distances >= 2.
    # For two w1 vectors: |u-v|^2 = 4+4-2u.v = 8-2u.v >= 4 => u.v <= 2
    # w1 vectors have dot products: 4 (same), 0 (different positions), -4 (opposite sign same pos)
    # So w1 vectors are compatible with each other only if u.v <= 2, which is always true.
    # Actually all w1 vectors are mutually compatible!
    
    # For w1 and w2: |u-v|^2 = 4+8-2u.v = 12-2u.v >= 4 => u.v <= 4
    # Max dot product is 4 (if same position, same sign). So all compatible.
    
    # For two w2 vectors: |u-v|^2 = 8+8-2u.v = 16-2u.v >= 4 => u.v <= 6
    # Dot products for w2: 8 (same), 4 (one position same sign), -4 (one position opposite), 0 (no common), etc.
    # Max is 8. Need u.v <= 6. So w2 vectors conflict if u.v >= 7, i.e., u.v = 8 (identical).
    # Actually if they share one position with same sign: dot = 4. If two positions same sign: dot = 8.
    # So two w2 vectors conflict only if they are identical or have both positions with same signs.
    
    # Let's use a greedy approach with all w1 and w2 vectors.
    all_candidates = list(w1_vectors) + list(w2_vectors)
    
    # Greedily select
    selected = []
    selected_arr = None
    
    for v in all_candidates:
        if time.time() > deadline:
            break
        v = np.array(v, dtype=np.int64)
        if not selected:
            selected.append(v)
            selected_arr = v.reshape(1, -1)
        else:
            dots = selected_arr @ v
            if np.all(dots <= 6):  # For norm-2 vectors, need u.v <= 6 for distance >= 2
                # Wait, let me recalculate. Max norm is 2. Need |u-v| >= 2.
                # |u-v|^2 >= 4. |u|^2 + |v|^2 - 2u.v >= 4.
                # Both have norm^2 = 8 (for w2) or 4 (for w1).
                # Case w1-w1: 4+4-2u.v >= 4 => u.v <= 2. Max u.v = 4. So need u.v <= 2.
                # But w1 vectors can have u.v = 4 (same vector). So we can't include duplicates.
                # Since all w1 vectors are distinct, u.v for distinct w1 is 0 or -4. Both <= 2. OK.
                # Case w1-w2: 4+8-2u.v >= 4 => u.v <= 4. Max u.v = 4. So need u.v <= 4. OK always.
                # Case w2-w2: 8+8-2u.v >= 4 => u.v <= 6. Max u.v = 8. So need u.v <= 6.
                # So the constraint is u.v <= 6 for w2-w2 pairs, and always OK for others.
                
                # Let me just check the actual distance condition properly
                norms_sq = selected_arr @ selected_arr.T
                # Actually let me just compute distances properly
                
                # Recompute: for each selected vector s, check |s-v| >= 2
                # |s-v|^2 = |s|^2 + |v|^2 - 2*s.v >= 4
                # |s|^2 is 4 or 8, |v|^2 is 4 or 8
                # So we need 2*s.v <= |s|^2 + |v|^2 - 4
                
                v_norm_sq = int(v @ v)
                s_norms_sq = selected_arr @ selected_arr.T  # This is wrong, should be column norms
                
                # Let me just do it simply
                s_norms_sq = np.sum(selected_arr ** 2, axis=1)
                dots = selected_arr @ v
                dist_sq = s_norms_sq + v_norm_sq - 2 * dots
                if np.all(dist_sq >= 4):
                    selected.append(v)
                    selected_arr = np.vstack([selected_arr, v.reshape(1, -1)])
    
    if selected:
        return np.array(selected, dtype=np.int64)
    else:
        return np.array([], dtype=np.int64).reshape(0, D)

# EVOLVE-BLOCK-END
