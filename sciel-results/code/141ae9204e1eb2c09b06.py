import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    D = 11
    
    # Type A: 22 vectors
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)
    vectors_A = np.array(vectors_A, dtype=np.int64)

    # Type C: 5280 vectors
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)
    
    # Map support to index
    support_to_idx = {s: i for i, s in enumerate(supports)}
    
    # Precompute neighbors and pattern mappings
    # For each support S, find neighbors S' (intersect in 3)
    # For each neighbor, map each of the 16 vectors of S to a pattern (0-7) on the intersection
    # Pattern is determined by the 3 shared indices.
    
    # support_masks for fast intersection
    support_masks = []
    for s in supports:
        mask = 0
        for i in s:
            mask |= (1 << i)
        support_masks.append(mask)
    support_masks = np.array(support_masks, dtype=np.int64)
    
    bit_count_lut = np.array([bin(i).count('1') for i in range(2048)], dtype=np.int8)
    
    # neighbors[S] = list of (S_idx, shared_indices_tuple)
    # pattern_map[S][v_idx][n_idx] = pattern_bitmask
    # v_idx is 0-15, corresponding to signs of the 4 elements in S
    
    neighbors = [[] for _ in range(N_S)]
    pattern_maps = [None] * N_S # pattern_maps[S] will be a dict or array
    
    # Precompute sign vectors for a support
    # Signs are (s1, s2, s3, s4)
    sign_vecs = list(itertools.product([1, -1], repeat=4))
    # Map sign tuple to index 0-15
    sign_to_idx = {s: i for i, s in enumerate(sign_vecs)}
    
    for i, s in enumerate(supports):
        mask_i = support_masks[i]
        for j, s_j in enumerate(supports):
            if i == j:
                continue
            inter = mask_i & support_masks[j]
            inter_count = bit_count_lut[inter]
            if inter_count == 3:
                # Find the 3 shared indices
                shared = tuple(idx for idx in range(D) if (inter >> idx) & 1)
                # Find the index of j in neighbors of i
                # We need to map the vector in S to a pattern on 'shared'
                # The vector has signs on s[0], s[1], s[2], s[3]
                # The pattern is the signs on the shared indices.
                # We need to know which positions in s correspond to shared indices.
                
                # For each of the 16 vectors in S:
                # v_idx corresponds to signs (s0, s1, s2, s3)
                # Pattern is the subsequence of signs at positions where s[k] is in shared.
                
                # Precompute for this pair (i, j)
                # shared_indices_in_s: list of indices k in 0..3 such that s[k] in shared
                shared_pos_in_s = []
                for k, val in enumerate(s):
                    if val in shared:
                        shared_pos_in_s.append(k)
                
                # For each v_idx (0-15), compute pattern
                # pattern_bits: 3 bits.
                # bit 0 corresponds to shared_pos_in_s[0]
                # bit 1 corresponds to shared_pos_in_s[1]
                # bit 2 corresponds to shared_pos_in_s[2]
                # Sign 1 -> 1, Sign -1 -> 0? Or just use the sign value.
                # We just need a unique ID for the pattern.
                # There are 8 patterns.
                
                patterns_for_v = [0] * 16
                for v_idx, signs in enumerate(sign_vecs):
                    p = 0
                    for bit_idx, pos in enumerate(shared_pos_in_s):
                        if signs[pos] == 1:
                            p |= (1 << bit_idx)
                        # else 0
                    patterns_for_v[v_idx] = p
                    
                # Store this mapping
                # We need to access this quickly during the greedy phase.
                # neighbors[i].append((j, shared))
                # pattern_maps[i] needs to be indexed by neighbor index in the list.
                
                # Let's store pattern_maps[i] as a list of lists, parallel to neighbors[i]
                if pattern_maps[i] is None:
                    pattern_maps[i] = []
                
                # Find index of j in neighbors[i] (should be the last one added if we build it properly, but we are iterating j)
                # Actually, we are building neighbors[i] here.
                # So the index is len(neighbors[i]) before append.
                n_idx = len(neighbors[i])
                neighbors[i].append((j, shared))
                pattern_maps[i].append(patterns_for_v)
                
    # Greedy Selection
    best_vectors = []
    best_count = 0
    
    # We will run multiple restarts
    # Each restart: shuffle vectors, greedily add
    
    # Total vectors C: 5280
    # Create list of all (support_idx, v_idx)
    all_vecs = []
    for s_idx in range(N_S):
        for v_idx in range(16):
            all_vecs.append((s_idx, v_idx))
            
    rng = random.Random(42)
    
    for restart in range(50):
        if time.time() > deadline - 20:
            break
            
        rng.shuffle(all_vecs)
        
        # used_patterns[S][n_idx] is a bitmask of patterns used by U_S on intersection with neighbor n_idx
        # Note: We only need to track patterns used by U_S.
        # Conflict check: For v in S, pattern p on neighbor S'.
        # Check if p is in used_patterns[S'][idx_of_S_in_S'_neighbors].
        
        # To make lookup fast, precompute for each (S, S') pair, the index of S in S' neighbors.
        # rev_idx[S][S'] = index of S in neighbors[S']
        # This is a 330x330 matrix, mostly -1.
        rev_idx = [[-1]*N_S for _ in range(N_S)]
        for i in range(N_S):
            for n_idx, (j, _) in enumerate(neighbors[i]):
                rev_idx[j][i] = n_idx
                
        used_patterns = [[0]*len(neighbors[i]) for i in range(N_S)]
        
        current_vecs = []
        
        for s_idx, v_idx in all_vecs:
            # Check conflicts with neighbors
            conflict = False
            patterns_to_set = []
            
            for n_idx, (n_s_idx, shared) in enumerate(neighbors[s_idx]):
                p = pattern_maps[s_idx][n_idx][v_idx]
                # Check if this pattern is already used by U_{n_s_idx} on the intersection with s_idx
                # The index of s_idx in neighbors[n_s_idx] is rev_idx[n_s_idx][s_idx]
                rev_n_idx = rev_idx[n_s_idx][s_idx]
                if used_patterns[n_s_idx][rev_n_idx] & (1 << p):
                    conflict = True
                    break
                patterns_to_set.append((n_idx, p))
                
            if not conflict:
                # Add vector
                current_vecs.append((s_idx, v_idx))
                # Update used_patterns for s_idx
                for n_idx, p in patterns_to_set:
                    used_patterns[s_idx][n_idx] |= (1 << p)
                    
        # Convert to array
        if len(current_vecs) > best_count:
            best_count = len(current_vecs)
            best_vectors = current_vecs
            
    # Generate the actual vectors for best_vectors
    vectors_C = []
    for s_idx, v_idx in best_vectors:
        s = supports[s_idx]
        signs = sign_vecs[v_idx]
        v = [0] * D
        for idx, sign in zip(s, signs):
            v[idx] = sign
        vectors_C.append(v)
        
    vectors_C = np.array(vectors_C, dtype=np.int64)
    
    final_vectors = np.vstack([vectors_A, vectors_C])
    
    return final_vectors
