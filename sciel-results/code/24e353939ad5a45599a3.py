import numpy as np
import itertools
import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    D = 11
    
    # Type A: Vectors with one non-zero entry +/- 2
    # Norm^2 = 4. Universally compatible with all Type D vectors.
    vectors_A = []
    for i in range(D):
        v1 = [0] * D
        v1[i] = 2
        vectors_A.append(v1)
        v2 = [0] * D
        v2[i] = -2
        vectors_A.append(v2)
        
    vectors_A = np.array(vectors_A, dtype=np.int64)
    
    # Type D: Vectors with four non-zero entries +/- 1
    # Norm^2 = 4. Two Type D vectors u,v are compatible iff dot(u,v) <= 2.
    # If supports intersect in k positions, max possible dot = k.
    # So we need intersection size <= 2 for guaranteed compatibility.
    # But even with intersection size 2, sign choices matter.
    # Key insight: if we select a set of supports with pairwise intersection <= 2,
    # we can choose signs to make all pairwise dots <= 2.
    # Actually, we need to be more careful. Let's use the full vector-level approach.
    
    indices = list(range(D))
    supports = list(itertools.combinations(indices, 4))
    N_S = len(supports)
    
    # Build conflict graph at support level: conflict if intersection >= 3
    support_masks = np.zeros(N_S, dtype=np.uint64)
    for i, s in enumerate(supports):
        mask = 0
        for idx in s:
            mask |= (1 << idx)
        support_masks[i] = mask
    
    # Two supports conflict if they share 3 or more indices
    # In that case, no matter the signs, the dot product can be up to 3, which violates dot <= 2
    # Wait: if intersection is 3, max dot = 3 (all same signs), min dot = -3.
    # We need dot <= 2, so we CAN choose signs to make dot <= 2 even with intersection 3.
    # The real constraint is at the vector level, not just support level.
    # 
    # Let me reconsider. The proper approach:
    # 1. Find a large set of supports where we can assign signs such that all pairwise dots <= 2.
    # 2. This is equivalent to: for any two selected supports S1, S2 with intersection I,
    #    we need sum_{i in I} s1[i]*s2[i] <= 2.
    #
    # A sufficient condition: intersection size <= 2 (then max dot = 2, achievable).
    # With intersection size 3, we need the sum of 3 products of ±1 to be <= 2,
    # i.e., not all three products are +1. This is always achievable by flipping one sign.
    #
    # So actually, the support-level conflict graph with threshold >= 3 is too restrictive.
    # Let's use threshold >= 4 for the support graph, and handle the rest at vector level.
    # 
    # Actually, let's just do the full vector-level search properly.
    # Generate all 2^4 * C(11,4) = 16 * 330 = 5280 Type D vectors.
    # Build a compatibility graph and find a large independent set.
    
    # Generate all Type D vectors
    all_vectors_D = []
    for s in supports:
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(s, signs):
                v[idx] = sign
            all_vectors_D.append(v)
            
    all_vectors_D = np.array(all_vectors_D, dtype=np.int64)
    N_all_D = len(all_vectors_D)  # 5280
    
    # Build compatibility matrix: compatible if dot <= 2
    # This is a 5280x5280 matrix, which is 27.9M entries - manageable
    # But computing it naively is O(N^2 * D) = 5280^2 * 11 ≈ 307M ops, might be slow
    # Let's use block-wise computation
    
    # Actually, let's be smarter. Use the support structure.
    # For two vectors with supports S1, S2:
    # dot = sum over intersection of (sign1 * sign2)
    # If |intersection| <= 2, always compatible (max dot = 2)
    # If |intersection| == 3, compatible unless all three sign products are +1
    # If |intersection| == 4 (same support), dot = sum of 4 sign products, need <= 2
    #   i.e., not all four are +1 (dot=4) and not three +1 and one -1 (dot=2, OK)
    #   So incompatible only if all four signs match (dot=4)
    
    # Let's build the graph using bitmasks and sign patterns
    
    # Assign each vector an index
    # For each pair, check compatibility
    
    # Use a more efficient approach: group by support
    # For each pair of supports, determine which sign combinations are compatible
    
    # First, let's just do a greedy approach with local search at the vector level
    # since exact MIS on 5280 nodes is too expensive
    
    # Greedy: pick vectors in order of "compatibility degree" (fewest conflicts first)
    
    # Compute for each vector, how many other vectors it's compatible with
    # This is expensive, so let's use a heuristic instead
    
    # Heuristic: vectors with more "spread out" signs are more compatible
    # Let's just do a randomized greedy with multiple restarts
    
    rng = np.random.RandomState(42)
    
    best_set = []
    best_size = 0
    
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline - 60:
            break
            
        # Random order
        order = np.arange(N_all_D)
        rng.shuffle(order)
        
        in_set = np.zeros(N_all_D, dtype=bool)
        
        for v_idx in order:
            if time.time() > deadline - 55:
                break
            
            v = all_vectors_D[v_idx]
            
            # Check if v is compatible with all vectors already in the set
            if np.any(in_set):
                in_set_indices = np.where(in_set)[0]
                dots = all_vectors_D[in_set_indices] @ v
                if np.any(dots >= 3):
                    continue
            
            in_set[v_idx] = True
        
        current_size = np.sum(in_set)
        
        if current_size > best_size:
            best_size = current_size
            best_set = np.where(in_set)[0].tolist()
    
    # Local search: try to add more vectors
    if best_set:
        in_set = np.zeros(N_all_D, dtype=bool)
        in_set[best_set] = True
        
        improved = True
        while improved:
            if time.time() > deadline - 20:
                break
            improved = False
            
            # Try adding each vector not in set
            for v_idx in range(N_all_D):
                if in_set[v_idx]:
                    continue
                
                v = all_vectors_D[v_idx]
                if np.any(in_set):
                    in_set_indices = np.where(in_set)[0]
                    dots = all_vectors_D[in_set_indices] @ v
                    if np.any(dots >= 3):
                        continue
                
                in_set[v_idx] = True
                improved = True
    
    best_set = np.where(in_set)[0].tolist()
    
    # Extract the selected vectors
    if best_set:
        selected_vectors_D = all_vectors_D[best_set]
    else:
        selected_vectors_D = np.empty((0, D), dtype=np.int64)
    
    # Try adding Type B vectors (two non-zero ±1, norm²=2)
    # For Type B with norm²=2 and Type A/D with norm²=4:
    # dist² = 2 + 4 - 2*dot >= 4 => dot <= 1
    # Type B vs Type B: dist² = 4 - 2*dot >= 4 => dot <= 0 (supports disjoint or orthogonal)
    
    vectors_B_candidates = []
    for i in range(D):
        for j in range(i+1, D):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * D
                    v[i] = s1
                    v[j] = s2
                    vectors_B_candidates.append(v)
    
    vectors_B_candidates = np.array(vectors_B_candidates, dtype=np.int64)
    
    added_B = []
    added_B_set = set()
    
    for v in vectors_B_candidates:
        if time.time() > deadline - 5:
            break
        vt = tuple(v)
        if vt in added_B_set:
            continue
        
        can_add = True
        # Check against Type A: dot <= 1
        # Type A has one entry ±2. dot = 2*sign_A*sign_B if index matches, else 0.
        # So dot is either 0, 1, or -1. Always <= 1. So Type B is always compatible with Type A.
        
        # Check against Type D: dot <= 1
        if len(selected_vectors_D) > 0:
            dots_D = selected_vectors_D @ v
            if np.any(dots_D >= 2):
                can_add = False
        
        if not can_add:
            continue
        
        # Check against other Type B: dot <= 0
        for u in added_B:
            if np.dot(v, u) > 0:
                can_add = False
                break
        if not can_add:
            continue
        
        added_B.append(v)
        added_B_set.add(vt)
    
    if added_B:
        added_B = np.array(added_B, dtype=np.int64)
        if len(selected_vectors_D) > 0:
            final_vectors = np.vstack([vectors_A, selected_vectors_D, added_B])
        else:
            final_vectors = np.vstack([vectors_A, added_B])
    else:
        if len(selected_vectors_D) > 0:
            final_vectors = np.vstack([vectors_A, selected_vectors_D])
        else:
            final_vectors = vectors_A
    
    return final_vectors
