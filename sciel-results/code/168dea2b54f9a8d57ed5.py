import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Generate all 4-subsets of range(11)
    all_subsets = list(itertools.combinations(range(D), 4))
    n_subsets = len(all_subsets)
    
    # Precompute bitmasks for subsets
    masks = [0] * n_subsets
    for i, s in enumerate(all_subsets):
        m = 0
        for idx in s:
            m |= (1 << idx)
        masks[i] = m
        
    # Precompute popcount for 11-bit numbers
    max_val = 1 << D
    popcount = [0] * max_val
    for i in range(1, max_val):
        popcount[i] = popcount[i >> 1] + (i & 1)
        
    # Precompute compatibility bitmasks
    # compat[i] is a bitmask of nodes j such that j is compatible with i
    # j is compatible with i if popcount(masks[i] & masks[j]) <= 2
    compat = [0] * n_subsets
    for i in range(n_subsets):
        m = 0
        for j in range(n_subsets):
            if i == j:
                continue
            inter = popcount[masks[i] & masks[j]]
            if inter <= 2:
                m |= (1 << j)
        compat[i] = m
        
    best_size = 0
    best_indices = []
    
    # Greedy with random restarts and Local Search
    max_restarts = 2000 # Limited by time
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Shuffle
        candidates = list(range(n_subsets))
        random.shuffle(candidates)
        
        current_set = set()
        blocked_mask = 0
        
        for c in candidates:
            # Check if c is compatible with all in current_set
            # c is compatible if (compat[c] & blocked_mask) == 0?
            # No, blocked_mask contains nodes incompatible with current set.
            # If c is in blocked_mask, it's incompatible with SOME node in set.
            # We need c to be compatible with ALL nodes in set.
            # So c must NOT be in the union of incompatibility sets of all nodes in set.
            # Let's maintain `incompatible_union` mask.
            pass
        
        # Better: Maintain `blocked` mask of all nodes that are incompatible with at least one node in current_set.
        # If we add node c, new_blocked = blocked | (~compat[c] & full_mask) ?
        # No, `compat[c]` is the set of nodes compatible with c.
        # The nodes incompatible with c are `full_mask ^ compat[c] ^ (1 << c)`.
        # Let `incompat[c]` be the mask of nodes incompatible with c.
        # Then `blocked` is the union of `incompat[x]` for all x in current_set.
        # A node c can be added if `c not in current_set` and `c not in blocked`.
        
        # Precompute incompat masks
        full_mask = (1 << n_subsets) - 1
        incompat = [0] * n_subsets
        for i in range(n_subsets):
            # Nodes j such that j != i and popcount(masks[i] & masks[j]) > 2
            # This is full_mask ^ compat[i] ^ (1 << i)
            incompat[i] = full_mask ^ compat[i] ^ (1 << i)
            
        # Restart loop logic
        candidates = list(range(n_subsets))
        random.shuffle(candidates)
        
        current_set = set()
        blocked = 0
        
        for c in candidates:
            if c in current_set:
                continue
            if (blocked >> c) & 1:
                continue
            # c is compatible with all in current_set
            current_set.add(c)
            blocked |= incompat[c]
            
        # Local Search: Try to add more nodes
        improved = True
        while improved:
            improved = False
            # Try to add any node not in current_set and not blocked
            # Iterate through all nodes
            for c in range(n_subsets):
                if c in current_set:
                    continue
                if (blocked >> c) & 1:
                    continue
                # Can add c
                current_set.add(c)
                blocked |= incompat[c]
                improved = True
            if improved:
                # Restart the scan? Or just continue?
                # Since adding a node only increases blocked, we can just continue the loop
                # but we need to be careful not to miss nodes that became unblocked?
                # No, blocked only grows. So if a node was blocked, it stays blocked.
                # So we can just do a single pass if we check all nodes.
                # But adding a node might block others, but it doesn't unblock anyone.
                # So a single pass of checking all nodes is sufficient to find all addable nodes?
                # Yes, because the condition "not blocked" only becomes false, never true.
                # So we can just break the inner loop and do another pass?
                # Actually, if we iterate 0..N, and add node 5, it might block node 10.
                # If we haven't reached 10 yet, we will check it and see it's blocked.
                # If we already passed 10, we missed it? No, if it was unblocked then, we would have added it.
                # So a single pass is enough to saturate the set?
                # Yes.
                break 
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_indices = list(current_set)
            
    # Construct vectors
    vectors = []
    for idx in best_indices:
        subset = all_subsets[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(subset, signs):
                v[idx] = sign
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
