"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    # 90s limit, use 85s to be safe
    deadline = start_time + 85
    
    # 1. Generate all 4-subsets of [1, 16] as bitmasks
    # Elements 1..16 map to bits 0..15
    all_masks = []
    for c in combinations(range(16), 4):
        m = 0
        for x in c:
            m |= (1 << x)
        all_masks.append(m)
        
    n_sets = len(all_masks)
    
    # 2. Precompute pair properties for fast conflict checking
    # For a pair (i, j), we need:
    # K = M_i & M_j
    # DA = M_i & ~K  (Elements in M_i not in K)
    # DB = M_j & ~K  (Elements in M_j not in K)
    # Conflict with new M if:
    # (M & K) == K  AND (M & DA) == 0 AND (M & DB) == 0
    
    # We will store these in lists indexed by pair index
    # Pair index p corresponds to (i, j) where i < j
    # We can iterate i, j and append to lists
    
    pair_K = []
    pair_DA = []
    pair_DB = []
    pair_i = []
    pair_j = []
    
    for i in range(n_sets):
        mi = all_masks[i]
        for j in range(i + 1, n_sets):
            mj = all_masks[j]
            k = mi & mj
            da = mi & ~k
            db = mj & ~k
            pair_K.append(k)
            pair_DA.append(da)
            pair_DB.append(db)
            pair_i.append(i)
            pair_j.append(j)
            
    # To make lookup faster during greedy, we might need to map (i,j) to index in these lists.
    # Or, we can just iterate over the selected sets and compute on the fly?
    # Computing on the fly: 3 bitwise ops.
    # Precomputation: 1 lookup.
    # Given n_sets=1820, pairs ~1.6M.
    # In the greedy loop, we check O(|F|^2) pairs.
    # If |F| is 100, pairs ~5000.
    # 5000 lookups vs 5000 bitwise ops. Bitwise ops are very fast in Python C implementation.
    # Precomputation might not be worth the memory overhead and index management complexity.
    # Let's try computing on the fly first, but optimize the loop structure.
    
    best_family = []
    
    # Helper to check if a mask can be added to a family (list of masks)
    def can_add(m, family):
        n_fam = len(family)
        for i in range(n_fam):
            mi = family[i]
            for j in range(i + 1, n_fam):
                mj = family[j]
                k = mi & mj
                # Check if m forms sunflower with mi, mj
                # Condition: m & k == k AND m & (mi ^ k) == 0 AND m & (mj ^ k) == 0
                # Note: mi ^ k is same as mi & ~k
                if (m & k) == k:
                    if (m & (mi ^ k)) == 0:
                        if (m & (mj ^ k)) == 0:
                            return False
        return True

    # Optimization: 
    # The inner loop is the bottleneck.
    # We can pre-compute the "conflict signature" for each pair in the family?
    # No, the family changes.
    
    # Let's try a different greedy strategy:
    # Instead of checking all pairs for each new item, 
    # we can maintain a list of "forbidden" conditions?
    # A new set M is forbidden if it matches the pattern of any pair (A, B) in F.
    # Pattern for (A, B): 
    #   Core K = A & B
    #   Forbidden1 = A \ K
    #   Forbidden2 = B \ K
    #   M must contain K and be disjoint from Forbidden1 U Forbidden2.
    
    # We can store for each pair (i, j) in F:
    #   K_ij, F_ij (union of forbidden parts)
    #   Check: (M & K_ij) == K_ij AND (M & F_ij) == 0
    
    # This reduces 3 checks to 2 checks per pair.
    
    def can_add_optimized(m, family, pair_props):
        # pair_props is a list of (K, F) for each pair in family
        # We need to update pair_props when family grows.
        # This is O(|F|) to add new pair props.
        # Total O(|F|^2) storage and O(|F|^2) time per insertion?
        # No, we check all existing pairs.
        
        for K, F in pair_props:
            if (m & K) == K:
                if (m & F) == 0:
                    return False
        return True

    def add_to_family(m, family, pair_props, all_masks_set):
        # Check if m can be added
        if not can_add_optimized(m, family, pair_props):
            return False
        
        # Add m
        family.append(m)
        
        # Update pair_props with new pairs involving m
        n = len(family) - 1
        for i in range(n):
            mi = family[i]
            k = mi & m
            f = (mi ^ k) | (m ^ k) # (mi \ k) U (m \ k)
            # Note: mi ^ k is mi & ~k
            pair_props.append((k, f))
            
        return True

    # Randomized Greedy with Restart
    max_restarts = 500
    best_len = 0
    best_family_list = []
    
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
            
        random.shuffle(all_masks)
        
        family = []
        pair_props = []
        
        for m in all_masks:
            if time.time() > deadline - 2:
                break
            
            if add_to_family(m, family, pair_props, set(all_masks)):
                pass # Added
        
        if len(family) > best_len:
            best_len = len(family)
            best_family_list = family[:]
            
            # Early stop if we hit theoretical max? 
            # Max possible is 1820. 
            # Known bounds for sunflower-free families are much lower.
            # If best_len is very high, we can stop.
            if best_len > 1000: # Arbitrary high bound, unlikely to reach
                break

    # Convert best_family_list (masks) back to sorted lists of 1..16
    result = []
    for m in best_family_list:
        s = []
        for i in range(16):
            if m & (1 << i):
                s.append(i + 1)
        result.append(s)
        
    return result
# EVOLVE-BLOCK-END
