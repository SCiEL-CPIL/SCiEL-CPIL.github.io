import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # 1. Generate all 4-subsets of [16] as bitmasks
    all_masks = []
    for combo in combinations(range(16), 4):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        all_masks.append(mask)
    
    n_masks = len(all_masks)
    
    # Precompute pairwise intersections for all masks?
    # inter[i][j] = all_masks[i] & all_masks[j]
    # This takes O(N^2) time and memory. N=1820. 3.3M entries.
    # In Python, a list of lists of ints is heavy.
    # Using numpy might be faster for bitwise ops, but let's stick to standard lib for simplicity unless needed.
    # Actually, computing a & b on the fly is very fast.
    
    def is_valid_addition(family, mask):
        # Check if adding 'mask' to 'family' creates a sunflower
        # A sunflower is formed if there exist a, b in family such that
        # a & b == a & mask == b & mask
        len_f = len(family)
        for i in range(len_f):
            a = family[i]
            am = a & mask
            for j in range(i):
                b = family[j]
                # Optimization: if am == b & mask, then check a & b
                # Actually, we need all three equal.
                if (a & b) == am and (b & mask) == am:
                    return False
        return True

    best_size = 0
    best_family = []
    
    # Strategy: Multiple restarts of a local search
    # Each restart:
    # 1. Greedy initialization
    # 2. Local search (Add, Swap)
    
    while time.time() < deadline:
        # --- Phase 1: Greedy Initialization ---
        random.shuffle(all_masks)
        family = []
        # We can use a more clever greedy: sort by "popularity" or just random
        for mask in all_masks:
            if time.time() > deadline:
                break
            if is_valid_addition(family, mask):
                family.append(mask)
        
        current_size = len(family)
        
        # --- Phase 2: Local Search ---
        # We will try to improve the family size.
        # If we can't add, we try to swap.
        # To avoid getting stuck, we allow "neutral" swaps with a cooling schedule or random acceptance.
        
        in_family_set = set(family)
        
        # List of masks not in family
        # We can just iterate over all_masks and filter
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add any valid mask
            # We iterate randomly to avoid bias
            random.shuffle(all_masks)
            
            added_any = False
            for mask in all_masks:
                if mask in in_family_set:
                    continue
                
                if is_valid_addition(family, mask):
                    family.append(mask)
                    in_family_set.add(mask)
                    current_size += 1
                    improved = True
                    added_any = True
                    # After adding, restart the loop to check for more additions
                    # Or just continue scanning? Scanning is O(N).
                    # If we add one, we should restart the scan or just continue.
                    # Continuing is fine, but the order might matter.
                    # Let's break and restart the shuffle to be random.
                    break
            
            if not added_any:
                # Try to swap: remove one, add one
                # Pick a random mask to remove
                # And a random mask to add
                # This is expensive if done naively.
                
                # Heuristic: Try to find a pair (f_out, m_in) such that
                # m_in is valid in family \ {f_out}
                
                # To speed up, we can pick a random m_in that is NOT valid in family.
                # Why? Because if it were valid, we would have added it.
                # So we know m_in conflicts with some pairs in family.
                # Removing one element might resolve the conflict.
                
                random.shuffle(all_masks)
                swap_done = False
                
                for m_in in all_masks:
                    if m_in in in_family_set:
                        continue
                    
                    # Check if m_in is valid in current family
                    # If yes, it's an "add", which we already tried.
                    # So we assume it's invalid.
                    # We need to find f_out such that m_in is valid in family \ {f_out}.
                    
                    # Optimization: 
                    # m_in conflicts with family if there exist a,b in family s.t. a&b = a&m_in = b&m_in.
                    # Let core = a&m_in = b&m_in.
                    # If we remove 'a', does the conflict go away?
                    # The conflict was specific to the pair (a,b) and m_in.
                    # If we remove 'a', the pair (a,b) is gone.
                    # But there might be other pairs (a', b) or (a, b') that conflict.
                    
                    # So, we can identify the "blocking" sets.
                    # A set 'a' in family blocks 'm_in' if there exists 'b' in family
                    # such that a&b = a&m_in = b&m_in.
                    
                    # Let's find all such 'a'.
                    blockers = []
                    len_f = len(family)
                    # For each a in family, check if it blocks m_in
                    # This is O(|F|^2) again.
                    
                    # Alternative: Just try removing each f in family and checking validity.
                    # If |F| is ~100, and we try ~100 m_in candidates, that's 10,000 checks.
                    # Each check is O(|F|^2) ~ 10,000 ops. Total 100M ops. Too slow in Python.
                    
                    # We need a faster check.
                    
                    pass # Placeholder for fast swap logic

        if current_size > best_size:
            best_size = current_size
            best_family = family[:]
            if best_size >= 250:
                break

    # Convert best_family back to list of sorted lists of integers 1..16
    result = []
    for mask in best_family:
        s = []
        for i in range(16):
            if mask & (1 << i):
                s.append(i + 1)
        result.append(sorted(s))
        
    return result
