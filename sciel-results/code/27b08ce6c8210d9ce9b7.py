import random

def construct():
    n = 128
    # Indices 0 to 127 represent values 1 to 128
    
    # 1. Precompute all "bad" sets (sets of indices that form a violation)
    bad_sets = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            for c in range(b + 1, n + 1):
                s_ab = a + b
                s_ac = a + c
                s_bc = b + c
                if s_bc > n:
                    break # Since c is increasing, b+c will only get larger
                
                # The values involved are a, b, c, s_ab, s_ac, s_bc
                # We need all of these to be in the selected set S.
                # Convert to 0-based indices
                vals = {a, b, c, s_ab, s_ac, s_bc}
                # Check if any value is out of bounds (shouldn't happen due to break, but s_ab, s_ac could be > n? 
                # If s_bc <= n, then s_ab < s_bc <= n and s_ac < s_bc <= n. So all are <= n.)
                
                # Create a frozenset of 0-based indices
                idx_set = frozenset(v - 1 for v in vals)
                bad_sets.append(idx_set)
                
    # 2. Greedy Hitting Set to find a set Z to remove
    # We want to remove minimum elements to hit all bad_sets.
    # Actually, we want to MAXIMIZE kept elements, so MINIMIZE removed.
    
    def greedy_hit(bad_sets_list):
        Z = set()
        active = list(bad_sets_list)
        
        while active:
            # Count frequency of each index in active bad sets
            freq = {}
            for bs in active:
                for idx in bs:
                    freq[idx] = freq.get(idx, 0) + 1
            
            # Pick index with max frequency
            if not freq:
                break
            max_idx = max(freq, key=freq.get)
            Z.add(max_idx)
            
            # Remove all bad sets containing max_idx
            new_active = []
            for bs in active:
                if max_idx not in bs:
                    new_active.append(bs)
            active = new_active
            
        return Z

    # Run greedy multiple times with random tie-breaking to find a better Z
    best_Z = None
    best_size = -1
    
    for _ in range(20): # 20 restarts
        Z = greedy_hit(bad_sets)
        size = n - len(Z)
        if size > best_size:
            best_size = size
            best_Z = Z
            
    # 3. Local Search: Try to add back removed elements
    # Start with the best Z found
    Z = set(best_Z)
    
    # Helper to check if S (complement of Z) is valid
    # S is valid if no bad_set is a subset of S
    # i.e., for all bs in bad_sets, bs is NOT a subset of (all_indices - Z)
    # i.e., for all bs in bad_sets, bs INTERSECT Z is NOT empty
    
    def is_valid(Z_set):
        for bs in bad_sets:
            if not (bs & Z_set):
                return False
        return True

    # Try to remove elements from Z (add to S)
    changed = True
    while changed:
        changed = False
        # Try removing each element from Z
        # Random order to avoid bias
        z_list = list(Z)
        random.shuffle(z_list)
        for x in z_list:
            if x not in Z:
                continue
            # Try removing x from Z
            Z_test = Z.copy()
            Z_test.discard(x)
            if is_valid(Z_test):
                Z = Z_test
                changed = True
                break # Restart loop as Z changed significantly? Or just continue? 
                # Breaking restarts the shuffle, which is good for exploration.
                
    # Final set S
    S = [i + 1 for i in range(n) if i not in Z]
    
    return S
