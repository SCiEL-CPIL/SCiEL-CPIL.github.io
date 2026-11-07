import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s buffer
    
    n = 12
    m = 40
    num_masks = 1 << n
    
    # Helper to get list from mask
    def mask_to_list(mask):
        return [i for i in range(n) if (mask >> i) & 1]
    
    # Helper to check subset
    def is_subset(a, b):
        return (a & b) == a

    # 1. Initialization
    # Start with a maximal chain
    F = []
    for i in range(n + 1):
        # Chain: 0, 1, 3, 7, ... (adding lowest bits)
        # Actually, let's just use the first i bits?
        # Mask for size i: (1 << i) - 1
        F.append((1 << i) - 1)
        
    # Greedily add remaining sets
    # Maintain counts
    counts_sub = [0] * num_masks
    counts_sup = [0] * num_masks
    
    def update_counts_add(mask):
        for m in range(num_masks):
            if is_subset(mask, m) and mask != m:
                counts_sup[m] += 1 # mask is subset of m, so m gains a subset? 
                # Wait. counts_sub[m] is count of subsets of m.
                # If mask is subset of m, then mask contributes to counts_sub[m].
                # So counts_sub[m] += 1.
                # And counts_sup[mask] += 1? 
                # counts_sup[mask] is count of supersets of mask.
                # m is a superset of mask. So counts_sup[mask] += 1.
            elif is_subset(m, mask) and m != mask:
                # m is subset of mask
                # m contributes to counts_sub[mask]
                # mask contributes to counts_sup[m]
                pass # Handled in loop logic? 
                
    # This O(N^2) update is 16M ops per add. 27 adds = 400M ops. Might be slow in Python.
    # Let's optimize update.
    # We can precompute list of subsets and supersets for each mask?
    # Subsets of m: iterate submasks of m.
    # Supersets of m: iterate supermasks of m.
    
    # Better:
    # When adding mask 'x':
    # 1. Iterate all submasks s of x (s != x). 
    #    For each s, counts_sup[s] += 1. (x is a superset of s)
    # 2. Iterate all supermasks s of x (s != x).
    #    For each s, counts_sub[s] += 1. (x is a subset of s)
    
    # Number of submasks of x is 2^popcount(x).
    # Number of supermasks of x is 2^(n-popcount(x)).
    # Sum is maximized when popcount is n/2 = 6. 2^6 + 2^6 = 128.
    # So update is O(128) per add! Very fast.
    
    def add_to_counts(mask):
        # Update counts_sup for all strict subsets of mask
        sub = (mask - 1) & mask
        while sub:
            counts_sup[sub] += 1
            sub = (sub - 1) & mask
        # Note: need to handle 0? 
        # If mask is not 0, sub starts from mask-1...
        # Standard submask iteration:
        sub = mask
        while sub:
            sub = (sub - 1) & mask
            if sub != mask: # Strict subset
                 # But this loop includes 0?
                 # If mask=1, sub becomes 0.
                 # We want strict subsets.
                 pass
                 
    # Let's just do the simple O(4096) update for now, it's 4096 * 40 = 160k ops.
    # Wait, 4096 * 40 is for the WHOLE family?
    # No, update is per ADD.
    # Add 27 sets. 27 * 4096 = 110k ops.
    # This is negligible.
    
    def update_counts(mask, sign):
        # sign = 1 for add, -1 for remove
        for m in range(num_masks):
            if m == mask:
                continue
            if (mask & m) == mask: # mask is subset of m
                counts_sub[m] += sign
            elif (m & mask) == m: # m is subset of mask
                counts_sup[m] += sign # Wait.
                # If m is subset of mask, then mask is a superset of m.
                # So m's superset count increases.
                # And mask's subset count increases?
                # No.
                # counts_sub[m] = # subsets of m.
                # counts_sup[m] = # supersets of m.
                
                # Case 1: mask < m (mask is strict subset of m)
                # mask contributes to counts_sub[m].
                # m contributes to counts_sup[mask].
                
                # Case 2: m < mask (m is strict subset of mask)
                # m contributes to counts_sub[mask].
                # mask contributes to counts_sup[m].
                
                # So:
                if (mask & m) == mask: # mask subset of m
                    counts_sub[m] += sign
                    counts_sup[mask] += sign
                else: # m subset of mask
                    counts_sub[mask] += sign
                    counts_sup[m] += sign

    # Initialize F with chain
    F = []
    for i in range(n + 1):
        F.append((1 << i) - 1)
        
    for mask in F:
        update_counts(mask, 1)
        
    # Greedy Add
    while len(F) < m:
        best_mask = -1
        best_score = -1
        for c in range(num_masks):
            if c in F: continue
            score = counts_sub[c] + counts_sup[c]
            if score > best_score:
                best_score = score
                best_mask = c
        F.append(best_mask)
        update_counts(best_mask, 1)
        
    # Hill Climbing
    def get_total_score():
        # Sum of degrees / 2
        # Or just sum counts_sub[c] for c in F?
        # Sum of counts_sub[c] for c in F is the number of pairs (s, c) with s < c.
        total = 0
        for c in F:
            total += counts_sub[c]
        return total
    
    current_score = get_total_score()
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_gain = 0
        best_swap = None
        
        for i, s_i in enumerate(F):
            # Try replacing s_i with c
            for c in range(num_masks):
                if c == s_i: continue
                if c in F: continue
                
                # Calculate gain of c in F \ {s_i}
                # Gain = (counts_sub[c] - (1 if s_i < c else 0)) + (counts_sup[c] - (1 if c < s_i else 0))
                # Loss = counts_sub[s_i] + counts_sup[s_i]
                
                adj_sub = counts_sub[c]
                adj_sup = counts_sup[c]
                
                if (s_i & c) == s_i and s_i != c: # s_i subset of c
                    adj_sub -= 1 # s_i was counted in counts_sub[c]
                elif (c & s_i) == c and c != s_i: # c subset of s_i
                    adj_sup -= 1 # s_i was counted in counts_sup[c]
                    
                gain = adj_sub + adj_sup
                loss = counts_sub[s_i] + counts_sup[s_i]
                
                delta = gain - loss
                if delta > best_gain:
                    best_gain = delta
                    best_swap = (i, c)
                    
        if best_swap:
            i, c = best_swap
            s_i = F[i]
            # Update F
            F[i] = c
            # Update Counts
            update_counts(s_i, -1)
            update_counts(c, 1)
            current_score += best_gain
            improved = True
            
    # Convert to lists
    result = [mask_to_list(mask) for mask in F]
    return result
