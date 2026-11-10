"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 55  # 55 seconds safety margin
    
    # Precompute 4-APs as bitmasks for the other 3 elements
    # For each x, store a list of masks. If (S & mask) == mask, conflict.
    ap_masks = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # Mask for the other 3 elements
            mask_a = (1 << b) | (1 << c) | (1 << e)
            mask_b = (1 << a) | (1 << c) | (1 << e)
            mask_c = (1 << a) | (1 << b) | (1 << e)
            mask_e = (1 << a) | (1 << b) | (1 << c)
            
            ap_masks[a].append(mask_a)
            ap_masks[b].append(mask_b)
            ap_masks[c].append(mask_c)
            ap_masks[e].append(mask_e)

    def conflicts_with(S_bits, x):
        for mask in ap_masks[x]:
            if (S_bits & mask) == mask:
                return True
        return False

    # 1. Greedy Construction
    S_bits = 0
    for x in range(1, N + 1):
        if time.time() > deadline:
            break
        if not conflicts_with(S_bits, x):
            S_bits |= (1 << x)
            
    # 2. Local Search
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        
        # Get list of elements in S
        S_list = [i for i in range(1, N + 1) if (S_bits >> i) & 1]
        
        for x in S_list:
            if time.time() > deadline:
                break
            
            # Tentatively remove x
            S_bits_new = S_bits & ~(1 << x)
            
            # Find candidates to add
            # We want to see if we can add > 1 elements
            # Optimization: Just check if there are any candidates at all?
            # No, we need to count how many we can add.
            
            added_count = 0
            S_bits_candidate = S_bits_new
            
            # Iterate through all y not in S
            # To speed up, we can break early if we have added enough to know it's a win?
            # We need added_count > 1 to improve size.
            # So if we add 2, we can stop checking for more for this x?
            # Yes, because we only care if size increases.
            
            for y in range(1, N + 1):
                if time.time() > deadline:
                    break
                if (S_bits_new >> y) & 1:
                    continue
                if not conflicts_with(S_bits_candidate, y):
                    S_bits_candidate |= (1 << y)
                    added_count += 1
                    if added_count > 1:
                        # We have a strict improvement
                        # We can break early to save time? 
                        # Yes, we know it's a win. 
                        # But we might add even more. 
                        # Let's just take this state.
                        break
                        
            if added_count > 1:
                S_bits = S_bits_candidate
                improved = True
            # If added_count <= 1, we revert (S_bits remains unchanged)
            
    # 3. Final Greedy Pass
    for x in range(1, N + 1):
        if time.time() > deadline:
            break
        if not ((S_bits >> x) & 1) and not conflicts_with(S_bits, x):
            S_bits |= (1 << x)
            
    # Convert back to list
    result = [i for i in range(1, N + 1) if (S_bits >> i) & 1]
    return result
# EVOLVE-BLOCK-END
