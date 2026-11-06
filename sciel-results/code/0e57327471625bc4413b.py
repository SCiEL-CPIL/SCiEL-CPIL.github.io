import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    best_set = []
    
    def is_valid_addition(S, x, sums_list, ge2_list):
        # S is current list of elements
        # sums_list[i] corresponds to S \ {S[i]}
        # ge2_list[i] corresponds to S \ {S[i]}
        
        # 1. Check if x is mean of subset of S
        # Need subset of S with size >= 2 summing to k*x
        # This is ge2 of S. 
        # We don't have global ge2 of S directly in the list, but we can compute it or store it.
        # Let's store global sums and ge2 for S as well.
        pass

    # To simplify, we will store global sums and ge2 for S, and per-element ones.
    # But per-element ones can be derived if we store them.
    
    def try_greedy(order):
        S = []
        # sums_a[i] is bitset of all subset sums of S \ {S[i]}
        # ge2_a[i] is bitset of subset sums of S \ {S[i]} with size >= 2
        sums_a = []
        ge2_a = []
        
        # Global sums for S
        sums_S = 1  # bit 0 set
        ge2_S = 0   # no subset of size >= 2 in empty set
        
        for x in order:
            if len(S) >= 48:
                break
            if time.time() > deadline:
                return S
            
            # Check if adding x is valid
            valid = True
            
            # Check for a = x
            # Subset must be from S. Size >= 2.
            # So check if ge2_S has any bit k*x set for k>=2
            # k*x <= max_sum
            max_sum = sum(S) if S else 0
            # Actually max possible sum is sum(S)
            # We just iterate k
            for k in range(2, len(S) + 1): # k can be up to |S|
                target = k * x
                if target > sum(S): break
                if (ge2_S >> target) & 1:
                    valid = False
                    break
            
            if valid:
                # Check for a in S
                for i, a in enumerate(S):
                    # We need to check if x makes a invalid.
                    # New set for a is (S \ {a}) U {x}
                    # We have sums_a[i] and ge2_a[i] for S \ {a}
                    old_sums = sums_a[i]
                    old_ge2 = ge2_a[i]
                    
                    # New ge2 for (S \ {a}) U {x}
                    # new_ge2 = old_ge2 | ((old_sums & ~1) << x)
                    # We only need to check if any k*a is in new_ge2
                    # k*a could be in old_ge2 (already checked? No, we checked S, not S\{a} U {x})
                    # Wait, if k*a was in old_ge2, it would have been caught when we added the element that completed that sum?
                    # No, the set S was valid. So k*a was NOT in ge2 of (S\{a})?
                    # Actually, S is valid means: for each a in S, no subset of S\{a} of size>=2 sums to k*a.
                    # So old_ge2 does NOT contain any k*a.
                    # So we only need to check the NEW part: ((old_sums & ~1) << x)
                    
                    new_part = (old_sums & ~1) << x
                    
                    for k in range(2, len(S) + 2): # size can be up to |S|+1
                        target = k * a
                        if target > sum(S) + x: break
                        if (new_part >> target) & 1:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                # Add x to S
                S.append(x)
                
                # Update global sums and ge2
                # new_sums_S = sums_S | (sums_S << x)
                # new_ge2_S = ge2_S | ((sums_S & ~1) << x)
                
                # Update per-element lists
                # For existing elements a_i:
                # new_sums_a_i = old_sums_a_i | (old_sums_a_i << x)
                # new_ge2_a_i = old_ge2_a_i | ((old_sums_a_i & ~1) << x)
                
                # For new element x:
                # S \ {x} = S_old
                # new_sums_a_x = sums_S (old)
                # new_ge2_a_x = ge2_S (old)
                
                new_sums_S = sums_S | (sums_S << x)
                new_ge2_S = ge2_S | ((sums_S & ~1) << x)
                
                new_sums_a = []
                new_ge2_a = []
                
                for i in range(len(S) - 1):
                    old_s = sums_a[i]
                    old_g = ge2_a[i]
                    ns = old_s | (old_s << x)
                    ng = old_g | ((old_s & ~1) << x)
                    new_sums_a.append(ns)
                    new_ge2_a.append(ng)
                
                # Append for x
                new_sums_a.append(sums_S)
                new_ge2_a.append(ge2_S)
                
                sums_S = new_sums_S
                ge2_S = new_ge2_S
                sums_a = new_sums_a
                ge2_a = new_ge2_a
        
        return S

    # Try descending order
    set1 = try_greedy(range(512, 0, -1))
    if len(set1) > len(best_set):
        best_set = set1
        
    if time.time() > deadline - 5:
        return best_set

    # Try ascending order
    set2 = try_greedy(range(1, 513))
    if len(set2) > len(best_set):
        best_set = set2
        
    if time.time() > deadline - 5:
        return best_set

    # Try random orders
    for _ in range(20):
        if time.time() > deadline - 2:
            break
        rand_order = list(range(1, 513))
        random.shuffle(rand_order)
        set_r = try_greedy(rand_order)
        if len(set_r) > len(best_set):
            best_set = set_r

    return best_set
