import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 256
    # Precompute reciprocals
    recip = [0.0] * (N + 1)
    for i in range(1, N + 1):
        recip[i] = 1.0 / i
        
    # Precompute suffix sums of reciprocals for upper bound estimation
    # suffix_sum[i] = sum(1/j for j in range(i, N+1))
    suffix_sum = [0.0] * (N + 2)
    for i in range(N, 0, -1):
        suffix_sum[i] = suffix_sum[i + 1] + recip[i]

    best_sum = 0.0
    best_set = []
    
    # Initial greedy solution to establish a baseline
    # Greedily pick smallest numbers (highest weight) that don't form 3-AP
    def get_greedy_solution():
        chosen = set()
        current_sum = 0.0
        for i in range(1, N + 1):
            # Check if i conflicts with chosen
            conflict = False
            for a in chosen:
                # Check if i, a, b form AP
                # b = 2i - a or 2a - i or (i+a)/2
                b1 = 2 * i - a
                if b1 in chosen and b1 != i and b1 != a:
                    conflict = True
                    break
                b2 = 2 * a - i
                if b2 in chosen and b2 != i and b2 != a:
                    conflict = True
                    break
                if (i + a) % 2 == 0:
                    b3 = (i + a) // 2
                    if b3 in chosen and b3 != i and b3 != a:
                        conflict = True
                        break
            if not conflict:
                chosen.add(i)
                current_sum += recip[i]
        return current_sum, list(chosen)

    best_sum, best_set = get_greedy_solution()
    
    # Bitmask representation
    # Bit i is set if number i is available
    full_mask = (1 << (N + 1)) - 2  # Bits 1..N set (bit 0 unused)
    
    # Precompute for each pair (a, b) in 1..N, the set of c such that {a,b,c} is 3-AP
    # Actually, we need to know, given a new element v and existing chosen set C,
    # which elements u become forbidden.
    # Forbidden(u) if exists w in C such that {u, v, w} is 3-AP.
    # This depends on C.
    
    # To speed up, we can precompute for each v, a list of (w, u) pairs?
    # Or simply iterate through chosen elements.
    
    # Optimization: 
    # The "Available" mask is updated by removing forbidden elements.
    # When we add v, we remove u if (u, v, w) is AP for some w in Chosen.
    
    def dfs(available_mask, chosen_sum, chosen_list, depth):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return False # Signal timeout
            
        # Pruning: Upper bound
        # Sum of reciprocals of all available elements
        # We can approximate this or compute it.
        # Computing sum of reciprocals of bits in mask is expensive if done naively.
        # However, N=256 is small.
        # We can precompute a lookup? No, mask is 256 bits.
        # Let's just use the suffix sum of the smallest available index?
        # Or simply sum(1/i for i in available).
        # This is O(N) per node. With many nodes, this might be slow.
        
        # Better bound:
        # Let min_avail be the smallest index in available_mask.
        # The max possible sum is current_sum + sum(1/i for i in range(min_avail, N+1)).
        # This is an upper bound because we are picking a subset.
        # But we might not be able to pick all of them.
        # A tighter bound: Greedy coloring / packing bound?
        # For now, let's use the suffix sum of the minimum available index.
        # If available_mask is 0, bound is current_sum.
        
        if available_mask == 0:
            if chosen_sum > best_sum:
                best_sum = chosen_sum
                best_set = list(chosen_list)
            return True
            
        # Find the smallest available index
        # We process in order of index to maximize weight early?
        # Yes, smaller index -> larger 1/i.
        # Find lowest set bit.
        # lowbit = available_mask & -available_mask
        # idx = lowbit.bit_length() - 1
        
        # To make the bound tighter, we should consider that we can't pick all.
        # But for now, let's try the simple bound first.
        
        # Extract min index
        # Note: bit 0 is unused.
        # We want the smallest i such that bit i is set.
        # Python int: 
        # if available_mask == 0: return
        # We can iterate? No.
        # Trick: 
        # temp = available_mask
        # while temp & 1 == 0: temp >>= 1 ... no, bit 0 is 0.
        # Let's just find the first set bit.
        
        # Since N is small, we can just iterate 1 to N to find the first available?
        # That is O(N).
        # Or use bit operations.
        # lowest_bit = available_mask & -available_mask
        # idx = lowest_bit.bit_length() - 1
        
        # But we must ensure idx >= 1.
        # Since bit 0 is always 0 in our mask, this is fine.
        
        if available_mask == 0:
             return True
             
        lowest_bit = available_mask & -available_mask
        idx = lowest_bit.bit_length() - 1
        
        # Upper bound: current_sum + sum(1/j for j in available)
        # We can't sum all available quickly.
        # Approximation: current_sum + suffix_sum[idx]
        # This assumes we can pick all numbers from idx to N.
        if chosen_sum + suffix_sum[idx] <= best_sum:
            return True
            
        # Branch 1: Include idx
        # Check if idx is compatible with chosen_list
        # If yes, new_available = available_mask without idx and without forbidden elements
        # Forbidden elements: u such that {u, idx, w} is 3-AP for some w in chosen_list
        
        new_chosen_list = chosen_list + [idx]
        new_available = available_mask & ~lowest_bit
        
        # Calculate forbidden mask
        forbidden = 0
        for w in chosen_list:
            # u = 2*idx - w
            u1 = 2 * idx - w
            if 1 <= u1 <= N:
                forbidden |= (1 << u1)
            # u = 2*w - idx
            u2 = 2 * w - idx
            if 1 <= u2 <= N:
                forbidden |= (1 << u2)
            # u = (idx + w) / 2
            if (idx + w) % 2 == 0:
                u3 = (idx + w) // 2
                if 1 <= u3 <= N:
                    forbidden |= (1 << u3)
                    
        new_available &= ~forbidden
        
        # Recurse
        # Pass new_chosen_list? 
        # Storing list is expensive. 
        # We can pass a bitmask for chosen too? 
        # But we need to iterate chosen to compute forbidden.
        # If we pass a bitmask for chosen, iterating is slow.
        # Given N=256, chosen size is small (< 100 usually).
        # List is fine.
        
        # Optimization: If time is tight, stop
        if time.time() > deadline:
            return False
            
        dfs(new_available, chosen_sum + recip[idx], new_chosen_list, depth + 1)
        
        if time.time() > deadline:
            return False
            
        # Branch 2: Exclude idx
        # new_available is just available_mask without idx
        dfs(available_mask & ~lowest_bit, chosen_sum, chosen_list, depth + 1)
        
        return True

    # Start DFS
    # We want to explore including small numbers first.
    # The DFS above picks the smallest available number at each step.
    # This aligns with maximizing weight.
    
    # One issue: The bound `suffix_sum[idx]` is very loose.
    # We can improve by using a greedy packing bound?
    # For now, let's run and see if it finishes.
    
    # To prevent stack overflow or excessive recursion, we rely on time check.
    
    # Initial call
    # available_mask has bits 1..N set
    initial_mask = 0
    for i in range(1, N + 1):
        initial_mask |= (1 << i)
        
    dfs(initial_mask, 0.0, [], 0)
    
    if not best_set:
        return [1]
        
    return best_set
