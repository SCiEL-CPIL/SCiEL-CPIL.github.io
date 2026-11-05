import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    n = 256
    
    # Base set: 171..256
    base = list(range(171, 257))
    
    # Precompute all_mask
    all_mask = (1 << (n + 1)) - 2  # bits 1..n
    
    # Cache for residue masks to avoid recomputing if same x is checked often?
    # x is unique in the set, but in local search we check many x.
    # We'll compute on the fly.
    
    def get_residue_masks(x):
        masks = [0] * x
        for k in range(1, n + 1):
            r = k % x
            masks[r] |= (1 << k)
        return masks

    def is_valid_addition(S_mask, x):
        """
        Check if x can be added to set represented by S_mask.
        x is NOT in S_mask.
        Returns True if valid.
        """
        # 1. x as 'a': exists b, c in S such that x < b <= c and x | (b+c)
        # Precompute residue masks for x
        res_masks_x = get_residue_masks(x)
        
        # Iterate b in S with b > x
        temp = S_mask
        while temp:
            lsb = temp & -temp
            b = lsb.bit_length() - 1
            temp ^= lsb
            
            if b <= x:
                continue
            
            # b > x
            target_res = (-b) % x
            # Need c in S, c >= b, c % x == target_res
            # M = S_mask & res_masks_x[target_res]
            # Check if M has any bit >= b
            M = S_mask & res_masks_x[target_res]
            if M:
                # Check if M has any bit >= b
                # Mask for bits >= b: all_mask ^ ((1 << b) - 1)
                if M & (all_mask ^ ((1 << b) - 1)):
                    return False
        
        # 2. x as 'b' (middle): exists a, c in S such that a < x <= c and a | (x+c)
        # Iterate a in S with a < x
        temp = S_mask
        while temp:
            lsb = temp & -temp
            a = lsb.bit_length() - 1
            temp ^= lsb
            
            if a >= x:
                continue
            
            # a < x
            # Check c=x case:
            if (2 * x) % a == 0:
                return False
            
            # Check c > x case:
            # Need c in S, c > x, c % a == (-x) % a
            target_res = (-x) % a
            # Iterate c in S with c > x
            temp_c = S_mask
            while temp_c:
                lsb_c = temp_c & -temp_c
                c = lsb_c.bit_length() - 1
                temp_c ^= lsb_c
                
                if c <= x:
                    continue
                
                if c % a == target_res:
                    return False
        
        # 3. x as 'c' (largest): exists a, b in S such that a < b < x and a | (b+x)
        # Collect elements < x
        elements_less_than_x = []
        temp = S_mask
        while temp:
            lsb = temp & -temp
            val = lsb.bit_length() - 1
            temp ^= lsb
            if val < x:
                elements_less_than_x.append(val)
        
        m_small = len(elements_less_than_x)
        for i in range(m_small):
            b = elements_less_than_x[i]
            # b < x
            for j in range(i):
                a = elements_less_than_x[j]
                # a < b
                if (b + x) % a == 0:
                    return False
        
        return True

    def full_valid(S_list):
        S_mask = 0
        for x in S_list:
            S_mask |= (1 << x)
        # Re-check all triples? 
        # For safety, we can just rely on incremental checks if we start from valid base.
        # But for final verification, let's do a quick O(M^3) or O(M^2) check.
        # Given time, we assume incremental is correct.
        # However, to be safe against bugs, we can do a spot check.
        return True

    # Start with base
    S_list = list(base)
    S_list.sort()
    S_mask = 0
    for x in S_list:
        S_mask |= (1 << x)
    S_set = set(S_list)

    best_size = len(S_list)
    best_S_list = S_list
    best_S_mask = S_mask

    # Greedy: try adding smaller numbers
    smaller = list(range(1, 171))
    
    # Try multiple orders
    orders = [
        smaller[::-1], # Descending
        smaller,       # Ascending
    ]
    
    # Add random orders
    rng = random.Random(42)
    for _ in range(20):
        o = smaller[:]
        rng.shuffle(o)
        orders.append(o)

    for order in orders:
        if time.time() > deadline - 5.0:
            break
            
        # Reset to base for each order? Or keep best?
        # Greedy is path-dependent. Resetting allows exploring different paths.
        curr_S_list = list(base)
        curr_S_list.sort()
        curr_S_mask = 0
        for x in curr_S_list:
            curr_S_mask |= (1 << x)
        curr_S_set = set(curr_S_list)

        for x in order:
            if x in curr_S_set:
                continue
            if is_valid_addition(curr_S_mask, x):
                curr_S_list.append(x)
                curr_S_list.sort()
                curr_S_mask |= (1 << x)
                curr_S_set.add(x)
        
        if len(curr_S_list) > best_size:
            best_size = len(curr_S_list)
            best_S_list = curr_S_list
            best_S_mask = curr_S_mask

    # Local Search on best_S_list
    # Try to swap elements
    # To make it fast, we limit iterations.
    if time.time() < deadline - 10.0:
        S_list = best_S_list
        S_mask = best_S_mask
        S_set = set(S_list)
        
        # Precompute candidates not in S
        candidates = [x for x in range(1, n + 1) if x not in S_set]
        
        improved = True
        iters = 0
        max_iters = 500 # Limit local search steps
        
        while improved and iters < max_iters and time.time() < deadline - 5.0:
            improved = False
            iters += 1
            
            # Try to remove each element and add a candidate
            for i in range(len(S_list)):
                if time.time() > deadline - 5.0:
                    break
                
                u = S_list[i]
                # Remove u
                S_mask_no_u = S_mask ^ (1 << u)
                
                # Try to add candidates
                # Prioritize candidates close to u? Or random?
                # Let's try all candidates, but break early if found.
                # To speed up, we can shuffle candidates.
                for v in candidates:
                    if is_valid_addition(S_mask_no_u, v):
                        # Check if new set is valid? 
                        # is_valid_addition ensures v doesn't conflict with S \ {u}.
                        # Since S \ {u} is valid, and v is compatible, the new set is valid.
                        # Update
                        S_list.remove(u)
                        S_list.append(v)
                        S_list.sort()
                        S_mask = S_mask_no_u | (1 << v)
                        S_set.remove(u)
                        S_set.add(v)
                        # Update candidates
                        candidates.remove(v)
                        candidates.append(u)
                        improved = True
                        break
                if improved:
                    break
        
        if len(S_list) > best_size:
            best_size = len(S_list)
            best_S_list = S_list
            best_S_mask = S_mask

    # Final verification
    # Just to be absolutely sure, we can run a full check on best_S_list
    # O(M^3) is ~1M ops, which is < 0.1s.
    def verify(S):
        m = len(S)
        for i in range(m):
            for j in range(i, m):
                for k in range(j, m):
                    a, b, c = S[i], S[j], S[k]
                    if a < b and (b + c) % a == 0:
                        return False
        return True

    if not verify(best_S_list):
        # Fallback to base if something went wrong
        best_S_list = list(base)

    return best_S_list
