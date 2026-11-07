import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s limit, 10s margin
    
    n = 128
    # Initialize with upper half
    mask = 0
    for i in range(65, n + 1):
        mask |= (1 << (i - 1))
        
    # Function to check if adding x to current mask creates a forbidden triple
    # We only need to check triples involving x, as the rest of the set is valid.
    def is_valid_addition(x, current_mask):
        # x is 1-indexed value
        # new_mask would be current_mask | (1 << (x-1))
        # We check if there exist distinct u, v in current_mask
        # such that x+u, x+v, u+v are in new_mask.
        
        # Optimization: 
        # u, v must be in current_mask.
        # Let's iterate through set bits of current_mask.
        # But we need pairs.
        
        # A faster way:
        # For each u in current_mask:
        #   For each v in current_mask, v > u:
        #      Check conditions.
        
        # To get list of elements in current_mask:
        # We can precompute or just iterate.
        # Since we do this many times, let's keep a list of elements in the set.
        pass

    # Better: Maintain a list `elements` in addition to mask.
    elements = list(range(65, n + 1))
    
    # Greedy addition
    # Try adding x from 64 down to 1
    # We can try multiple orders or just one.
    # Let's try 64 down to 1.
    
    for x in range(64, 0, -1):
        if time.time() > deadline:
            break
            
        # Check validity of adding x
        valid = True
        # Iterate over pairs in current elements
        # Note: elements does not contain x
        m = len(elements)
        for i in range(m):
            u = elements[i]
            # Check sums involving x and u
            sum_xu = x + u
            sum_xu_bit = 1 << (sum_xu - 1) if sum_xu <= n else 0
            if sum_xu_bit and not (mask & sum_xu_bit):
                # sum_xu is not in current mask. 
                # But it could be x itself? No, sum_xu > x.
                # It could be in the set we are trying to add? No, we are checking if the NEW set is valid.
                # The new set contains x.
                # So if sum_xu == x, that's impossible for u>=1.
                # If sum_xu is not in current_mask and sum_xu != x, then sum_xu is NOT in new_mask.
                # So condition "x+u in new_mask" fails.
                # Thus this pair (u, v) cannot form a forbidden triple with x.
                # Wait, we need ALL three sums to be in new_mask.
                # If one is missing, the triple is safe.
                # So if sum_xu is not in current_mask and sum_xu != x, then for any v, the triple {x, u, v} is safe?
                # No, we need to check if there EXISTS a v.
                # If for a specific u, sum_xu is not in new_mask, then no v can form a forbidden triple with x and u.
                # So we can skip checking v for this u.
                continue
                
            # sum_xu is in new_mask (either in current_mask or sum_xu == x which is impossible)
            # So sum_xu is in current_mask.
            
            for j in range(i + 1, m):
                v = elements[j]
                
                # Check x + v
                sum_xv = x + v
                if sum_xv > n:
                    continue
                sum_xv_bit = 1 << (sum_xv - 1)
                if not (mask & sum_xv_bit) and sum_xv != x:
                    continue
                # sum_xv is in new_mask
                
                # Check u + v
                sum_uv = u + v
                if sum_uv > n:
                    continue
                sum_uv_bit = 1 << (sum_uv - 1)
                if not (mask & sum_uv_bit) and sum_uv != x:
                    continue
                # sum_uv is in new_mask
                
                # All three sums are in new_mask.
                # Check distinctness: x, u, v are distinct.
                # x is not in elements, so x != u and x != v.
                # u != v because i != j.
                # So {x, u, v} are distinct.
                valid = False
                break
            if not valid:
                break
        
        if valid:
            mask |= (1 << (x - 1))
            elements.append(x)
            
    return elements
