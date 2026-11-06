import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85

    # Precompute bitsets
    # Residue[x][r] : bitset of k in [1, n] such that k % x == r
    # Note: k is 1-indexed, bit k-1 is set.
    Residue = [[0] * x for x in range(n + 1)]
    for x in range(2, n + 1):
        for k in range(1, n + 1):
            r = k % x
            Residue[x][r] |= (1 << (k - 1))
    
    # DivisorsSum[s] : bitset of a in [1, n] such that a divides s
    # s can go up to 2n = 512
    DivisorsSum = [0] * (2 * n + 1)
    for s in range(2, 2 * n + 1):
        for a in range(1, n + 1):
            if s % a == 0:
                DivisorsSum[s] |= (1 << (a - 1))

    # Masks for range queries
    # mask_lt[x] : bits 0 to x-2 set (elements 1 to x-1)
    # mask_le[x] : bits 0 to x-1 set (elements 1 to x)
    # mask_gt[x] : bits x to n-1 set (elements x+1 to n)
    # mask_ge[x] : bits x-1 to n-1 set (elements x to n)
    
    mask_lt = [0] * (n + 2)
    mask_le = [0] * (n + 2)
    mask_gt = [0] * (n + 2)
    mask_ge = [0] * (n + 2)
    
    for i in range(1, n + 2):
        mask_le[i] = (1 << i) - 1  # bits 0..i-1
        mask_lt[i] = (1 << (i - 1)) - 1 if i > 1 else 0
        mask_ge[i] = ((1 << n) - 1) ^ ((1 << (i - 1)) - 1) # bits i-1..n-1
        mask_gt[i] = ((1 << n) - 1) ^ ((1 << i) - 1) # bits i..n-1

    def can_add(S, x):
        # Check if S U {x} is valid
        # S is a bitset int. x is 1-indexed int.
        
        # Case 1: x is 'a' (smallest)
        # Need: No b, c in S with b > x, b <= c, such that x | (b+c)
        # b in S & mask_gt[x]
        b_mask = S & mask_gt[x]
        while b_mask:
            # Extract lowest set bit
            lb = b_mask & (-b_mask)
            b = lb.bit_length() # 1-indexed index
            b_mask ^= lb
            
            # c in S & mask_ge[b]
            # Need c % x == (-b) % x
            r = (-b) % x
            c_mask = S & mask_ge[b] & Residue[x][r]
            if c_mask:
                return False
        
        # Case 2: x is 'b' (middle)
        # Need: No a, c in S with a < x, c >= x, such that a | (x+c)
        # a in S & mask_lt[x]
        a_mask = S & mask_lt[x]
        while a_mask:
            lb = a_mask & (-a_mask)
            a = lb.bit_length()
            a_mask ^= lb
            
            # c in S & mask_ge[x]
            # Need c % a == (-x) % a
            r = (-x) % a
            c_mask = S & mask_ge[x] & Residue[a][r]
            if c_mask:
                return False
        
        # Case 3: x is 'c' (largest)
        # Need: No a, b in S with a < b <= x, such that a | (b+x)
        # b in S & mask_le[x]
        b_mask = S & mask_le[x]
        while b_mask:
            lb = b_mask & (-b_mask)
            b = lb.bit_length()
            b_mask ^= lb
            
            # a in S & mask_lt[b]
            # Need a | (b+x)
            # So a must be in DivisorsSum[b+x]
            a_mask = S & mask_lt[b] & DivisorsSum[b + x]
            if a_mask:
                return False
        
        return True

    # Initial set: Upper tail
    S = 0
    for i in range(129, n + 1):
        S |= (1 << (i - 1))
    
    # Greedy add
    for x in range(1, 129):
        if time.time() > deadline - 5:
            break
        if can_add(S, x):
            S |= (1 << (x - 1))
    
    # Local Search
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try to add any missing element
        for x in range(1, n + 1):
            if S & (1 << (x - 1)):
                continue
            if can_add(S, x):
                S |= (1 << (x - 1))
                improved = True
                break # Restart scan from beginning as set changed
        
        if not improved:
            # Try swaps: remove one, add one
            # Or remove multiple?
            # For now, try single swaps that increase size is impossible (size stays same).
            # We want to maximize size. So we only add if we can.
            # If we can't add, we might need to remove some to add more.
            # Let's try to find a set of conflicts for a missing x, and if small, remove them.
            
            for x in range(1, n + 1):
                if S & (1 << (x - 1)):
                    continue
                
                # Find conflicts
                conflicts = 0
                # Case 1: x is a
                b_mask = S & mask_gt[x]
                while b_mask:
                    lb = b_mask & (-b_mask)
                    b = lb.bit_length()
                    b_mask ^= lb
                    r = (-b) % x
                    c_mask = S & mask_ge[b] & Residue[x][r]
                    if c_mask:
                        conflicts |= c_mask
                        conflicts |= (1 << (b - 1))
                
                # Case 2: x is b
                a_mask = S & mask_lt[x]
                while a_mask:
                    lb = a_mask & (-a_mask)
                    a = lb.bit_length()
                    a_mask ^= lb
                    r = (-x) % a
                    c_mask = S & mask_ge[x] & Residue[a][r]
                    if c_mask:
                        conflicts |= c_mask
                        conflicts |= (1 << (a - 1))
                
                # Case 3: x is c
                b_mask = S & mask_le[x]
                while b_mask:
                    lb = b_mask & (-b_mask)
                    b = lb.bit_length()
                    b_mask ^= lb
                    a_mask = S & mask_lt[b] & DivisorsSum[b + x]
                    if a_mask:
                        conflicts |= a_mask
                        conflicts |= (1 << (b - 1))
                
                # If conflicts are few, try removing them
                if conflicts and conflicts.bit_count() <= 3:
                    new_S = S & ~conflicts
                    new_S |= (1 << (x - 1))
                    if can_add(new_S, x): # Should be true if conflicts were exhaustive
                         # Actually, removing conflicts might introduce new issues? No, subset of valid set is valid.
                         # But we added x. So we just need to check if new_S is valid.
                         # Since new_S \ {x} is subset of S, it's valid.
                         # And we identified all conflicts with x.
                         # So new_S should be valid.
                         S = new_S
                         improved = True
                         break

    # Convert bitset to list
    result = []
    for i in range(1, n + 1):
        if S & (1 << (i - 1)):
            result.append(i)
    
    return result
