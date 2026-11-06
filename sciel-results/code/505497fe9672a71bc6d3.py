import time

def construct():
    start_time = time.time()
    limit = 512
    
    # Step 1: Base-3 construction
    S = set()
    for x in range(1, limit + 1):
        temp = x
        valid = True
        while temp > 0:
            if temp % 3 == 2:
                valid = False
                break
            temp //= 3
        if valid:
            S.add(x)
    
    # Step 2: Greedy addition
    # We try to add elements from [1, 512] not in S
    # We do this in a loop until no progress is made
    improved = True
    while improved:
        improved = False
        # We can try adding in increasing order or random order
        # Increasing order is deterministic
        for x in range(1, limit + 1):
            if x in S:
                continue
            
            # Check if x can be added
            can_add = True
            
            # Case 1: x is 1st term: x, x+d, x+2d, x+3d
            # Need x+d, x+2d, x+3d in S
            # d >= 1
            # x+3d <= 512 => d <= (512-x)//3
            max_d = (limit - x) // 3
            for d in range(1, max_d + 1):
                if (x + d in S) and (x + 2*d in S) and (x + 3*d in S):
                    can_add = False
                    break
            
            if not can_add:
                continue
                
            # Case 2: x is 2nd term: x-d, x, x+d, x+2d
            # Need x-d, x+d, x+2d in S
            # x-d >= 1 => d <= x-1
            # x+2d <= 512 => d <= (512-x)//2
            max_d = min(x - 1, (limit - x) // 2)
            for d in range(1, max_d + 1):
                if (x - d in S) and (x + d in S) and (x + 2*d in S):
                    can_add = False
                    break
            
            if not can_add:
                continue

            # Case 3: x is 3rd term: x-2d, x-d, x, x+d
            # Need x-2d, x-d, x+d in S
            # x-2d >= 1 => d <= (x-1)//2
            # x+d <= 512 => d <= 512-x
            max_d = min((x - 1) // 2, limit - x)
            for d in range(1, max_d + 1):
                if (x - 2*d in S) and (x - d in S) and (x + d in S):
                    can_add = False
                    break
            
            if not can_add:
                continue

            # Case 4: x is 4th term: x-3d, x-2d, x-d, x
            # Need x-3d, x-2d, x-d in S
            # x-3d >= 1 => d <= (x-1)//3
            max_d = (x - 1) // 3
            for d in range(1, max_d + 1):
                if (x - 3*d in S) and (x - 2*d in S) and (x - d in S):
                    can_add = False
                    break
            
            if can_add:
                S.add(x)
                improved = True
                
            # Time check
            if time.time() - start_time > 50:
                return sorted(S)
                
    return sorted(S)
