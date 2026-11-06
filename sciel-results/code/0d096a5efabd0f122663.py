import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 45.0  # 45s margin for 60s limit

    n = 256
    # Start with the largest known valid set based on 3*min > 2*max
    # 3*K > 2*256 => K > 170.66 => K=171
    K = 171
    S = list(range(K, n + 1))
    
    # To speed up checks, we can maintain a set of S
    S_set = set(S)
    
    # Iterate downwards from K-1 to 1
    for a in range(K - 1, 0, -1):
        if time.time() > deadline:
            break
            
        # Check if adding 'a' is valid
        # Condition: For all b, c in S such that a < b <= c, a does NOT divide b+c
        # If we find any b, c such that a divides b+c, we cannot add a.
        
        valid = True
        # Optimization: We only need to check pairs (b, c) where b+c is a multiple of a.
        # Since b, c > a, b+c > 2a.
        # Multiples of a in range [2*a+2, 2*256]
        
        # Naive check is fine for now, but let's be slightly smarter.
        # Iterate through b in S where b > a
        # For each b, check if there exists c in S such that c >= b and (b+c) % a == 0.
        # (b+c) % a == 0 => c % a == (-b) % a.
        
        # Let's just do the double loop but break early.
        # S is sorted.
        # We can precompute S elements > a.
        # Actually, S changes as we add elements.
        
        # For each b in S:
        #   if b <= a: continue
        #   For each c in S:
        #     if c < b: continue
        #     if (b+c) % a == 0:
        #        valid = False
        #        break
        #   if not valid: break

        # This is O(|S|^2) per a. |S| ~ 100. 100^2 = 10000. 170 * 10000 = 1.7M ops. Very fast.
        
        for b in S:
            if b <= a:
                continue
            for c in S:
                if c < b:
                    continue
                if (b + c) % a == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            S.append(a)
            S_set.add(a)
            S.sort() # Keep sorted for potential optimizations or just consistency, though not strictly needed for logic if we iterate carefully.
            # Actually, if we don't sort, the inner loop `if c < b` works fine on unsorted list?
            # Yes, we just check all pairs.
            # Sorting is O(N log N) which is small.
            
    return S
