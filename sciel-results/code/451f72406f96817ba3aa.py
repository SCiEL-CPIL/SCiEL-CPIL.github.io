import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Initial valid tail
    S_tail = set(range(171, 257))
    
    # Precompute conflicts with tail for x in 1..170
    # conflict_tail[x] is True if there exist b, c in S_tail such that x < b <= c and x | (b+c)
    conflict_tail = [False] * 171
    # Optimize: b+c ranges from 171+171=342 to 256+256=512
    # For a given x, we check multiples k*x in [342, 512]
    # and see if k*x can be written as b+c with b,c in S_tail, b<=c.
    # b = k*x - c.
    # Conditions:
    # 1. c in S_tail (171..256)
    # 2. b in S_tail (171..256)
    # 3. b <= c
    # 4. x < b (automatically true if b>=171 and x<=170)
    
    for x in range(1, 171):
        max_k = 512 // x
        for k in range(2, max_k + 1):
            s = k * x
            if s < 342:
                continue
            if s > 512:
                break
            # Check if s = b + c with b, c in [171, 256] and b <= c
            # c >= s/2
            # c <= 256
            # b = s - c >= 171 => c <= s - 171
            c_min = max(171, (s + 1) // 2)
            c_max = min(256, s - 171)
            if c_min <= c_max:
                # There exists a valid pair
                conflict_tail[x] = True
                break
                
    S_small = set()
    
    # Greedy addition in ascending order
    for x in range(1, 171):
        if time.time() > deadline - 10:
            break
            
        if conflict_tail[x]:
            continue
            
        # Check all potential conflicts with x as the new element
        valid = True
        
        # Case 1: x as 'a' in a < b <= c, where b,c are in S_tail
        # Already handled by conflict_tail[x]
        
        # Case 2: a in S_small, x as 'b', c in S_tail, a < x <= c
        # Check if a | (x + c) for any a in S_small, c in S_tail with x <= c
        for a in S_small:
            if a >= x:
                continue  # Need a < x
            # Check if there exists c in S_tail (171..256) with x <= c and a | (x+c)
            # x + c must be a multiple of a
            # Let x + c = k*a, so c = k*a - x
            # Need 171 <= c <= 256 and x <= c
            # So max(171, x) <= k*a - x <= 256
            # max(171, x) + x <= k*a <= 256 + x
            c_low = max(171, x)
            c_high = 256
            k_min = (c_low + x + a - 1) // a
            k_max = (c_high + x) // a
            for k in range(k_min, k_max + 1):
                c = k * a - x
                if c_low <= c <= c_high:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
            
        # Case 3: a, b in S_small, x as 'c', a < b <= x
        # Check if a | (b + x) for any a, b in S_small with a < b <= x
        for a in S_small:
            for b in S_small:
                if a < b <= x:
                    if (b + x) % a == 0:
                        valid = False
                        break
            if not valid:
                break
        
        if valid:
            S_small.add(x)
    
    # Local search: try to improve by swapping elements
    current_set = set(S_tail | S_small)
    
    def is_valid_set(S):
        """Check if set S satisfies the property."""
        S_list = sorted(S)
        for i, a in enumerate(S_list):
            for j, b in enumerate(S_list):
                if j <= i:
                    continue
                for k, c in enumerate(S_list):
                    if k < j:
                        continue
                    if a < b <= c and (b + c) % a == 0:
                        return False
        return True
    
    def try_swap(S, remove_elem, add_elem):
        """Try swapping remove_elem with add_elem in set S."""
        if remove_elem == add_elem:
            return False
        if add_elem in S:
            return False
        
        # Try the swap
        new_S = set(S)
        new_S.discard(remove_elem)
        new_S.add(add_elem)
        
        # Check validity
        if is_valid_set(new_S):
            S.clear()
            S.update(new_S)
            return True
        return False
    
    # Try local search improvements
    search_time_limit = deadline - 5
    while time.time() < search_time_limit:
        improved = False
        candidates = list(range(1, 171))
        for elem in current_set:
            if elem > 170:
                continue
            for candidate in candidates:
                if candidate in current_set:
                    continue
                if try_swap(current_set, elem, candidate):
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    result = list(current_set)
    return result
