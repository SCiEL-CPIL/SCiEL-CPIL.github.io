"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin within 90s limit
    
    n = 256
    
    # The upper 1/3 construction is known to be valid and has size 86.
    # This is our baseline.
    S_upper_1_3 = list(range(171, 257))
    best_set = set(S_upper_1_3)
    
    # We will try to improve upon this by using a greedy approach that tries to add
    # elements from the lower range, but we need to be smart about it.
    
    # A key insight from the inspirations is that a simple greedy approach (descending or ascending)
    # might find a different local optimum. Let's try a few different strategies.
    
    def is_valid_addition(x, S):
        """
        Check if adding x to set S maintains the property:
        For every a < b <= c in the resulting set, a does NOT divide b+c.
        """
        S_list = sorted(S)
        s_len = len(S_list)
        
        # Case 1: x is the smallest element a
        # Need: x < b <= c, and x does not divide b+c
        # Since we're checking against existing S, all elements in S are not necessarily > x.
        # But for the new triples involving x:
        # 1. (x, b, c) where x < b <= c. b, c in S.
        for i in range(s_len):
            b = S_list[i]
            if b <= x:
                continue
            # b > x
            # c must be >= b and in S
            for j in range(i, s_len):
                c = S_list[j]
                if (b + c) % x == 0:
                    return False
        
        # 2. (a, x, c) where a < x <= c. a in S, c in S.
        for a in S_list:
            if a >= x:
                continue
            for c in S_list:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # 3. (a, b, x) where a < b <= x. a in S, b in S.
        for i in range(s_len):
            a = S_list[i]
            for j in range(i, s_len):
                b = S_list[j]
                if a < b and b <= x:
                    if (b + x) % a == 0:
                        return False
        
        # 4. (a, x, x) where a < x. a in S.
        for a in S_list:
            if a < x:
                if (2 * x) % a == 0:
                    return False
                    
        return True

    # Strategy 1: Greedy descending, starting from empty set
    S_desc = set()
    for x in range(n, 0, -1):
        if time.time() > deadline:
            break
        if is_valid_addition(x, S_desc):
            S_desc.add(x)
    
    if len(S_desc) > len(best_set):
        best_set = S_desc
        
    # Strategy 2: Greedy ascending, starting from empty set
    if time.time() < deadline:
        S_asc = set()
        for x in range(1, n + 1):
            if time.time() > deadline:
                break
            if is_valid_addition(x, S_asc):
                S_asc.add(x)
        
        if len(S_asc) > len(best_set):
            best_set = S_asc

    # Strategy 3: Start with upper 1/3, try to add elements from lower range
    if time.time() < deadline:
        S_extend = set(S_upper_1_3)
        for x in range(170, 0, -1):
            if time.time() > deadline:
                break
            if is_valid_addition(x, S_extend):
                S_extend.add(x)
        
        if len(S_extend) > len(best_set):
            best_set = S_extend

    # Strategy 4: Local improvement on the best set found so far
    if time.time() < deadline:
        # Try to swap elements to increase size? Or just try adding more?
        # Since we've tried greedy, let's try a random restart with a different order.
        import random
        random.seed(42)
        for _ in range(10):
            if time.time() > deadline:
                break
            order = list(range(1, n + 1))
            random.shuffle(order)
            S_rand = set()
            for x in order:
                if time.time() > deadline:
                    break
                if is_valid_addition(x, S_rand):
                    S_rand.add(x)
            
            if len(S_rand) > len(best_set):
                best_set = S_rand

    # Final check: ensure the best set is valid (paranoid check)
    # If not, fall back to upper 1/3
    def _is_valid(S):
        S = sorted(S)
        n_set = len(S)
        for i in range(n_set):
            a = S[i]
            for j in range(i+1, n_set):
                b = S[j]
                for k in range(j, n_set):
                    c = S[k]
                    if (b + c) % a == 0:
                        return False
        return True

    if not _is_valid(best_set):
        best_set = set(S_upper_1_3)
        
    return list(best_set)
# EVOLVE-BLOCK-END
