import time

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Proven valid tail: for a >= 171, any b,c > a gives b+c > 2a >= 2*171=342
    # but b+c mod a = 0 requires b+c >= a (true) and b+c = k*a for some k
    # Since b,c <= 256, b+c <= 512. For a >= 171, k can be 1 or 2.
    # k=1: b+c = a, but b > a and c >= b > a, so b+c > 2a > a. Impossible.
    # k=2: b+c = 2a. With b,c in [a+1, 256], b+c in [2a+2, 512]. 
    #   2a in [342, 512]. Need b+c = 2a with b > a, c >= b.
    #   b >= a+1, c >= b >= a+1, so b+c >= 2a+2 > 2a. Impossible!
    # So [171, 256] is valid.
    
    base_tail = list(range(171, 257))  # 86 elements
    
    def can_add(S, x):
        """Check if adding x to set S (where S is a set of ints) keeps validity."""
        if x in S:
            return True
        
        S_list = sorted(S)
        
        # Case 1: x is smallest (a=x), need b,c in S with x < b <= c and x | (b+c)
        for i, b in enumerate(S_list):
            if b <= x:
                continue
            for c in S_list[i:]:  # c >= b
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is middle (b=x), need a in S with a < x, c in S with c >= x, a | (x+c)
        for a in S_list:
            if a >= x:
                break
            for c in S_list:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is largest (c=x), need a,b in S with a < b <= x, a | (b+x)
        for i, a in enumerate(S_list):
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                if b > x:
                    break
                if (b + x) % a == 0:
                    return False
        
        return True
    
    def try_greedy(S_initial, order):
        """Try greedy addition in given order. S_initial is a list."""
        S = set(S_initial)
        for x in order:
            if x in S:
                continue
            if can_add(S, x):
                S.add(x)
        return S
    
    # Candidate orders for adding elements 1..170
    candidates_to_add = list(range(1, 171))
    
    best_set = set(base_tail)
    best_size = len(base_tail)
    
    # Order 1: descending (170, 169, ..., 1)
    order_desc = list(range(170, 0, -1))
    S1 = try_greedy(base_tail, order_desc)
    if len(S1) > best_size:
        best_set = S1
        best_size = len(S1)
    
    if time.time() > deadline:
        return sorted(best_set)
    
    # Order 2: ascending (1, 2, ..., 170)
    order_asc = list(range(1, 171))
    S2 = try_greedy(base_tail, order_asc)
    if len(S2) > best_size:
        best_set = S2
        best_size = len(S2)
    
    if time.time() > deadline:
        return sorted(best_set)
    
    # Order 3: priority - try elements that are less likely to conflict
    # Elements coprime to many others, or elements in ranges with fewer multiples
    # Heuristic: try numbers that are prime or have few divisors first
    import math
    
    def count_divisors_in_range(x, lo, hi):
        """Count how many numbers in [lo, hi] are divisible by x"""
        if x == 0:
            return 0
        return (hi // x) - ((lo - 1) // x)
    
    # Sort by "conflict potential" - fewer multiples means less likely to cause issues
    order_priority = sorted(candidates_to_add, key=lambda x: count_divisors_in_range(x, 171, 256))
    S3 = try_greedy(base_tail, order_priority)
    if len(S3) > best_size:
        best_set = S3
        best_size = len(S3)
    
    if time.time() > deadline:
        return sorted(best_set)
    
    # Order 4: mix - try from the middle outward
    mid = 85
    order_mid = []
    for d in range(86):
        if mid - d >= 1:
            order_mid.append(mid - d)
        if mid + d <= 170 and mid + d != mid - d:
            order_mid.append(mid + d)
    S4 = try_greedy(base_tail, order_mid)
    if len(S4) > best_size:
        best_set = S4
        best_size = len(S4)
    
    if time.time() > deadline:
        return sorted(best_set)
    
    # Try to extend the tail further: maybe [165, 256] or lower is valid?
    # Check if [170, 256] is valid
    def is_valid_set(S):
        S_list = sorted(S)
        for i, a in enumerate(S_list):
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                for k in range(j, len(S_list)):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    # Try to lower the tail start
    for tail_start in range(170, 140, -1):
        tail = list(range(tail_start, 257))
        if is_valid_set(tail):
            # Greedily add elements below tail_start
            order_below = list(range(tail_start - 1, 0, -1))
            S_new = try_greedy(tail, order_below)
            if len(S_new) > best_size:
                best_set = S_new
                best_size = len(S_new)
            break
    
    if time.time() > deadline:
        return sorted(best_set)
    
    # Final: try a few more random-ish orders using deterministic pseudo-random
    import random
    rng = random.Random(42)
    
    for trial in range(10):
        if time.time() > deadline:
            break
        order = candidates_to_add[:]
        rng.shuffle(order)
        S_t = try_greedy(base_tail, order)
        if len(S_t) > best_size:
            best_set = S_t
            best_size = len(S_t)
    
    return sorted(best_set)
