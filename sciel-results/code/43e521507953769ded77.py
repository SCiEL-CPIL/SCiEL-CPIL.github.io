import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85.0

    # Bitset representation of S
    # bit i is 1 if i in S
    # Initial S: {171, ..., 256}
    S_bits = 0
    for i in range(171, n + 1):
        S_bits |= (1 << i)
    
    # Helper to check if x can be added to S_bits
    def can_add_bits(x, S_bits):
        # Case 1: x | (b+c) for b,c in S, x < b <= c
        # b+c = k*x
        # b,c > x
        # k from 3 to 2*n//x
        for k in range(3, (2 * n) // x + 1):
            target = k * x
            # Check if there exist b, c in S such that b+c=target, b<=c, b>x
            # b can be from max(x+1, target-n) to min(target-(x+1), n)
            # Actually, just iterate b?
            # With bitset, we can check pairs faster?
            # Not directly.
            # Iterate b in range?
            # Range size is up to n.
            # But we only care if b is in S.
            # We can iterate b from 1 to n, check bit, then check target-b bit.
            # But we need b > x.
            
            # Optimization:
            # If we have a list of elements in S, we can iterate that.
            # Let's maintain a list `S_list` alongside `S_bits`.
            pass
        # This bitset approach is tricky for pair sums without iterating.
        # Let's stick to the list-based `can_add` but optimize it.
        
        # The previous list-based `can_add` was:
        # Case 1: O(|S| * n/x)
        # Case 2: O(|S| * n/a)
        # Case 3: O(|S| * |S|) -> O(|S|^2)
        
        # For x=171, this is fast.
        # For x=1, Case 1 is O(86 * 256) = 22k.
        # Case 2/3 with a=1:
        # a=1 divides everything.
        # Case 2: 1 | (1+c). Always true.
        # So if 1 in S, can_add(1, S) is always False if S has any c >= 1?
        # Wait, if a=1, then 1 | (x+c) is always true.
        # So if 1 is in S, we can NEVER add any x?
        # No, if 1 is in S, then for any x, we check a=1.
        # Case 2: a=1 < x. c in S, c >= x.
        # 1 | (x+c) is always true.
        # So if 1 in S, we cannot add any x such that there exists c in S with c >= x.
        # Since S is non-empty and max(S) >= 1, if x <= max(S), there is such c.
        # So 1 is a "poison" element. It prevents adding most other elements.
        # Therefore, optimal sets likely do NOT contain 1.
        
        # This suggests we can prune small numbers early.
        
        return True # Placeholder

    # Let's use the list-based version but keep S as a list for iteration speed.
    # S_list sorted?
    
    S_list = list(range(171, n + 1))
    S_set = set(S_list)
    
    def can_add_list(x, S_list, S_set):
        # Case 1: x | (b+c)
        # b, c in S_list
        # b > x
        # We can pre-filter S_list to get elements > x?
        # Or just check.
        
        # Optimization: If x is small, many conflicts.
        # If x is large, few conflicts.
        
        # Case 1
        # b+c = k*x
        # b,c in S
        # Iterate b in S_list
        for b in S_list:
            if b <= x:
                continue
            # c must be in S, c >= b
            # c = k*x - b
            # k must be such that c is in [b, n]
            # b <= k*x - b <= n
            # 2b <= k*x <= n+b
            # k_min = ceil(2b/x)
            # k_max = floor((n+b)/x)
            
            k_min = (2 * b + x - 1) // x
            k_max = (n + b) // x
            
            for k in range(k_min, k_max + 1):
                c = k * x - b
                if c < b:
                    continue
                if c in S_set:
                    return False
        
        # Case 2: a | (x+c)
        # a < x, c >= x
        for a in S_list:
            if a >= x:
                continue
            # c in S, c >= x, (x+c) % a == 0
            rem = (-x) % a
            # c = rem + k*a
            # c >= x
            if rem < x:
                k_needed = (x - rem + a - 1) // a
                c_start = rem + k_needed * a
            else:
                c_start = rem
            
            c = c_start
            while c <= n:
                if c in S_set:
                    return False
                c += a

        # Case 3: a | (b+x)
        # a < b <= x
        for a in S_list:
            if a >= x:
                continue
            # b in S, a < b <= x, (b+x) % a == 0
            rem = (-x) % a
            # b = rem + k*a
            # a < b <= x
            # b > a => rem + k*a > a
            # b <= x => rem + k*a <= x
            
            # Find k range
            # k*a > a - rem
            # k*a <= x - rem
            
            # Just iterate b in S_list?
            # S_list is small.
            for b in S_list:
                if b <= a or b > x:
                    continue
                if (b + x) % a == 0:
                    return False

        return True

    # Main Loop
    max_iters = 5000
    iters = 0
    
    while time.time() < deadline and iters < max_iters:
        iters += 1
        improved = False
        
        # Try Add
        candidates = [x for x in range(1, n + 1) if x not in S_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add_list(x, S_list, S_set):
                S_list.append(x)
                S_set.add(x)
                improved = True
                break
        
        if improved:
            continue
            
        # Try Swap
        # Limit swaps to avoid long runs
        swap_attempts = 0
        max_swaps_per_iter = 50
        
        random.shuffle(candidates)
        for x in candidates:
            if swap_attempts >= max_swaps_per_iter:
                break
            swap_attempts += 1
            
            S_list_copy = S_list[:]
            S_set_copy = S_set.copy()
            
            # Try removing one y
            random.shuffle(S_list_copy)
            for y in S_list_copy:
                S_list_try = [z for z in S_list_copy if z != y]
                S_set_try = S_set_copy.copy()
                S_set_try.remove(y)
                
                if can_add_list(x, S_list_try, S_set_try):
                    # Perform swap
                    S_list = S_list_try + [x]
                    S_set = S_set_try
                    S_set.add(x)
                    improved = True
                    break
            
            if improved:
                break

    return S_list
