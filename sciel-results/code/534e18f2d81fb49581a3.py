import time
import random

def construct():
    start = time.time()
    deadline = start + 85.0
    N = 128
    rng = random.Random(42)

    def time_left():
        return deadline - time.time()

    def is_valid_add(inS, x):
        """
        Check if adding x to the set represented by inS creates a forbidden triple.
        The triple must be {a, b, x} with a, b, x distinct.
        Forbidden if a+b, a+x, b+x are all in S (after adding x).
        Note: a+b must be in S (original S, since x is new and x is not in a+b unless a+b=x, 
        but a,b,x distinct implies a+b != x generally? No, a+b can be x. 
        If a+b=x, then x is in S (we are adding it). So a+b is in S.
        So we just check if a+b, a+x, b+x are in the NEW set.
        Since x is being added, a+x and b+x are in the new set iff they are in old S or equal to x.
        a+x = x => a=0 (not in 1..N). So a+x is in new set iff inS[a+x].
        Same for b+x.
        a+b is in new set iff inS[a+b] or a+b==x.
        """
        for a in range(1, N + 1):
            if not inS[a] or a == x:
                continue
            for b in range(a + 1, N + 1):
                if not inS[b] or b == x:
                    continue
                
                # Check sums
                sum_ab = a + b
                sum_ax = a + x
                sum_bx = b + x
                
                # sum_ab is in new set if it was in old set OR if it is x
                if sum_ab != x and not inS[sum_ab]:
                    continue
                if sum_ab == x:
                    # x is being added, so it is in the set
                    pass
                else:
                    # inS[sum_ab] is True
                    pass
                
                # sum_ax is in new set if it was in old set OR if it is x
                # sum_ax = x => a=0, impossible. So must be in old set.
                if sum_ax > N or not inS[sum_ax]:
                    continue
                    
                # sum_bx is in new set if it was in old set OR if it is x
                if sum_bx > N or not inS[sum_bx]:
                    continue
                
                # If we reach here, all three sums are in the new set
                return False
        
        return True

    def full_valid(mem):
        s = set(mem)
        m = sorted(mem)
        n = len(m)
        for i in range(n):
            a = m[i]
            for j in range(i + 1, n):
                b = m[j]
                ab = a + b
                if ab not in s:
                    continue
                for k in range(j + 1, n):
                    c = m[k]
                    if (a + c) in s and (b + c) in s:
                        return False
        return True

    # Base set: All odds
    base_inS = bytearray(N + 1)
    base_mem = []
    for i in range(1, N + 1, 2):
        base_inS[i] = 1
        base_mem.append(i)
        
    best_mem = list(base_mem)
    best_size = len(base_mem)
    
    # Candidates: All evens
    evens = list(range(2, N + 1, 2))
    
    def try_greedy(order):
        nonlocal best_mem, best_size
        inS = bytearray(base_inS) # Copy
        mem = list(base_mem)
        
        for x in order:
            if time_left() < 2.0:
                break
            if inS[x]:
                continue
            if is_valid_add(inS, x):
                inS[x] = 1
                mem.append(x)
        
        if len(mem) > best_size:
            best_size = len(mem)
            best_mem = list(mem)
            
    # Try different orders
    # 1. Descending
    try_greedy(sorted(evens, reverse=True))
    
    # 2. Ascending
    try_greedy(sorted(evens))
    
    # 3. Random orders
    for _ in range(500):
        if time_left() < 5.0:
            break
        o = list(evens)
        rng.shuffle(o)
        try_greedy(o)
        
    # Local search on best found
    # If we have time, try to improve best_mem
    # Convert best_mem to inS
    best_inS = bytearray(N + 1)
    for x in best_mem:
        best_inS[x] = 1
        
    # Try removing one element and adding more
    # This is complex to implement robustly in the remaining time.
    # Given the greedy might be close to optimal for this structure, 
    # and 66-70 is a good target, we return best.
    
    if not full_valid(best_mem):
        # Fallback to odds
        best_mem = [i for i in range(1, N + 1, 2)]
        
    return [int(x) for x in best_mem]
