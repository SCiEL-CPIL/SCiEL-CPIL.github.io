import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    n = 256
    base_start = n // 2 + 1  # 129
    S = set(range(base_start, n + 1))
    
    # Forbidden sums F: {x+y | x,y in S, x <= y}
    # Note: x=y is allowed in the sum generation for the constraint check
    # because the condition is a < b <= c, so b can equal c.
    # So we need to check if a divides 2b for any b in S.
    
    F = set()
    S_list = sorted(S)
    for i in range(len(S_list)):
        b = S_list[i]
        F.add(2 * b)
        for j in range(i + 1, len(S_list)):
            c = S_list[j]
            F.add(b + c)
            
    # Greedy phase
    for a in range(base_start - 1, 0, -1):
        if time.time() > deadline:
            break
        
        # Check if a divides any sum in F
        can_add = True
        max_sum = 2 * n
        # Multiples of a: 2a, 3a, ...
        # We only care if they are in F.
        # Max element in F is 2*256 = 512.
        for k in range(2, max_sum // a + 1):
            if k * a in F:
                can_add = False
                break
                
        if can_add:
            S.add(a)
            # Update F with new sums involving a
            # New sums: 2a and a+b for b in S_old
            F.add(2 * a)
            for b in S:
                if b != a:
                    F.add(a + b)

    # Randomized Add Phase
    remaining = [a for a in range(1, base_start) if a not in S]
    random.shuffle(remaining)
    
    for a in remaining:
        if time.time() > deadline:
            break
        can_add = True
        max_sum = 2 * n
        for k in range(2, max_sum // a + 1):
            if k * a in F:
                can_add = False
                break
        if can_add:
            S.add(a)
            F.add(2 * a)
            for b in S:
                if b != a:
                    F.add(a + b)

    return list(S)
