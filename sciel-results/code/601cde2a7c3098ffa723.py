import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 256
    S = set()
    
    # Try greedy: add elements from largest to smallest
    # Larger elements are "safer" because fewer multiples fit in range
    for x in range(n, 0, -1):
        if time.time() > deadline:
            break
        
        if _can_add(x, S, n):
            S.add(x)
    
    if not S:
        return [n]
    
    return sorted(S)

def _can_add(x, S, n):
    """Check if x can be added to S without violating the property."""
    # After adding x, for all a<b<=c in S∪{x}, a must not divide b+c
    # x can play the role of a, b, or c
    
    S_list = list(S)
    
    # Case 1: x is the smallest (a=x), need b>c in S with x < b <= c and x | (b+c)
    for b in S_list:
        if b <= x:
            continue
        for c in S_list:
            if c < b:
                continue
            if (b + c) % x == 0:
                return False
    
    # Case 2: x is b or c, some a in S with a < x
    for a in S_list:
        if a >= x:
            continue
        # x as b: need c in S with x <= c and a | (x+c)
        for c in S_list:
            if c < x:
                continue
            if (x + c) % a == 0:
                return False
        # x as c: need b in S with a < b <= x and a | (b+x)
        for b in S_list:
            if b <= a or b > x:
                continue
            if (b + x) % a == 0:
                return False
    
    return True
