import time

def construct():
    p = 1009
    N = 1008
    
    # Precompute modular inverses
    # inv[x] = x^{-1} mod p
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
        
    # Precompute self-inverse check
    # x is self-inverse if x^2 % p == 1
    is_self_inv = [False] * (p + 1)
    for x in range(1, p):
        if (x * x) % p == 1:
            is_self_inv[x] = True
            
    best_length = 0
    best_start = 1
    
    # Iterate over all possible start positions
    for start in range(1, N + 1):
        # If the start element itself is self-inverse, no valid interval can start here
        # (length would be 0, which is not "nonempty" in the sense of containing start)
        # Actually, the problem says "nonempty interval". If start is self-inverse, 
        # any interval containing it is invalid. So max length starting at 'start' is 0.
        if is_self_inv[start]:
            continue
            
        current_set = set()
        current_length = 0
        
        for end in range(start, N + 1):
            x = end
            # Check if x can be added to the interval [start, end]
            # 1. x must not be self-inverse
            if is_self_inv[x]:
                break
            # 2. x's inverse must not be in the current set
            if inv[x] in current_set:
                break
            
            # If we reach here, x is valid to add
            current_set.add(x)
            current_length = end - start + 1
            
            if current_length > best_length:
                best_length = current_length
                best_start = start
                
    return [best_start, best_length]
