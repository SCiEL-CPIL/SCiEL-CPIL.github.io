import time

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85
    
    p = 1009
    n = p - 1  # 1008
    
    # Compute modular inverses
    # inv[x] is x^(-1) mod p
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
        
    best_start = 2
    best_len = 0
    
    # s cannot be 1 or 1008 because they are self-inverse
    # e cannot be 1008 because it is self-inverse
    # So s in [2, 1007], e in [s, 1007]
    
    for s in range(2, 1008):
        # Prune if max possible length from this s is less than current best
        if (1007 - s + 1) <= best_len:
            break
            
        in_interval = [False] * (p + 1)
        valid_end = s - 1
        
        for e in range(s, 1008):
            # Check if e is self-inverse (should be handled by range, but good to be safe)
            if e * e % p == 1:
                break
                
            # Check if inv[e] is already in the interval [s, e-1]
            if in_interval[inv[e]]:
                break
                
            # e is valid to add
            in_interval[e] = True
            valid_end = e
            
            # Update best if current length is larger
            current_len = e - s + 1
            if current_len > best_len:
                best_len = current_len
                best_start = s
                
            # Deadline check (unlikely to trigger given O(N^2) with N=1008)
            if time.time() > DEADLINE:
                return [best_start, best_len]
                
    return [best_start, best_len]
