import time

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    p = 1009
    n = p - 1  # 1008
    
    # Precompute inverses
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
        
    best_len = 0
    best_start = 1
    
    # O(N^2) search
    for s in range(1, n + 1):
        if time.time() > deadline:
            break
            
        seen = set()
        max_e = s - 1
        for e in range(s, n + 1):
            # Check self-inverse
            if e == 1 or e == 1008:
                break
            # Check if inverse is in the current interval [s, e-1]
            if inv[e] in seen:
                break
            
            seen.add(e)
            max_e = e
            
        length = max_e - s + 1
        if length > best_len:
            best_len = length
            best_start = s
            
    return [best_start, best_len]
