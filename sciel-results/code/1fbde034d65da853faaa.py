def construct():
    n = 1008
    p = 1009
    
    # Precompute inverses
    inv = [0] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
        
    best_len = 0
    best_start = 1
    
    a = 1
    b = 0 # Empty window [1, 0]
    C = 0
    
    while a <= n:
        # Try to expand b
        while b < n:
            next_b = b + 1
            y = inv[next_b]
            
            # 1. Does next_b become bad?
            # next_b is bad if y in [a, next_b]
            delta1 = 0
            if a <= y <= next_b:
                delta1 = 1
                
            # 2. Does any existing x become bad?
            # x in [a, b] becomes bad if inv[x] == next_b
            # This is equivalent to x == y
            # So if y in [a, b], then x=y becomes bad.
            delta2 = 0
            if a <= y <= b:
                delta2 = 1
                
            delta_C = delta1 + delta2
            
            if C + delta_C == 0:
                b = next_b
                C += delta_C
            else:
                break
        
        if b >= a:
            length = b - a + 1
            if length > best_len:
                best_len = length
                best_start = a
        
        # Shrink a
        if b >= a:
            y = inv[a]
            
            # 1. a was bad?
            # a was bad if y in [a, b]
            dec1 = 0
            if a <= y <= b:
                dec1 = 1
                
            # 2. Some existing x becomes valid?
            # x in [a+1, b] becomes valid if inv[x] == a
            # This is equivalent to x == y
            # So if y in [a+1, b], then x=y becomes valid.
            dec2 = 0
            if a + 1 <= y <= b:
                dec2 = 1
                
            C -= (dec1 + dec2)
            a += 1
        else:
            a += 1
            if b < a - 1:
                b = a - 1
                C = 0

    return [best_start, best_len]
