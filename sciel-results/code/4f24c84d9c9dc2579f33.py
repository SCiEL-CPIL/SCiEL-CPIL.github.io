"""Find the longest interval in [1,1008] with no modular inverse pair (including diagonal)."""

def construct():
    p = 1009
    # Precompute modular inverses
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
    
    best_len = 0
    best_start = 1
    
    for s in range(1, p):
        # Optimization: if max possible length from s is <= best_len, break
        if p - 1 - s + 1 <= best_len:
            break
        
        for e in range(s, p):
            # Check diagonal: e^2 == 1 mod p (i.e., e == 1 or e == p-1)
            if e == 1 or e == p - 1:
                break
            
            # Check off-diagonal: inv(e) in [s, e-1]
            if s <= inv[e] <= e - 1:
                break
            
            current_len = e - s + 1
            if current_len > best_len:
                best_len = current_len
                best_start = s
    
    return [best_start, best_len]
