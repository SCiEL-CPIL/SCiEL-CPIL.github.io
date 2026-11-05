import time
import itertools

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def get_paley_orientation(p):
    squares = {x * x % p for x in range(1, p)}
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            # i -> j iff (j - i) % p is a square
            if (j - i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return orientation

def is_valid_paley(p):
    # Check if Paley tournament on p vertices satisfies the condition
    # Condition: For every 4 distinct vertices a,b,c,d, there exists v such that v->a, v->b, v->c, v->d
    
    squares = {x * x % p for x in range(1, p)}
    
    # Pred[u] is a bitmask of vertices v such that v -> u
    # v -> u iff (u - v) % p in squares
    # v = u - q (mod p) for q in squares
    # So Pred[u] = { (u - q) % p for q in squares }
    
    pred_masks = [0] * p
    for u in range(p):
        mask = 0
        for q in squares:
            v = (u - q) % p
            mask |= (1 << v)
        pred_masks[u] = mask
        
    # Check all combinations of 4
    for a, b, c, d in itertools.combinations(range(p), 4):
        if (pred_masks[a] & pred_masks[b] & pred_masks[c] & pred_masks[d]) == 0:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = 110.0 # 10 second margin
    
    # Check Paley primes p ≡ 3 (mod 4) in increasing order
    candidates = [p for p in range(31, 84) if is_prime(p) and p % 4 == 3]
    
    best_n = 83
    best_orientation = get_paley_orientation(83)
    
    for p in candidates:
        if time.time() > deadline:
            break
            
        if is_valid_paley(p):
            best_n = p
            best_orientation = get_paley_orientation(p)
            break # Found the smallest valid Paley prime
            
    return {'n': best_n, 'orientation': best_orientation}
