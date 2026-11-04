import math

def calc_score(U):
    if not U:
        return 0
    U = list(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    num = len(diffs) / len(sums)
    den = math.log(2 * max_u + 1)
    if den == 0:
        return 0
    return 1 + math.log(num) / den

def generate_set(D, B, n):
    U = [0]
    for mask in range(1, 1 << n):
        val = 0
        for j in range(n):
            if mask & (1 << j):
                val += D[j] * (B ** j) # This is wrong, D is a set of digits, not indexed by j
        # Correct generation:
    # All combinations of digits from D, length n
    from itertools import product
    U = []
    for digits in product(D, repeat=n):
        val = 0
        for j in range(n):
            val += digits[j] * (B ** j)
        U.append(val)
    return list(set(U))

def construct():
    best = [0, 1, 3]
    best_score = calc_score(best)
    
    # Search over digit sets D and base B
    # D must contain 0
    # Try D = {0, 1, k}
    
    for k in range(2, 200):
        D = [0, 1, k]
        # Need B > max(D+D) to avoid carries
        # max(D+D) = 2k
        B = 2 * k + 1
        
        # Try different lengths n
        for n in range(2, 6):
            U = generate_set(D, B, n)
            if len(U) > 1000: # Safety check
                break
            s = calc_score(U)
            if s > best_score:
                best_score = s
                best = U
                
    # Also try D = {0, k}
    for k in range(2, 200):
        D = [0, k]
        B = 2 * k + 1
        for n in range(2, 8):
            U = generate_set(D, B, n)
            if len(U) > 1000:
                break
            s = calc_score(U)
            if s > best_score:
                best_score = s
                best = U

    return best
