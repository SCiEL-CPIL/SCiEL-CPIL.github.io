import bisect
from collections import defaultdict

def construct():
    # Search space parameters
    # We want to find a, b <= 10^7
    # We search b up to N_B, a up to 1.5 * N_B
    # Let's set N_B to 2,000,000 to balance time and quality
    N_B = 2_000_000
    N_A = int(1.5 * N_B) + 100
    
    # Sieve for sigma(n)
    # Using a list of integers
    sigma = [0] * (N_A + 1)
    for i in range(1, N_A + 1):
        for j in range(i, N_A + 1, i):
            sigma[j] += i
            
    # Group numbers by their sigma value
    groups = defaultdict(list)
    for n in range(1, N_A + 1):
        groups[sigma[n]].append(n)
        
    best_a = 31
    best_b = 21
    best_err = abs(31/21 - 1.5)
    
    # Iterate through possible b values
    for b in range(1, N_B + 1):
        S = sigma[b]
        L = groups[S]
        if not L:
            continue
            
        target = 1.5 * b
        # Find the closest a in L to target
        idx = bisect.bisect_left(L, target)
        
        candidates = []
        if idx < len(L):
            candidates.append(L[idx])
        if idx > 0:
            candidates.append(L[idx-1])
            
        for a in candidates:
            if a == b:
                continue
            # Avoid exact 1.5 ratio
            if 2 * a == 3 * b:
                continue
            err = abs(a / b - 1.5)
            if err < best_err:
                best_err = err
                best_a = a
                best_b = b
                
    return [best_a, best_b]
