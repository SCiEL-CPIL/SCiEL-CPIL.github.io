# EVOLVE-BLOCK-START
from math import isqrt

def construct():
    n = 256
    # Precompute divisors for all possible products b*c where b,c in 2..256
    # Max product is 256*255 = 65280
    max_prod = n * (n - 1)
    divisors = [[] for _ in range(max_prod + 1)]
    for d in range(2, n + 1):
        for multiple in range(d, max_prod + 1, d):
            divisors[multiple].append(d)
    
    S = set(range(2, n + 1))
    
    while True:
        degrees = {x: 0 for x in S}
        has_conflict = False
        S_list = list(S)
        L = len(S_list)
        
        # Iterate over all pairs (b, c) in S
        for i in range(L):
            b = S_list[i]
            for j in range(i + 1, L):
                c = S_list[j]
                prod = b * c
                # Check divisors of prod that are in S and distinct from b, c
                for a in divisors[prod]:
                    if a in S and a != b and a != c:
                        has_conflict = True
                        # This triple (a, b, c) is a conflict
                        degrees[a] += 1
                        degrees[b] += 1
                        degrees[c] += 1
        
        if not has_conflict:
            break
        
        # Find element with max degree
        # Tie-break: Smallest value (heuristic: smaller numbers are more "divisive")
        max_deg = -1
        best_candidate = None
        for x in S:
            d = degrees[x]
            if d > max_deg:
                max_deg = d
                best_candidate = x
            elif d == max_deg:
                if best_candidate is None or x < best_candidate:
                    best_candidate = x
        
        S.remove(best_candidate)
        
    return list(S)
# EVOLVE-BLOCK-END
