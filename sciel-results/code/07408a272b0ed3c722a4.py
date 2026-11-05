# EVOLVE-BLOCK-START
import time
import random

N = 192
K = 3
MAX_CARD = 48

def _check_add(c, S):
    """Return True if adding c to set S (list) keeps validity.
    Check: (1) c does not divide sum of any 3 distinct elements of S.
           (2) for each a in S, no triple in (S\{a}) U {c} sums to multiple of a.
    """
    n = len(S)
    # (1) c divides sum of 3 distinct from S?
    for i in range(n):
        si = S[i]
        for j in range(i+1, n):
            s_ij = si + S[j]
            for k in range(j+1, n):
                if (s_ij + S[k]) % c == 0:
                    return False
    # (2) for each a in S, new triples containing c
    #     We need to check if (c + x + y) % a == 0 for any x, y in S, x!=a, y!=a, x!=y
    for a in S:
        # Optimization: If a is large, it's harder to divide small sums?
        # Just iterate
        for i in range(n):
            xi = S[i]
            if xi == a:
                continue
            for j in range(n):
                if j == i or S[j] == a:
                    continue
                # xi and S[j] are distinct because S has distinct elements and i!=j
                if (c + xi + S[j]) % a == 0:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin: return well before 60s kill
    rng = random.Random(42)
    
    best_set = []
    best_size = 0
    
    # Strategy: Multiple randomized greedy passes
    # Each pass tries to build a set by adding elements in a random order.
    # This is much faster than local search with expensive validity checks.
    
    for pass_num in range(100):
        if time.time() > deadline:
            break
            
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        
        S = []
        for c in candidates:
            if _check_add(c, S):
                S.append(c)
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = list(S)
            
        # Early termination if we hit the max possible cardinality
        if best_size >= MAX_CARD:
            break
            
    return best_set
# EVOLVE-BLOCK-END
