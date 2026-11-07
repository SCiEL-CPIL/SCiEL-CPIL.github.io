"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

def construct():
    N = 256
    candidates = list(range(2, N + 1))
    
    def is_valid_set(S):
        # Check if the set S is valid
        # For every distinct a, b, c in S, a does not divide b*c
        n = len(S)
        for i in range(n):
            a = S[i]
            for j in range(n):
                if i == j: continue
                b = S[j]
                for k in range(n):
                    if k == i or k == j: continue
                    c = S[k]
                    if (b * c) % a == 0:
                        return False
        return True

    def is_valid_addition(S, a):
        # Check if adding a to S violates any triple constraint
        # S is a list of ints
        for i in range(len(S)):
            b = S[i]
            for j in range(len(S)):
                if i == j:
                    continue
                c = S[j]
                # Check 1: a | b*c
                if (b * c) % a == 0:
                    return False
                # Check 2: b | a*c
                if (a * c) % b == 0:
                    return False
                # Check 3: c | a*b
                if (a * b) % c == 0:
                    return False
        return True

    def greedy(order):
        S = []
        for a in order:
            if is_valid_addition(S, a):
                S.append(a)
        return S

    # Heuristic 1: Upper half set {N/2 + 1, ..., N}
    # This is a known large valid set for many such problems.
    S_upper = list(range(N // 2 + 1, N + 1))
    if is_valid_set(S_upper):
        return S_upper

    # Fallback to greedy if upper half is not valid (unlikely)
    best_S = []
    
    # Heuristic 2: Primes first
    primes = [x for x in candidates if all(x % i != 0 for i in range(2, int(x**0.5) + 1))]
    composites = [x for x in candidates if x not in primes]
    order1 = primes + composites
    S1 = greedy(order1)
    best_S = S1
    
    # Heuristic 3: Sorted by value
    order2 = sorted(candidates)
    S2 = greedy(order2)
    if len(S2) > len(best_S):
        best_S = S2
        
    # Heuristic 4: Random restarts
    for _ in range(20):
        random.shuffle(candidates)
        S_rand = greedy(candidates)
        if len(S_rand) > len(best_S):
            best_S = S_rand
            
    return best_S
# EVOLVE-BLOCK-END
