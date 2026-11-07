"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random

def construct():
    N = 256
    candidates = list(range(2, N + 1))
    
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

    best_S = []
    
    # Heuristic 1: Primes first (sorted by value)
    primes = [x for x in candidates if all(x % i != 0 for i in range(2, int(x**0.5) + 1))]
    composites = [x for x in candidates if x not in primes]
    order1 = primes + composites
    S1 = greedy(order1)
    best_S = S1
    
    # Heuristic 2: Sorted by value descending
    order2 = sorted(candidates, reverse=True)
    S2 = greedy(order2)
    if len(S2) > len(best_S):
        best_S = S2

    # Heuristic 3: Sorted by number of divisors ascending
    def get_div_count(x):
        cnt = 0
        for i in range(1, int(x**0.5) + 1):
            if x % i == 0:
                cnt += 1
                if i != x // i:
                    cnt += 1
        return cnt

    order3 = sorted(candidates, key=get_div_count)
    S3 = greedy(order3)
    if len(S3) > len(best_S):
        best_S = S3

    # Heuristic 4: Random restarts
    for _ in range(30):
        random.shuffle(candidates)
        S_rand = greedy(candidates)
        if len(S_rand) > len(best_S):
            best_S = S_rand
            
    return best_S
# EVOLVE-BLOCK-END
