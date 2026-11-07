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
    # Use random restarts to explore different greedy paths
    # The deterministic greedy (sorted by divisors) is just one specific order.
    # Random orders can find different local optima.
    
    # Run a few deterministic heuristics first
    # Heuristic 1: Primes first (sorted by value)
    primes = [x for x in candidates if all(x % i != 0 for i in range(2, int(x**0.5) + 1))]
    composites = [x for x in candidates if x not in primes]
    order1 = primes + composites
    S1 = greedy(order1)
    best_S = S1
    
    # Heuristic 2: Sorted by value
    order2 = sorted(candidates)
    S2 = greedy(order2)
    if len(S2) > len(best_S):
        best_S = S2
        
    # Heuristic 3: Random restarts
    # Limit iterations to stay within time limits
    # 255 elements, O(N^2) check per element -> O(N^3) per run ~ 16M ops.
    # Python can do ~10M-100M simple ops/sec.
    # 100 restarts * 16M = 1.6B ops. Too slow.
    # We need to be faster or do fewer restarts.
    
    # Optimized check: 
    # If S contains all primes, we proved we can't add composites.
    # So the best is likely close to 54.
    # Let's do 50 random restarts.
    
    for _ in range(50):
        random.shuffle(candidates)
        S_rand = greedy(candidates)
        if len(S_rand) > len(best_S):
            best_S = S_rand
            
    return best_S
# EVOLVE-BLOCK-END
