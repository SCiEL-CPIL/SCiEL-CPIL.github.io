import sys
from math import isqrt

# Increase recursion limit if needed, though not strictly necessary here
sys.setrecursionlimit(2000)

def construct():
    N = 10**6
    # Sieve to compute sigma(k) for all k <= N
    sigma = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
            
    solutions = {}
    max_k = 0
    best_M = 0
    best_count = 0
    
    for k in range(1, N + 1):
        M = k * sigma[k]
        if M <= 10**18:
            if M not in solutions:
                solutions[M] = []
            solutions[M].append(k)
            if len(solutions[M]) > best_count:
                best_count = len(solutions[M])
                best_M = M

    # The problem asks for distinct k_i. The list construction ensures distinctness as k iterates uniquely.
    # We return the M with the most solutions.
    # Note: It's possible that for the best_M, there are k > N that also map to it.
    # But we can't easily find them without a different search strategy.
    # Given the density, the best M is likely found within small k.
    
    return {'M': best_M, 'k': solutions[best_M]}
