import time
import random
from math import gcd
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    N = 511
    # Weights: 1 / (512 - a)
    weights = [0.0] * (N + 1)
    for a in range(1, N + 1):
        weights[a] = 1.0 / (512 - a)
    
    # Precompute prime factors for each number
    # Using a list of frozensets of prime factors
    factors = [frozenset() for _ in range(N + 1)]
    # Sieve of Eratosthenes style to find smallest prime factor, then factorize
    spf = list(range(N + 1))
    for i in range(2, int(N**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    for i in range(2, N + 1):
        x = i
        pf = set()
        while x > 1:
            p = spf[x]
            pf.add(p)
            while x % p == 0:
                x //= p
        factors[i] = frozenset(pf)
    factors[1] = frozenset() # 1 has no prime factors, coprime to everything

    def is_coprime(a, b):
        # Check if they share any prime factor
        return not (factors[a] & factors[b])

    def is_valid_set(S):
        # Check if all pairs in S are coprime
        # Since N is small, O(k^2) is fine for k ~ 50-100
        k = len(S)
        for i in range(k):
            for j in range(i+1, k):
                if S[i] != S[j] and not is_coprime(S[i], S[j]):
                    return False
        return True

    def get_score(S):
        return sum(weights[a] for a in S)

    def greedy_construct():
        """Greedy construction of a valid set."""
        S = []
        # Sort candidates by weight descending
        candidates = list(range(1, N + 1))
        candidates.sort(key=lambda x: weights[x], reverse=True)
        
        for a in candidates:
            # Check if a is coprime to all in S
            valid = True
            for b in S:
                if not is_coprime(a, b):
                    valid = False
                    break
            if valid:
                S.append(a)
        return S

    def local_search(S):
        """Improve S using single-swap local search."""
        S = set(S)
        improved = True
        while improved:
            improved = False
            # Try swapping each a in S with each b not in S
            # To speed up, we can just check b > a in weight
            for a in list(S):
                # Find best b not in S that is coprime to S \ {a} and has weight > weight[a]
                best_b = None
                best_w = weights[a]
                for b in range(1, N + 1):
                    if b in S:
                        continue
                    if weights[b] <= best_w:
                        continue
                    # Check if b is coprime to all in S \ {a}
                    valid = True
                    for c in S:
                        if c == a:
                            continue
                        if not is_coprime(b, c):
                            valid = False
                            break
                    if valid:
                        best_b = b
                        best_w = weights[b]
                
                if best_b is not None:
                    S.remove(a)
                    S.add(best_b)
                    improved = True
        return S

    def lns_step(S, k_remove):
        """Large Neighborhood Search step: remove k elements, refill greedily."""
        S = set(S)
        if len(S) < k_remove:
            return S
            
        # Remove k random elements
        to_remove = random.sample(list(S), k_remove)
        for a in to_remove:
            S.remove(a)
            
        # Refill greedily from remaining candidates
        # We want to add elements that are valid and have high weight
        candidates = [a for a in range(1, N + 1) if a not in S]
        candidates.sort(key=lambda x: weights[x], reverse=True)
        
        for a in candidates:
            # Check if a is coprime to all in S
            valid = True
            for b in S:
                if not is_coprime(a, b):
                    valid = False
                    break
            if valid:
                S.add(a)
                
        # Polish with local search
        S = local_search(S)
        return S

    # Main Search Loop
    best_S = greedy_construct()
    best_score = get_score(best_S)
    
    # Run LNS iterations
    iteration = 0
    while time.time() < deadline:
        # Vary k_remove
        k_remove = random.randint(1, 5)
        new_S = lns_step(best_S, k_remove)
        new_score = get_score(new_S)
        
        if new_score > best_score:
            best_score = new_score
            best_S = new_S
            # Optional: If we found a new best, we can also try a "restart" from scratch
            # but LNS is usually sufficient.
            
        # Occasionally try a fresh greedy start to escape deep local minima
        if iteration % 100 == 0:
            fresh_S = greedy_construct()
            fresh_S = local_search(fresh_S)
            fresh_score = get_score(fresh_S)
            if fresh_score > best_score:
                best_score = fresh_score
                best_S = fresh_S
                
        iteration += 1

    return list(best_S)

# EVOLVE-BLOCK-END
