# EVOLVE-BLOCK-START
import math
from math import gcd

def construct():
    n = 512
    # Precompute weights
    weights = [0.0] * n
    for a in range(1, n):
        weights[a] = 1.0 / (n - a)
        
    # Precompute conflicts: conflicts[a] is a set of b such that gcd(a,b) > 1
    conflicts = [set() for _ in range(n)]
    for a in range(1, n):
        for b in range(a + 1, n):
            if gcd(a, b) > 1:
                conflicts[a].add(b)
                conflicts[b].add(a)
                
    def is_prime(x):
        if x < 2:
            return False
        if x == 2:
            return True
        if x % 2 == 0:
            return False
        for i in range(3, int(math.sqrt(x)) + 1, 2):
            if x % i == 0:
                return False
        return True

    def local_search(initial_set):
        S = set(initial_set)
        improved = True
        while improved:
            improved = False
            best_gain = 1e-9
            best_a = -1
            best_conflicts = set()
            
            # Evaluate all possible additions/swaps
            # We iterate over candidates not in S
            for a in range(1, n):
                if a in S:
                    continue
                
                # Find conflicts with current selected set
                c_set = S.intersection(conflicts[a])
                
                if not c_set:
                    gain = weights[a]
                else:
                    cost = sum(weights[b] for b in c_set)
                    gain = weights[a] - cost
                
                if gain > best_gain:
                    best_gain = gain
                    best_a = a
                    best_conflicts = c_set
            
            if best_a != -1:
                # Apply the best move
                S.difference_update(best_conflicts)
                S.add(best_a)
                improved = True
                
        return S

    # Strategy 1: Start with all primes
    seed_primes = {p for p in range(2, n) if is_prime(p)}
    seed_primes.add(1)
    result1 = local_search(seed_primes)
    
    # Strategy 2: Start with greedy by weight (descending a)
    S_greedy = set()
    for a in range(n - 1, 0, -1):
        c_set = S_greedy.intersection(conflicts[a])
        if not c_set:
            S_greedy.add(a)
    result2 = local_search(S_greedy)
    
    # Strategy 3: Randomized Greedy (a few times)
    import random
    best_result = result1
    best_score = sum(weights[a] for a in result1)
    
    score2 = sum(weights[a] for a in result2)
    if score2 > best_score:
        best_result = result2
        best_score = score2
        
    # Try a few random restarts
    for _ in range(10):
        S_rand = set()
        candidates = list(range(1, n))
        random.shuffle(candidates)
        for a in candidates:
            c_set = S_rand.intersection(conflicts[a])
            if not c_set:
                S_rand.add(a)
        result_rand = local_search(S_rand)
        score_rand = sum(weights[a] for a in result_rand)
        if score_rand > best_score:
            best_result = result_rand
            best_score = score_rand

    return list(best_result)
# EVOLVE-BLOCK-END
