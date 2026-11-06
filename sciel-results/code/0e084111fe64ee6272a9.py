import time
from math import gcd

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 150s limit
    
    N = 512
    MAX_A = N - 1  # 511
    
    # Precompute prime factor masks for each number
    # We'll use a list of sets of prime factors
    prime_factors = [set() for _ in range(N)]
    for a in range(2, N):
        x = a
        p = 2
        while p * p <= x:
            while x % p == 0:
                prime_factors[a].add(p)
                x //= p
            p += 1
        if x > 1:
            prime_factors[a].add(x)
    # 1 has no prime factors
    
    # Precompute coprime check function
    def is_coprime(a, b):
        return not (prime_factors[a] & prime_factors[b])
    
    # Precompute coprime_sets for quick lookup
    coprime_sets = [set() for _ in range(N)]
    for a in range(1, N):
        for b in range(1, N):
            if gcd(a, b) == 1:
                coprime_sets[a].add(b)
    
    # Score function
    def score(S):
        return sum(1.0 / (N - a) for a in S)
    
    # Greedy initialization: pick from highest value to lowest
    S = set()
    for a in range(MAX_A, 0, -1):
        if S.issubset(coprime_sets[a]):
            S.add(a)
    
    current_score = score(S)
    best_S = set(S)
    best_score = current_score
    
    # Local search with hill climbing
    improved = True
    while improved and (time.time() - start_time) < time_limit - 5.0:
        improved = False
        
        # Try swaps: replace a_out with a_in where a_in gives better score
        for a_out in list(S):
            best_a_in = None
            best_gain = 0
            S_minus = S - {a_out}
            # Check candidates from 511 down
            for a_in in range(MAX_A, 0, -1):
                if a_in in S:
                    continue
                if S_minus.issubset(coprime_sets[a_in]):
                    gain = 1.0 / (N - a_in) - 1.0 / (N - a_out)
                    if gain > best_gain:
                        best_gain = gain
                        best_a_in = a_in
            
            if best_a_in is not None:
                S.remove(a_out)
                S.add(best_a_in)
                current_score += best_gain
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_S = set(S)
                break
        
        # Try adds
        if not improved:
            best_a_in = None
            best_gain = 0
            for a_in in range(MAX_A, 0, -1):
                if a_in in S:
                    continue
                if S.issubset(coprime_sets[a_in]):
                    gain = 1.0 / (N - a_in)
                    if gain > best_gain:
                        best_gain = gain
                        best_a_in = a_in
            if best_a_in is not None:
                S.add(best_a_in)
                current_score += best_gain
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_S = set(S)
        
        # Try removals (if adding a low-value element blocks high-value ones)
        if not improved:
            for a_out in list(S):
                # Check if removing a_out allows adding something better
                S_minus = S - {a_out}
                best_a_in = None
                best_gain = 0
                for a_in in range(MAX_A, 0, -1):
                    if a_in in S:
                        continue
                    if S_minus.issubset(coprime_sets[a_in]):
                        gain = 1.0 / (N - a_in) - 1.0 / (N - a_out)
                        if gain > best_gain:
                            best_gain = gain
                            best_a_in = a_in
                if best_a_in is not None:
                    S.remove(a_out)
                    S.add(best_a_in)
                    current_score += best_gain
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_S = set(S)
                    break
    
    # Final polish: exhaustive local search on best_S
    S = set(best_S)
    current_score = best_score
    improved = True
    while improved and (time.time() - start_time) < time_limit - 1.0:
        improved = False
        for a_out in list(S):
            best_a_in = None
            best_gain = 0
            S_minus = S - {a_out}
            for a_in in range(MAX_A, 0, -1):
                if a_in in S:
                    continue
                if S_minus.issubset(coprime_sets[a_in]):
                    gain = 1.0 / (N - a_in) - 1.0 / (N - a_out)
                    if gain > best_gain:
                        best_gain = gain
                        best_a_in = a_in
            if best_a_in is not None:
                S.remove(a_out)
                S.add(best_a_in)
                current_score += best_gain
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_S = set(S)
                break
        if not improved:
            best_a_in = None
            best_gain = 0
            for a_in in range(MAX_A, 0, -1):
                if a_in in S:
                    continue
                if S.issubset(coprime_sets[a_in]):
                    gain = 1.0 / (N - a_in)
                    if gain > best_gain:
                        best_gain = gain
                        best_a_in = a_in
            if best_a_in is not None:
                S.add(best_a_in)
                current_score += best_gain
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_S = set(S)
    
    return list(best_S)
