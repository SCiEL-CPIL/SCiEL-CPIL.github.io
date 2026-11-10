import math
import random
import time
from math import gcd

def construct():
    N = 512
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety
    
    # Precompute prime factors for fast coprime checks
    def _factorize(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors
    
    _PRIME_FACTORS = {}
    for _i in range(1, N):
        _PRIME_FACTORS[_i] = _factorize(_i)
    
    def _is_coprime_to_all_fast(a, chosen_set):
        """Check if a is coprime to all elements in chosen_set using precomputed factors."""
        fa = _PRIME_FACTORS[a]
        for b in chosen_set:
            if fa & _PRIME_FACTORS[b]:
                return False
        return True
    
    def _score(chosen):
        return sum(1.0 / (N - a) for a in chosen)
    
    # 1. Greedy Initialization
    S = set()
    for a in sorted(range(1, N), key=lambda x: 1.0/(N-x), reverse=True):
        if _is_coprime_to_all_fast(a, S):
            S.add(a)
    
    best_S = set(S)
    best_score = _score(best_S)
    current_score = best_score
    
    # 2. Simulated Annealing
    T = 0.02
    alpha = 0.9999
    min_T = 1e-5
    
    while time.time() < deadline - 10:  # Leave 10s for polish
        # Generate Move
        # Bias: 50% Swap, 30% Add, 20% Remove
        r = random.random()
        new_S = set(S)
        delta = 0.0
        move_made = False
        
        if r < 0.5:  # Swap
            s = random.choice(list(S))
            a = random.randint(1, N-1)
            if a not in S:
                # Check validity
                if all(math.gcd(a, b) == 1 for b in S if b != s):
                    new_S.remove(s)
                    new_S.add(a)
                    delta = 1.0/(N-a) - 1.0/(N-s)
                    move_made = True
        elif r < 0.8:  # Add
            a = random.randint(1, N-1)
            if a not in S:
                if _is_coprime_to_all_fast(a, S):
                    new_S.add(a)
                    delta = 1.0/(N-a)
                    move_made = True
        else:  # Remove
            s = random.choice(list(S))
            new_S.remove(s)
            delta = -1.0/(N-s)
            move_made = True
        
        if move_made:
            # Acceptance
            if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                S = new_S
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_S = set(S)
        
        # Cool
        T *= alpha
        if T < min_T:
            T = min_T
    
    # 3. Local Search Polish on best_S
    S = best_S
    current_score = best_score
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        
        # Try Add
        for a in range(N-1, 0, -1):  # Prefer higher values
            if a not in S and _is_coprime_to_all_fast(a, S):
                S.add(a)
                current_score += 1.0/(N-a)
                if current_score > best_score:
                    best_score = current_score
                    best_S = set(S)
                improved = True
                break  # Restart loop to check other adds/swaps
        
        # Try Swap
        S_list = list(S)
        for s in S_list:
            for a in range(N-1, 0, -1):  # Prefer higher values
                if a not in S:
                    if all(math.gcd(a, b) == 1 for b in S if b != s):
                        gain = 1.0/(N-a) - 1.0/(N-s)
                        if gain > 0:
                            S.remove(s)
                            S.add(a)
                            current_score += gain
                            if current_score > best_score:
                                best_score = current_score
                                best_S = set(S)
                            improved = True
                            break  # Restart loop
            if improved:
                break
        
        if time.time() > deadline - 1:
            break
    
    return list(best_S)
