import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin from 150s
    
    N = 512
    
    # Precompute prime factors for fast coprime checking
    def prime_factors(n):
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
    
    factors = {a: prime_factors(a) for a in range(1, N)}
    factors[1] = set()  # 1 has no prime factors
    
    def is_coprime(a, b):
        if a == 1 or b == 1:
            return True
        return len(factors[a] & factors[b]) == 0
    
    # Weights: 1/(512 - a), sorted descending by weight (i.e., a descending)
    candidates = list(range(511, 0, -1))
    weights = {a: 1.0 / (N - a) for a in candidates}
    
    # Greedy construction: pick highest weight first if coprime to all selected
    selected = []
    for a in candidates:
        if all(is_coprime(a, b) for b in selected):
            selected.append(a)
    
    current_score = sum(weights[a] for a in selected)
    
    # Local search with swap improvements
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        best_swap = None
        best_new_score = current_score
        selected_set = set(selected)
        
        # For each element in selected, try removing it and finding best replacement
        for a_out in selected:
            remaining = [b for b in selected if b != a_out]
            
            # Try adding any candidate not in selected
            for a_in in candidates:
                if a_in in selected_set:
                    continue
                
                # Check if a_in is coprime to all in remaining
                if all(is_coprime(a_in, b) for b in remaining):
                    new_score = current_score - weights[a_out] + weights[a_in]
                    if new_score > best_new_score + 1e-12:
                        best_new_score = new_score
                        best_swap = (a_out, a_in)
                        improved = True
        
        if improved:
            a_out, a_in = best_swap
            selected.remove(a_out)
            selected.append(a_in)
            current_score = best_new_score
    
    return selected
