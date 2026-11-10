import math

def construct():
    # 1. Generate primes < 512 and prime factors for all numbers
    max_n = 512
    factors = [set() for _ in range(max_n)]
    for p in range(2, max_n):
        is_prime = True
        for i in range(2, int(math.sqrt(p)) + 1):
            if p % i == 0:
                is_prime = False
                break
        if is_prime:
            for multiple in range(p, max_n, p):
                factors[multiple].add(p)
                
    primes = [i for i in range(2, max_n) if len(factors[i]) == 1]
    
    # Initialize with all primes
    chosen = set(primes)
    
    # Precompute weights
    weights = [0.0] * max_n
    for a in range(1, max_n):
        weights[a] = 1.0 / (512 - a)
    
    # Helper to check if x is coprime to all in set S using precomputed factors
    def is_coprime_to_set(x, S):
        fx = factors[x]
        for y in S:
            if fx & factors[y]:
                return False
        return True
        
    # 2. Greedy Merging Phase
    improved = True
    while improved:
        improved = False
        current_list = list(chosen)
        for i in range(len(current_list)):
            for j in range(i + 1, len(current_list)):
                a = current_list[i]
                b = current_list[j]
                c = a * b
                if c >= max_n:
                    continue
                
                # Check if c is coprime to the rest
                # Optimization: Union of factors of rest
                # Since chosen is small (< 100), this is fast enough
                rest = [x for x in chosen if x != a and x != b]
                if is_coprime_to_set(c, set(rest)):
                    gain = weights[c] - weights[a] - weights[b]
                    if gain > 1e-9:
                        chosen.remove(a)
                        chosen.remove(b)
                        chosen.add(c)
                        improved = True
                        break
            if improved:
                break
                
    # 3. Local Search Phase (1-swap)
    improved = True
    max_iters = 1000
    iter_count = 0
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        current_list = list(chosen)
        for i in range(len(current_list)):
            a_out = current_list[i]
            rest = [x for x in chosen if x != a_out]
            rest_set = set(rest)
            
            best_in = -1
            best_gain = 0
            
            for a_in in range(1, max_n):
                if a_in in chosen:
                    continue
                if weights[a_in] <= weights[a_out]:
                    continue
                if is_coprime_to_set(a_in, rest_set):
                    gain = weights[a_in] - weights[a_out]
                    if gain > best_gain:
                        best_gain = gain
                        best_in = a_in
                        
            if best_in != -1:
                chosen.remove(a_out)
                chosen.add(best_in)
                improved = True
                break
                
    return list(chosen)
