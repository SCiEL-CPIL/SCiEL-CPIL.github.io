import time
import math

_START = time.time()
_HARD_DEADLINE = _START + 85.0

def _prime_factors(num):
    if num <= 1:
        return frozenset()
    out = []
    x = num
    if x % 2 == 0:
        out.append(2)
        while x % 2 == 0:
            x //= 2
    f = 3
    while f * f <= x:
        if x % f == 0:
            out.append(f)
            while x % f == 0:
                x //= f
        f += 2
    if x > 1:
        out.append(x)
    return frozenset(out)

def construct():
    deadline = time.time() + 85.0
    n = 12
    max_val = 10000
    
    # Precompute prime factors for numbers up to 20000
    max_pf = 20000
    pf_cache = [frozenset() for _ in range(max_pf + 1)]
    for i in range(2, max_pf + 1):
        pf_cache[i] = _prime_factors(i)
    
    best_result = None
    best_score = 10**9
    
    # Strategy: Use base B where B has prime factors P_base.
    # Choose x_i = B * k_i. Then x_i + x_j = B*(k_i+k_j).
    # prime_factors(x_i+x_j) ⊆ P_base ∪ prime_factors(k_i+k_j)
    # We want to minimize |P_base ∪ union over i<j of prime_factors(k_i+k_j)|
    
    # Best bases: those with few prime factors but large enough range
    # B=210: primes {2,3,5,7}, max_k = 47
    # B=120: primes {2,3,5}, max_k = 83  
    # B=60: primes {2,3,5}, max_k = 166
    # B=30: primes {2,3,5}, max_k = 333
    # B=42: primes {2,3,7}, max_k = 238
    # B=105: primes {3,5,7}, max_k = 95
    
    bases_to_try = [210, 120, 60, 30, 42, 105, 84, 168, 336, 672, 1344, 2688, 5376]
    
    for base in bases_to_try:
        if time.time() > deadline - 5.0:
            break
        
        base_factors = frozenset(_prime_factors(base))
        max_k = max_val // base
        
        if max_k < n:
            continue
        
        # Initial solution: k_i = 1, 2, ..., n
        cur_k = list(range(1, n + 1))
        
        def evaluate_k(k_list):
            union = set(base_factors)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    s = k_list[i] + k_list[j]
                    if s <= max_pf:
                        union |= pf_cache[s]
                    else:
                        union |= _prime_factors(s)
            return len(union)
        
        cur_score = evaluate_k(cur_k)
        
        if cur_score < best_score:
            best_score = cur_score
            best_result = [base * k for k in cur_k]
        
        # Local search with single moves and double swaps
        improved = True
        iters = 0
        max_iters = 200
        
        while improved and iters < max_iters and time.time() < deadline - 5.0:
            improved = False
            iters += 1
            
            # Try single replacements
            for i in range(n):
                if time.time() > deadline - 5.0:
                    break
                
                old_k = cur_k[i]
                best_new_k = None
                best_new_score = cur_score
                
                # Try all valid k values
                for new_k in range(1, max_k + 1):
                    if new_k == old_k:
                        continue
                    if new_k in cur_k:
                        continue
                    
                    cur_k[i] = new_k
                    new_score = evaluate_k(cur_k)
                    cur_k[i] = old_k
                    
                    if new_score < best_new_score:
                        best_new_score = new_score
                        best_new_k = new_k
                
                if best_new_k is not None:
                    cur_k[i] = best_new_k
                    cur_score = best_new_score
                    improved = True
                    if cur_score < best_score:
                        best_score = cur_score
                        best_result = [base * k for k in cur_k]
                    break
            
            # If no single move improved, try double swaps
            if not improved:
                for i in range(n - 1):
                    if time.time() > deadline - 5.0:
                        break
                    for j in range(i + 1, n):
                        if time.time() > deadline - 5.0:
                            break
                        
                        old_ik = cur_k[i]
                        old_jk = cur_k[j]
                        best_swap = None
                        best_swap_score = cur_score
                        
                        # Try swapping with other pairs
                        for a in range(n):
                            if a == i or a == j:
                                continue
                            for b in range(n):
                                if b == i or b == j or b == a:
                                    continue
                                if a >= b:
                                    continue
                                
                                new_ik = cur_k[a]
                                new_jk = cur_k[b]
                                
                                if new_ik == old_ik and new_jk == old_jk:
                                    continue
                                if new_ik == old_jk and new_jk == old_ik:
                                    continue
                                
                                cur_k[i] = new_ik
                                cur_k[j] = new_jk
                                cur_k[a] = old_ik
                                cur_k[b] = old_jk
                                
                                new_score = evaluate_k(cur_k)
                                
                                cur_k[i] = old_ik
                                cur_k[j] = old_jk
                                cur_k[a] = cur_k[a]  # already restored
                                cur_k[b] = cur_k[b]
                                
                                if new_score < best_swap_score:
                                    best_swap_score = new_score
                                    best_swap = (i, j, a, b, new_ik, new_jk)
                        
                        if best_swap is not None:
                            i_, j_, a_, b_, nik, njk = best_swap
                            cur_k[i_] = nik
                            cur_k[j_] = njk
                            cur_score = best_swap_score
                            improved = True
                            if cur_score < best_score:
                                best_score = cur_score
                                best_result = [base * k for k in cur_k]
                            break
                
                if improved:
                    break
        
        if cur_score < best_score:
            best_score = cur_score
            best_result = [base * k for k in cur_k]
        
        if best_score <= 4:
            break
    
    # Direct search on smooth numbers if base approach didn't do well
    if best_score > 5 and time.time() < deadline - 15.0:
        # Build pool of 2,3,5,7-smooth numbers
        pool = set()
        for a in range(0, 15):
            for b in range(0, 10):
                for c in range(0, 7):
                    for d in range(0, 6):
                        x = (2 ** a) * (3 ** b) * (5 ** c) * (7 ** d)
                        if 1 <= x <= max_val:
                            pool.add(x)
        
        pool = sorted(pool)
        
        if len(pool) >= n:
            cur_x = list(pool[:n])
            
            def evaluate_x(x_list):
                union = set()
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        s = x_list[i] + x_list[j]
                        if s <= max_pf:
                            union |= pf_cache[s]
                        else:
                            union |= _prime_factors(s)
                return len(union)
            
            cur_score = evaluate_x(cur_x)
            
            if cur_score < best_score:
                best_score = cur_score
                best_result = list(cur_x)
            
            # Local search
            improved = True
            iters = 0
            while improved and iters < 100 and time.time() < deadline - 5.0:
                improved = False
                iters += 1
                
                for i in range(n):
                    if time.time() > deadline - 5.0:
                        break
                    
                    old_x = cur_x[i]
                    best_new_x = None
                    best_new_score = cur_score
                    
                    # Try nearby smooth numbers
                    idx = pool.index(old_x) if old_x in pool else 0
                    lo = max(0, idx - 20)
                    hi = min(len(pool), idx + 80)
                    
                    for new_x in pool[lo:hi]:
                        if new_x == old_x:
                            continue
                        if new_x in cur_x:
                            continue
                        
                        cur_x[i] = new_x
                        new_score = evaluate_x(cur_x)
                        cur_x[i] = old_x
                        
                        if new_score < best_new_score:
                            best_new_score = new_score
                            best_new_x = new_x
                    
                    if best_new_x is not None:
                        cur_x[i] = best_new_x
                        cur_score = best_new_score
                        improved = True
                        if cur_score < best_score:
                            best_score = cur_score
                            best_result = list(cur_x)
                        break
                
                if cur_score < best_score:
                    best_score = cur_score
                    best_result = list(cur_x)
    
    if best_result is None:
        best_result = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    
    return best_result
