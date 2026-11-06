import time
import math

_START = time.time()
_HARD_DEADLINE = _START + 85.0  # Safety margin before 90s kill

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
    
    # Core insight: choose x_i = B * k_i where B is a large smooth number.
    # Then x_i + x_j = B * (k_i + k_j), and
    # prime_factors(x_i + x_j) ⊆ prime_factors(B) ∪ prime_factors(k_i + k_j).
    # So total distinct primes = |prime_factors(B) ∪ union of prime_factors(k_i + k_j)|.
    
    # Strategy:
    # 1. Precompute prime factors for small numbers (k_i + k_j up to ~200)
    # 2. For various bases B, search for k_i values that minimize the union
    # 3. Use simulated annealing / hill climbing on the k-space
    
    # Precompute prime factors for numbers up to 300
    max_sum = 300
    pf_cache = [frozenset() for _ in range(max_sum + 1)]
    for i in range(2, max_sum + 1):
        pf_cache[i] = _prime_factors(i)
    
    # Candidate bases: smooth numbers with many small prime factors
    # We want B to be large enough that B*k_i <= 10000 for our k_i range
    # B = 2520 = 2^3 * 3^2 * 5 * 7 (primes: {2,3,5,7})
    # B = 840 = 2^3 * 3 * 5 * 7 (primes: {2,3,5,7})
    # B = 360 = 2^3 * 3^2 * 5 (primes: {2,3,5})
    # B = 120 = 2^3 * 3 * 5 (primes: {2,3,5})
    # B = 60 = 2^2 * 3 * 5 (primes: {2,3,5})
    # B = 30 = 2 * 3 * 5 (primes: {2,3,5})
    
    # Let's focus on B=2520 (gives 4 base primes, k_i up to 3)
    # B=840 (4 base primes, k_i up to 11)
    # B=360 (3 base primes, k_i up to 27)
    # B=120 (3 base primes, k_i up to 83)
    # B=60 (3 base primes, k_i up to 166)
    # B=30 (3 base primes, k_i up to 333)
    
    # Actually, let me think about this differently.
    # The minimum possible answer is at least 2 (since we need at least one even sum and one odd sum,
    # or we could have all even sums giving prime 2, but then we need another prime for some sums).
    # 
    # Actually, if all x_i are even, all sums are even, so 2 is always a factor.
    # If all x_i are odd, all sums are even, so 2 is always a factor.
    # If we mix even and odd, some sums are odd.
    #
    # Can we get just {2}? That would require all pairwise sums to be powers of 2.
    # x_i + x_j = 2^m(i,j) for all i≠j. This is very restrictive.
    # For 3 elements: x1+x2=2^a, x1+x3=2^b, x2+x3=2^c.
    # Adding first two minus third: 2*x1 = 2^a + 2^b - 2^c.
    # This requires 2^a + 2^b - 2^c to be even and positive, so a,b,c must work out.
    # For 12 elements this is extremely unlikely to work.
    #
    # Can we get {2, 3}? All sums must be of form 2^a * 3^b.
    # This is more feasible.
    
    # Let me try a direct search approach with a good initial solution.
    
    # Approach: Start with a known good configuration and improve via local search.
    # Key: use the base-multiplier structure for efficiency.
    
    best_result = None
    best_score = 10**9
    
    # Try multiple base values
    bases_to_try = [2520, 840, 360, 120, 60, 30, 180, 720, 504, 252, 168, 84, 42, 24, 12, 6, 4, 8, 16, 28, 56, 112, 224, 448, 210, 420, 630, 1260]
    
    for base in bases_to_try:
        if time.time() > deadline - 5.0:
            break
        
        base_factors = _prime_factors(base)
        max_k = max_val // base
        
        if max_k < n:
            continue
        
        # We need n distinct k_i from 1..max_k
        # Start with k_i = 1, 2, ..., n
        cur_k = list(range(1, n + 1))
        
        def evaluate_k(k_list):
            """Evaluate the score for given k values with this base."""
            union = set(base_factors)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    s = k_list[i] + k_list[j]
                    if s <= max_sum:
                        union |= pf_cache[s]
                    else:
                        union |= _prime_factors(s)
            return len(union)
        
        cur_score = evaluate_k(cur_k)
        
        # Local search: try replacing each k_i with other values
        improved = True
        max_iters = 500
        iters = 0
        
        while improved and iters < max_iters and time.time() < deadline - 5.0:
            improved = False
            iters += 1
            
            # Try all possible single-element replacements
            for i in range(n):
                if time.time() > deadline - 5.0:
                    break
                
                old_k = cur_k[i]
                best_new_k = None
                best_new_score = cur_score
                
                # Try values from 1 to min(max_k, 100) - focus on small values
                search_limit = min(max_k, 100)
                
                for new_k in range(1, search_limit + 1):
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
                    break  # Restart i-loop after improvement
        
        # Update global best
        if cur_score < best_score:
            best_score = cur_score
            best_result = [base * k for k in cur_k]
        
        if best_score <= 4:
            break  # Excellent result
    
    # If base approach didn't find something great, try direct search
    if best_score > 5 and time.time() < deadline - 15.0:
        # Try with smooth numbers directly
        # Build a pool of numbers with few prime factors
        pool = []
        for d in range(1, max_val + 1):
            x = d
            while x % 2 == 0:
                x //= 2
            while x % 3 == 0:
                x //= 3
            while x % 5 == 0:
                x //= 5
            if x == 1:
                pool.append(d)
        
        # Sort by number of prime factors, then by value
        pool = sorted(set(pool), key=lambda x: (len(_prime_factors(x)), x))
        
        # Start with first n smooth numbers
        cur_x = list(pool[:n])
        
        def evaluate_x(x_list):
            union = set()
            for i in range(n - 1):
                for j in range(i + 1, n):
                    s = x_list[i] + x_list[j]
                    union |= _prime_factors(s)
            return len(union)
        
        cur_score = evaluate_x(cur_x)
        
        if cur_score < best_score:
            best_score = cur_score
            best_result = list(cur_x)
        
        # Local search on x values
        improved = True
        iters = 0
        while improved and iters < 200 and time.time() < deadline - 5.0:
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
                candidates = pool[max(0, idx-10):min(len(pool), idx+50)]
                
                for new_x in candidates:
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
    
    # Fallback if nothing found
    if best_result is None:
        best_result = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    
    return best_result
