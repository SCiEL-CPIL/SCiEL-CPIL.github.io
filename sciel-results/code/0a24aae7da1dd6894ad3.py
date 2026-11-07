import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin
    
    N_MIN = 211
    N_MAX = 10210
    
    # Target: sf(210) = 2*3*5*7 (all primes with odd exponent)
    # We need product of selected numbers to have sf = 210
    
    best_solution = None
    best_max = N_MAX + 1  # effectively infinity
    
    # --- Strategy 1: Algebraic pair search ---
    # Find a, b with 210 < a <= b, a*b = 210 * t^2 for some t
    # This means sf(a) ^ sf(b) = sf(210)
    
    # For each t, N = 210 * t^2
    # Find factor pairs (a, b) of N with a <= b, 210 < a
    # Track minimum max(a, b) = b
    
    t = 1
    while time.time() < deadline - 10:
        N = 210 * t * t
        # Find all divisors a of N with 210 < a <= sqrt(N)
        # Then b = N // a
        # We need a > 210 and b >= a (so a <= sqrt(N))
        
        sqrt_N = math.isqrt(N)
        # a must be > 210 and <= sqrt_N
        # Also a must divide N
        
        # Instead of iterating all a, iterate divisors
        # For small t, N is small, so we can iterate
        
        found_for_t = False
        # Iterate a from max(211, ...) to sqrt_N
        # But this is slow for large t. Let's factor N instead.
        
        # Factor N
        temp = N
        factors = {}
        d = 2
        while d * d <= temp:
            while temp % d == 0:
                factors[d] = factors.get(d, 0) + 1
                temp //= d
            d += 1 if d == 2 else 2
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1
        
        # Generate all divisors of N
        divisors = [1]
        for p, exp in factors.items():
            new_divs = []
            for d in divisors:
                val = d
                for e in range(exp + 1):
                    new_divs.append(val)
                    val *= p
            divisors = new_divs
        
        # Find pairs (a, b) with a * b = N, 210 < a <= b
        for a in divisors:
            if a <= 210:
                continue
            b = N // a
            if a * b != N:
                continue
            if b < a:
                continue
            # a <= b, a > 210, b >= a
            # Check b <= N_MAX
            if b > N_MAX:
                continue
            cur_max = b
            if cur_max < best_max:
                # Verify: 210 * a * b should be a perfect square
                prod = 210 * a * b
                s = math.isqrt(prod)
                if s * s == prod:
                    best_max = cur_max
                    best_solution = [a, b]
                    found_for_t = True
        
        # If we found a solution with max = 211 + something small, 
        # we can likely stop early
        if best_max <= 250:
            break
        
        t += 1
    
    # --- Strategy 2: Gaussian elimination (fallback / cross-check) ---
    # Find minimal M such that sf(210) is in span of {sf(x) : x in [211, M]}
    if time.time() < deadline - 5:
        # Sieve
        is_prime = [True] * (N_MAX + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(N_MAX)) + 1):
            if is_prime[i]:
                for j in range(i*i, N_MAX + 1, i):
                    is_prime[j] = False
        
        primes = [i for i, prime in enumerate(is_prime) if prime]
        prime_to_idx = {p: i for i, p in enumerate(primes)}
        
        # Compute parity bitsets
        def parity_bitset(x):
            temp = x
            b = 0
            for p in primes:
                if p * p > temp:
                    break
                if temp % p == 0:
                    count = 0
                    while temp % p == 0:
                        temp //= p
                        count += 1
                    if count % 2 == 1:
                        b |= (1 << prime_to_idx[p])
            if temp > 1:
                b |= (1 << prime_to_idx[temp])
            return b
        
        # Target
        target = parity_bitset(210)
        
        basis = []
        for M in range(N_MIN, N_MAX + 1):
            if time.time() > deadline - 5:
                break
            v = parity_bitset(M)
            m = 1 << (M - N_MIN)
            
            current_v = v
            current_m = m
            
            for b_v, b_m in basis:
                pivot_bit = 1 << (b_v.bit_length() - 1)
                if current_v & pivot_bit:
                    current_v ^= b_v
                    current_m ^= b_m
            
            if current_v != 0:
                basis.append((current_v, current_m))
                basis.sort(key=lambda x: x[0].bit_length(), reverse=True)
            
            # Check if target is in span
            w = target
            result_mask = 0
            for b_v, b_m in basis:
                pivot_bit = 1 << (b_v.bit_length() - 1)
                if w & pivot_bit:
                    w ^= b_v
                    result_mask ^= b_m
            
            if w == 0:
                # Extract selected numbers
                selected = []
                for i in range(M - N_MIN + 1):
                    if (result_mask >> i) & 1:
                        selected.append(N_MIN + i)
                sel_max = max(selected)
                if sel_max < best_max:
                    best_max = sel_max
                    best_solution = selected
                break  # M is minimal, so no need to continue
    
    # Fallback
    if best_solution is None:
        best_solution = [840]
    
    # Final verification
    prod = 210
    for x in best_solution:
        prod *= x
    s = math.isqrt(prod)
    if s * s != prod:
        # Fallback to known valid solution
        best_solution = [840]
    
    return list(best_solution)
