import math
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    # Primes p ≡ 1 mod 4
    primes_1mod4 = [5, 13, 17, 29, 37, 41, 53, 61, 73, 89, 97, 101, 109, 113, 137, 149, 157, 173, 181, 193, 197, 229, 233, 241, 257]
    
    # Precompute representations a, b for a^2 + b^2 = p
    p_to_rep = {}
    for p in primes_1mod4:
        found = False
        for a in range(1, int(math.isqrt(p)) + 1):
            b2 = p - a*a
            b = int(math.isqrt(b2))
            if b*b == b2:
                p_to_rep[p] = (a, b)
                found = True
                break
        if not found:
            p_to_rep[p] = (1, 0)

    max_val = 10**18
    MAX_K = 2001  # Max product of (2e_i + 1)
    
    best_M = 1
    best_K = 1
    best_exponents = {}
    
    search_primes = primes_1mod4[:15]
    
    def dfs(idx, current_M, current_K, exponents):
        nonlocal best_M, best_K, best_exponents
        
        if time.time() > deadline - 20.0:
            return

        if current_K > best_K:
            best_K = current_K
            best_M = current_M
            best_exponents = dict(exponents)
            
        if idx >= len(search_primes):
            return
            
        p = search_primes[idx]
        
        val = current_M
        exp = 0
        while val <= max_val:
            # Factor in K is (2*exp + 1)
            next_K = current_K * (2 * exp + 1)
            
            if next_K > MAX_K:
                break
            
            exponents[p] = exp
            dfs(idx + 1, val, next_K, exponents)
            
            exp += 1
            val = val * p * p
            if val > max_val:
                break
                
        if p in exponents:
            del exponents[p]
            
    dfs(0, 1, 1, {})
    
    if best_M == 1:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}
        
    used_primes = [p for p in search_primes if p in best_exponents and best_exponents[p] > 0]
    
    if not used_primes:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    exps = [best_exponents[p] for p in used_primes]
    
    def complex_mul(c1, c2):
        re1, im1 = c1
        re2, im2 = c2
        return (re1*re2 - im1*im2, re1*im2 + im1*re2)
        
    prime_powers = []
    for i, p in enumerate(used_primes):
        a, b = p_to_rep[p]
        pi = (a, b)
        pi_bar = (a, -b)
        e = exps[i]
        
        # We need powers of pi up to 2e
        powers_pi = [(1, 0)]
        for k in range(1, 2*e+1):
            powers_pi.append(complex_mul(powers_pi[-1], pi))
            
        # We need powers of pi_bar up to 2e
        powers_pi_bar = [(1, 0)]
        for k in range(1, 2*e+1):
            powers_pi_bar.append(complex_mul(powers_pi_bar[-1], pi_bar))
            
        prime_powers.append((powers_pi, powers_pi_bar))
        
    pairs_set = set()
    
    # Ranges for a_i: 0 to 2e_i
    ranges = [range(2*e + 1) for e in exps]
    
    for as_tuple in itertools.product(*ranges):
        z = (1, 0)
        for i in range(len(used_primes)):
            a_i = as_tuple[i]
            e_i = exps[i]
            powers_pi, powers_pi_bar = prime_powers[i]
            
            # z_i = pi_i^{a_i} * pi_bar_i^{2e_i - a_i}
            part1 = powers_pi[a_i]
            part2 = powers_pi_bar[2*e_i - a_i]
            
            z = complex_mul(z, part1)
            z = complex_mul(z, part2)
            
        u, v = z
        u = abs(u)
        v = abs(v)
        
        if u == 0 or v == 0:
            continue
            
        a_val = u*u
        b_val = v*v
        
        if a_val > b_val:
            a_val, b_val = b_val, a_val
            
        pairs_set.add((a_val, b_val))
        
        if time.time() > deadline - 10.0:
            break

    if not pairs_set:
        return {'N': 2, 'pairs': [[1, 1]]}

    pairs = sorted(list(pairs_set))
    
    # Ensure we don't exceed 1000 pairs
    if len(pairs) > 1000:
        pairs = pairs[:1000]
        
    N = best_M
    if N < 2 or N > 10**18:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': N, 'pairs': [[a, b] for a, b in pairs]}
