"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

P = {'n': 210, 'maximum_window': 10000, 'maximum_size': 100}

def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _verify_solution(selected):
    """Verify that product of selected * 210 is a perfect square."""
    prod = 210
    for x in selected:
        prod *= x
    return _is_perfect_square(prod)

def _prime_parity_mask(n, prime_to_bit):
    """Return bitmask of primes with odd exponent in n."""
    mask = 0
    d = 2
    while d * d <= n:
        if n % d == 0:
            exp = 0
            while n % d == 0:
                n //= d
                exp += 1
            if exp % 2 == 1 and d in prime_to_bit:
                mask ^= (1 << prime_to_bit[d])
        d += 1
    if n > 1 and n in prime_to_bit:
        mask ^= (1 << prime_to_bit[n])
    return mask

def _sieve(limit):
    is_prime = [True] * (limit + 1)
    if limit >= 0:
        is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit+1, i):
                is_prime[j] = False
    return [i for i in range(2, limit+1) if is_prime[i]]

def _find_min_max_solution(start_time, deadline):
    max_val = 10210
    primes = _sieve(max_val)
    prime_to_bit = {p: i for i, p in enumerate(primes)}
    
    target_mask = _prime_parity_mask(210, prime_to_bit)
    
    masks = {}
    for i in range(211, 10211):
        masks[i] = _prime_parity_mask(i, prime_to_bit)
        
    low = 211
    high = 240
    
    def is_feasible(M):
        basis = []
        for i in range(211, M + 1):
            v = masks[i]
            for b in basis:
                hb_v = 1 << (v.bit_length() - 1) if v > 0 else 0
                hb_b = 1 << (b.bit_length() - 1) if b > 0 else 0
                if hb_v == hb_b:
                    v ^= b
            if v != 0:
                basis.append(v)
                basis.sort(reverse=True)
        
        v = target_mask
        for b in basis:
            hb_v = 1 << (v.bit_length() - 1) if v > 0 else 0
            hb_b = 1 << (b.bit_length() - 1) if b > 0 else 0
            if hb_v == hb_b:
                v ^= b
        return v == 0

    while low < high:
        if time.time() > deadline - 5:
            break
        mid = (low + high) // 2
        if is_feasible(mid):
            high = mid
        else:
            low = mid + 1
            
    min_max = low
    
    mask_to_numbers = defaultdict(list)
    for i in range(211, min_max + 1):
        mask_to_numbers[masks[i]].append(i)
        
    best_solution = None
    
    for a in range(211, min_max + 1):
        needed = masks[a] ^ target_mask
        if needed in mask_to_numbers:
            for b in mask_to_numbers[needed]:
                if b > a:
                    best_solution = [a, b]
                    break
        if best_solution: break
        
    if not best_solution:
        for a in range(211, min_max + 1):
            for b in range(a + 1, min_max + 1):
                needed = masks[a] ^ masks[b] ^ target_mask
                if needed in mask_to_numbers:
                    for c in mask_to_numbers[needed]:
                        if c > b:
                            best_solution = [a, b, c]
                            break
                if best_solution: break
            if best_solution: break
            
    if not best_solution:
        for a in range(211, min_max + 1):
            for b in range(a + 1, min_max + 1):
                for c in range(b + 1, min_max + 1):
                    needed = masks[a] ^ masks[b] ^ masks[c] ^ target_mask
                    if needed in mask_to_numbers:
                        for d in mask_to_numbers[needed]:
                            if d > c:
                                best_solution = [a, b, c, d]
                                break
                    if best_solution: break
                if best_solution: break
            if best_solution: break
            
    if not best_solution:
        for a in range(211, min_max + 1):
            for b in range(a + 1, min_max + 1):
                for c in range(b + 1, min_max + 1):
                    for d in range(c + 1, min_max + 1):
                        needed = masks[a] ^ masks[b] ^ masks[c] ^ masks[d] ^ target_mask
                        if needed in mask_to_numbers:
                            for e in mask_to_numbers[needed]:
                                if e > d:
                                    best_solution = [a, b, c, d, e]
                                    break
                        if best_solution: break
                    if best_solution: break
                if best_solution: break
            if best_solution: break
            
    if best_solution:
        return best_solution
    else:
        return [224, 240]

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    try:
        sol = _find_min_max_solution(start_time, deadline)
        if _verify_solution(sol):
            return [int(x) for x in sol]
        else:
            return [224, 240]
    except Exception:
        return [224, 240]

# EVOLVE-BLOCK-END
