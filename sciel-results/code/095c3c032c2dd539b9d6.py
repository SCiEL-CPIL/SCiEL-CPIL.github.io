"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def _powerful_part(n):
    """Compute the powerful part of n: product of p^e for e>=2 in factorization."""
    if n <= 1:
        return 1
    pp = 1
    m = n
    p = 2
    while p * p <= m:
        if m % p == 0:
            e = 0
            while m % p == 0:
                m //= p
                e += 1
            if e >= 2:
                pp *= p ** e
        p += 1
    if m > 1:
        pass  # prime with exponent 1, no contribution
    return pp

def _generate_powerful_numbers(limit):
    """Generate all powerful numbers up to limit."""
    powerful = set()
    b_max = int(limit ** (1/3)) + 2
    for b in range(1, b_max + 1):
        b3 = b ** 3
        a_max = isqrt(limit // b3)
        for a in range(1, a_max + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    limit = 1000001
    powerful_nums = _generate_powerful_numbers(limit)
    powerful_set = set(powerful_nums)
    
    best_n = 10000
    best_score = 0
    
    # Phase 1: Find consecutive powerful numbers
    for pn in powerful_nums:
        if pn > 1000000:
            break
        if pn < 10000:
            continue
        if (pn + 1) in powerful_set:
            score = pn * (pn + 1)
            if score > best_score:
                best_score = score
                best_n = pn
    
    # Phase 2: Broader search with deadline guard
    # Focus on numbers with large square factors
    # A number n = a^2 * s (s squarefree) has powerful_part >= a^2
    # We want both n and n+1 to have large square parts
    
    # Strategy: iterate over possible square factors
    # For n in [10000, 1000000], n = a^2 * s where s is squarefree
    # a^2 <= 1000000, so a <= 1000
    # For each a, s ranges such that 10000 <= a^2*s <= 1000000
    
    # But this is still complex. Let's try a different approach:
    # Sample strategically and check powerful parts
    
    elapsed = time.time() - start_time
    if elapsed < 65:
        # Check numbers that are perfect squares and their neighbors
        # and numbers one away from perfect squares
        checked = set()
        
        # Perfect squares in range
        a_min = isqrt(10000)
        a_max = isqrt(1000000)
        for a in range(a_min, a_max + 1):
            n = a * a
            if 10000 <= n <= 1000000 and n not in checked:
                checked.add(n)
                pp_n = _powerful_part(n)
                pp_np1 = _powerful_part(n + 1)
                score = pp_n * pp_np1
                if score > best_score:
                    best_score = score
                    best_n = n
            if time.time() - start_time > deadline - 5:
                break
        
        # Numbers one less than perfect squares
        if time.time() - start_time < deadline - 5:
            for a in range(a_min, a_max + 1):
                n = a * a - 1
                if 10000 <= n <= 1000000 and n not in checked:
                    checked.add(n)
                    pp_n = _powerful_part(n)
                    pp_np1 = _powerful_part(n + 1)
                    score = pp_n * pp_np1
                    if score > best_score:
                        best_score = score
                        best_n = n
                if time.time() - start_time > deadline - 5:
                    break
        
        # Numbers with large square factors: n = (a*k)^2 * s where s small
        # Let's check n where n is divisible by a large square
        if time.time() - start_time < deadline - 5:
            # Check multiples of squares
            for a in range(100, a_max + 1):  # squares >= 10000
                sq = a * a
                # n = sq * s, s = 1, 2, 3, ... small
                for s in range(1, 10):
                    n = sq * s
                    if n > 1000000:
                        break
                    if n < 10000:
                        continue
                    if n not in checked:
                        checked.add(n)
                        pp_n = _powerful_part(n)
                        pp_np1 = _powerful_part(n + 1)
                        score = pp_n * pp_np1
                        if score > best_score:
                            best_score = score
                            best_n = n
                if time.time() - start_time > deadline - 5:
                    break
    
    return best_n
# EVOLVE-BLOCK-END
