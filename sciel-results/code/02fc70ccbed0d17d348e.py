"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    deadline = time.time() + 50  # safety margin before 60s limit
    best_val = None
    best_set = None
    
    n = 210
    max_bound = 10210
    
    # Strategy 1: 2-element solutions
    # a * b * 210 = perfect square => a * b = 210 * k^2
    # Need a, b in (210, 10210], a != b
    for k in range(1, 200):
        if time.time() > deadline:
            break
        target = 210 * k * k
        # Find factor pairs (a, b) with a < b, a > 210, b <= 10210
        # a must divide target, a in range [211, min(target//211, 10210)]
        # b = target // a, must be > 210, <= 10210, a != b
        max_a = min(target // 211, 10210)
        if max_a < 211:
            continue
        # Iterate a from 211 to sqrt(target) to find smallest max
        # Actually, to minimize max(a,b), we want a and b close to sqrt(target)
        # So iterate a from min(sqrt(target), max_a) downward to 211
        sqrt_t = math.isqrt(target)
        start_a = min(sqrt_t, max_a)
        found = False
        for a in range(start_a, 210, -1):
            if target % a == 0:
                b = target // a
                if a != b and 211 <= a <= 10210 and 211 <= b <= 10210:
                    val = max(a, b) - 210
                    if best_val is None or val < best_val:
                        best_val = val
                        best_set = [a, b]
                    found = True
                    break  # first found going down from sqrt gives smallest max
        if found and best_val is not None and best_val <= 1:
            break
    
    # Strategy 2: 3-element solutions (if needed)
    # a * b * c * 210 = perfect square
    # This is more complex; try small combinations
    if best_val is None or best_val > 5:
        if time.time() < deadline - 10:
            # Try finding 3 numbers whose product * 210 is a square
            # Use parity vectors mod 2 for primes
            # For small ranges, enumerate
            lo = 211
            hi = min(210 + best_val, 10210) if best_val else 250
            
            # For 3 elements, we need parity(a) XOR parity(b) XOR parity(c) = parity(210)
            # parity(210) has bits set for primes 2,3,5,7
            # Let's try to find triples where max is small
            # This is expensive, so limit the range
            search_hi = min(210 + (best_val if best_val else 30), 280)
            
            # Precompute parity signatures for numbers in range
            # Signature: for each prime p, v_p(n) mod 2
            # We only care about primes up to 10210, but for small numbers, few primes matter
            # Actually, for the product to be a square, ALL prime exponents must be even
            # So we need the XOR of signatures to equal signature of 210
            
            # This is getting complex. Let's try a different approach for 3 elements:
            # a*b*c = 210 * k^2
            # For small k, enumerate factorizations
            
            for k in range(1, 50):
                if time.time() > deadline - 5:
                    break
                target3 = 210 * k * k
                # Find 3 distinct factors a,b,c > 210, <= 10210, product = target3
                # a*b*c = target3
                # a > 210, so a*b*c > 210^3 = 9261000
                # target3 = 210*k^2, so k^2 > 9261000/210 = 44100, k > 210
                # That's too large for small k. 3 elements with small max might not work
                # unless the numbers are large enough
                if target3 < 211 * 212 * 213:
                    continue
                # This approach is too slow for general case
                pass
    
    # Strategy 3: Verify and return
    if best_set is None:
        # Fallback: single element 4*210 = 840
        best_set = [840]
        best_val = 840 - 210
    
    # Verify the result
    prod = 1
    for x in best_set:
        prod *= x
    prod *= 210
    if not _is_perfect_square(prod):
        # Fallback
        best_set = [840]
        best_val = 630
    
    # Ensure types are correct
    result = [int(x) for x in best_set]
    return result

# EVOLVE-BLOCK-END
