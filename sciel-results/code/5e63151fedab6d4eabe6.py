"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

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

def _factorize(n, primes):
    """Factorize n using given primes. Returns list of (prime, exponent)."""
    factors = []
    temp = n
    for p in primes:
        if p * p > temp:
            break
        if temp % p == 0:
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            factors.append((p, exp))
    if temp > 1:
        factors.append((temp, 1))
    return factors

def _squarefree_mask(n, prime_to_bit):
    """Return bitmask of primes with odd exponent in n's squarefree part."""
    mask = 0
    temp = n
    for p in sorted(prime_to_bit.keys()):
        if p * p > temp:
            break
        if temp % p == 0:
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            if exp % 2 == 1:
                mask ^= (1 << prime_to_bit[p])
    if temp > 1:
        if temp in prime_to_bit:
            mask ^= (1 << prime_to_bit[temp])
        # If temp is a prime not in our list, we can't represent it
        # But for numbers up to 10210, all prime factors are <= 10210
    return mask

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    # Sieve primes up to 10210
    limit = 10210
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit+1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit+1) if is_prime[i]]
    
    # Only keep primes that can appear in numbers 211..10210
    # All primes up to 10210 can appear
    prime_to_bit = {p: i for i, p in enumerate(primes)}
    
    # Target mask for 210 = 2*3*5*7
    target_mask = 0
    for p in [2, 3, 5, 7]:
        target_mask |= (1 << prime_to_bit[p])
    
    # Precompute squarefree masks for 211..10210
    # Use a faster method: precompute smallest prime factor
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, limit+1, i):
                if spf[j] == j:
                    spf[j] = i
    
    masks = {}
    for i in range(211, limit + 1):
        temp = i
        mask = 0
        while temp > 1:
            p = spf[temp]
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            if exp % 2 == 1:
                mask ^= (1 << prime_to_bit[p])
        masks[i] = mask
    
    # Build mask_to_numbers: map from mask to sorted list of numbers
    mask_to_numbers = defaultdict(list)
    for i in range(211, limit + 1):
        mask_to_numbers[masks[i]].append(i)
    
    best_raw = limit - 210  # worst case
    best_solution = None
    
    # Phase 1: Search for 2-element solutions over FULL range
    # For each a, need b such that mask_a XOR mask_b = target_mask
    # i.e., mask_b = mask_a XOR target_mask
    for a in range(211, limit + 1):
        if time.time() > deadline - 40:
            break
        needed = masks[a] ^ target_mask
        if needed in mask_to_numbers:
            for b in mask_to_numbers[needed]:
                if b > a:
                    raw = b - 210
                    if raw < best_raw:
                        # Verify
                        if _verify_solution([a, b]):
                            best_raw = raw
                            best_solution = [a, b]
    
    # Phase 2: Search for 3-element solutions with max < 210 + best_raw
    # Strategy: iterate over the largest element c, then find (a,b) with a<b<c
    # such that mask_a XOR mask_b = target_mask XOR mask_c
    # Precompute pair_mask_to_pairs for efficiency
    upper_bound = min(210 + best_raw - 1, limit)
    if upper_bound > 211:
        # Build mask_to_numbers for the restricted range
        restricted_numbers = list(range(211, upper_bound + 1))
        n_restricted = len(restricted_numbers)
        
        # For 3-element: we need mask_a ^ mask_b ^ mask_c = target
        # => mask_a ^ mask_b = target ^ mask_c
        # Precompute: for each possible pair (a,b) with a<b, compute their combined mask
        # But that's O(n^2) which is too slow for large n
        # Instead: iterate c from upper_bound down, and for each c, 
        # we need to find a<b<c with mask_a ^ mask_b = target ^ mask_c
        
        # Build a frequency map of masks in [211, c-1]
        # and for each c, check if there exist two numbers with masks that XOR to needed
        
        # More efficient: build mask_count dict incrementally
        # For each c (from high to low), maintain count of each mask seen so far (for a < b < c)
        # Actually, let's do: for each c, we need pairs (a,b) from [211, c-1] with mask_a ^ mask_b = needed
        
        # Precompute mask frequency for all numbers < c as we iterate c downward
        # But we want c to be as small as possible, so iterate c from 213 upward? No...
        # We want to find ANY valid triple with max < 210 + best_raw
        # Let's iterate c from upper_bound down to 213, and for each c check if a valid (a,b) exists
        
        # Build mask_to_list for numbers in [211, upper_bound]
        m2n = defaultdict(list)
        for i in range(211, upper_bound + 1):
            m2n[masks[i]].append(i)
        
        # For 3-element search: try each c, find (a,b) with a<b<c and mask_a^mask_b = target^mask_c
        # To make this fast, precompute for the range [211, c-1] a dict of mask -> list
        # But rebuilding for each c is O(n^2). Instead, iterate c from high to low,
        # and maintain a running mask_to_list for [211, c-1]
        
        # Alternative: just do a limited search. Since best_raw from phase 1 is likely small,
        # the range [211, 210+best_raw-1] might be small enough for O(n^2)
        
        if (upper_bound - 211) <= 200:
            # O(n^2) is feasible
            for a in range(211, upper_bound + 1):
                if time.time() > deadline - 20:
                    break
                for b in range(a + 1, upper_bound + 1):
                    if time.time() > deadline - 20:
                        break
                    needed = masks[a] ^ masks[b] ^ target_mask
                    if needed in m2n:
                        for c in m2n[needed]:
                            if c > b:
                                raw = c - 210
                                if raw < best_raw:
                                    if _verify_solution([a, b, c]):
                                        best_raw = raw
                                        best_solution = [a, b, c]
        
        elif (upper_bound - 211) <= 500:
            # Use smarter approach: for each c, check if needed mask exists in pairs
            # Precompute all pair masks for [211, upper_bound]
            # But that's O(n^2) memory... 
            # Instead: for each c, we need mask_a ^ mask_b = target ^ mask_c
            # Group numbers by mask. For a given needed_mask, we need two numbers 
            # (possibly same mask if mask^mask=0, but we need distinct numbers)
            # whose masks XOR to needed_mask.
            
            # Build mask -> sorted list of numbers
            # For each c, compute needed = target ^ mask_c
            # Find two masks m1, m2 such that m1 ^ m2 = needed
            # m1 can be any mask, m2 = m1 ^ needed
            # We need at least one number with mask m1 and one with mask m2, both < c
            
            # This is still complex. Let's just do the O(n^2) but with early termination
            found = False
            for c in range(213, upper_bound + 1):
                if time.time() > deadline - 20:
                    break
                needed = masks[c] ^ target_mask
                # Find a < b < c with mask_a ^ mask_b = needed
                # Use the fact that we can check mask pairs
                # Build a set of masks seen in [211, c-1]
                # For efficiency, precompute mask_list for [211, c-1]
                
                # Quick check: is there any mask m such that m and (m^needed) both have 
                # at least one (or two, if m == m^needed) numbers in [211, c-1]?
                # We need a<b, so if m == m^needed (i.e., needed==0), we need at least 2 numbers with mask m
                # Otherwise, we need at least 1 with mask m and 1 with mask (m^needed)
                
                # This requires knowing the counts, which changes as c changes
                # Let's just do the nested loop but break early
                for a in range(211, c):
                    if time.time() > deadline - 20:
                        break
                    for b in range(a + 1, c):
                        if time.time() > deadline - 20:
                            break
                        if masks[a] ^ masks[b] == needed:
                            raw = c - 210
                            if raw < best_raw:
                                if _verify_solution([a, b, c]):
                                    best_raw = raw
                                    best_solution = [a, b, c]
                                    found = True
                                    break
                    if found:
                        break
                if found:
                    break
    
    # Phase 3: 4-element search only if range is very small
    upper_bound4 = min(210 + best_raw - 1, limit)
    if upper_bound4 > 211 and (upper_bound4 - 211) <= 50:
        m2n4 = defaultdict(list)
        for i in range(211, upper_bound4 + 1):
            m2n4[masks[i]].append(i)
        
        for a in range(211, upper_bound4 + 1):
            if time.time() > deadline - 10:
                break
            for b in range(a + 1, upper_bound4 + 1):
                if time.time() > deadline - 10:
                    break
                for c in range(b + 1, upper_bound4 + 1):
                    if time.time() > deadline - 10:
                        break
                    needed = masks[a] ^ masks[b] ^ masks[c] ^ target_mask
                    if needed in m2n4:
                        for d in m2n4[needed]:
                            if d > c:
                                raw = d - 210
                                if raw < best_raw:
                                    if _verify_solution([a, b, c, d]):
                                        best_raw = raw
                                        best_solution = [a, b, c, d]
    
    # Fallback
    if best_solution is None:
        best_solution = [224, 240]
        best_raw = 30
    
    # Final verification
    if not _verify_solution(best_solution):
        best_solution = [224, 240]
    
    return [int(x) for x in best_solution]

# EVOLVE-BLOCK-END
