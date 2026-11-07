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

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Generate primes up to a reasonable bound
    def sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit+1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit+1) if is_prime[i]]
    
    # We'll search up to max_search; primes needed up to that value
    max_search = 260
    primes = sieve(max_search)
    prime_to_bit = {p: i for i, p in enumerate(primes)}
    
    # Target: sf(210) = 210 = 2*3*5*7
    # Target mask: bits for primes 2,3,5,7
    target_mask = 0
    for p in [2, 3, 5, 7]:
        if p in prime_to_bit:
            target_mask |= (1 << prime_to_bit[p])
    
    # Precompute parity masks for numbers 211..max_search
    masks = {}
    for i in range(211, max_search + 1):
        masks[i] = _prime_parity_mask(i, prime_to_bit)
    
    best_raw = 10000
    best_solution = [224, 240]  # Known good: raw=30
    
    # Phase 1: Search for 2-element solutions with max < 240 (raw < 30)
    # Need mask[a] ^ mask[b] == target_mask
    # i.e., mask[b] == mask[a] ^ target_mask
    
    mask_to_numbers = defaultdict(list)
    for i in range(211, max_search + 1):
        mask_to_numbers[masks[i]].append(i)
    
    for a in range(211, max_search + 1):
        if time.time() > deadline - 10:
            break
        needed_mask = masks[a] ^ target_mask
        if needed_mask in mask_to_numbers:
            for b in mask_to_numbers[needed_mask]:
                if b > a:
                    if _verify_solution([a, b]):
                        raw = max(a, b) - 210
                        if raw < best_raw:
                            best_raw = raw
                            best_solution = [a, b]
    
    # Phase 2: Search for 3-element solutions with smaller max
    # Need mask[a] ^ mask[b] ^ mask[c] == target_mask
    # i.e., mask[c] == mask[a] ^ mask[b] ^ target_mask
    upper_bound = min(210 + best_raw - 1, max_search)
    
    # Rebuild mask_to_numbers for the reduced range
    mask_to_numbers2 = defaultdict(list)
    for i in range(211, upper_bound + 1):
        mask_to_numbers2[masks[i]].append(i)
    
    for a in range(211, upper_bound + 1):
        if time.time() > deadline - 5:
            break
        for b in range(a + 1, upper_bound + 1):
            if time.time() > deadline - 5:
                break
            needed_mask = masks[a] ^ masks[b] ^ target_mask
            if needed_mask in mask_to_numbers2:
                for c in mask_to_numbers2[needed_mask]:
                    if c > b:
                        if _verify_solution([a, b, c]):
                            raw = max(a, b, c) - 210
                            if raw < best_raw:
                                best_raw = raw
                                best_solution = [a, b, c]
    
    # Phase 3: Search for 4-element solutions with even smaller max
    upper_bound4 = min(210 + best_raw - 1, max_search)
    mask_to_numbers3 = defaultdict(list)
    for i in range(211, upper_bound4 + 1):
        mask_to_numbers3[masks[i]].append(i)
    
    # For 4 elements: mask[a]^mask[b]^mask[c]^mask[d] == target_mask
    # mask[d] == mask[a]^mask[b]^mask[c]^target_mask
    if upper_bound4 - 211 < 50:  # Only do this if range is small
        for a in range(211, upper_bound4 + 1):
            if time.time() > deadline - 3:
                break
            for b in range(a + 1, upper_bound4 + 1):
                if time.time() > deadline - 3:
                    break
                for c in range(b + 1, upper_bound4 + 1):
                    if time.time() > deadline - 3:
                        break
                    needed_mask = masks[a] ^ masks[b] ^ masks[c] ^ target_mask
                    if needed_mask in mask_to_numbers3:
                        for d in mask_to_numbers3[needed_mask]:
                            if d > c:
                                if _verify_solution([a, b, c, d]):
                                    raw = max(a, b, c, d) - 210
                                    if raw < best_raw:
                                        best_raw = raw
                                        best_solution = [a, b, c, d]
    
    # Phase 4: If we found something better, try to extend search for even better
    remaining_time = deadline - time.time()
    if remaining_time > 5 and best_raw < 30:
        # Try to find solutions with raw < best_raw using more numbers
        new_upper = min(210 + best_raw - 1, max_search)
        if new_upper < max_search:
            # Extend masks if needed (already computed up to max_search)
            mask_to_numbers4 = defaultdict(list)
            for i in range(211, new_upper + 1):
                mask_to_numbers4[masks[i]].append(i)
            
            # 2-element
            for a in range(211, new_upper + 1):
                if time.time() > deadline - 2:
                    break
                needed_mask = masks[a] ^ target_mask
                if needed_mask in mask_to_numbers4:
                    for b in mask_to_numbers4[needed_mask]:
                        if b > a:
                            if _verify_solution([a, b]):
                                raw = max(a, b) - 210
                                if raw < best_raw:
                                    best_raw = raw
                                    best_solution = [a, b]
            
            # 3-element
            for a in range(211, new_upper + 1):
                if time.time() > deadline - 2:
                    break
                for b in range(a + 1, new_upper + 1):
                    if time.time() > deadline - 2:
                        break
                    needed_mask = masks[a] ^ masks[b] ^ target_mask
                    if needed_mask in mask_to_numbers4:
                        for c in mask_to_numbers4[needed_mask]:
                            if c > b:
                                if _verify_solution([a, b, c]):
                                    raw = max(a, b, c) - 210
                                    if raw < best_raw:
                                        best_raw = raw
                                        best_solution = [a, b, c]
    
    # Final verification
    if not _verify_solution(best_solution):
        best_solution = [224, 240]
        best_raw = 30
    
    # Ensure types
    result = [int(x) for x in best_solution]
    return result

# EVOLVE-BLOCK-END
