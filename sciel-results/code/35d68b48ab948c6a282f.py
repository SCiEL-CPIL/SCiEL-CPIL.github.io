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

def _gaussian_elimination_rows(vectors, target_mask, num_bits):
    """
    Given a list of (mask, index) pairs, use Gaussian elimination to find
    a subset whose XOR equals target_mask. Returns the list of indices or None.
    """
    # Build row-reduced echelon form
    # rows: list of (mask, set_of_original_indices)
    rows = []
    pivot_cols = {}  # col -> row index
    
    for idx, mask in enumerate(vectors):
        if mask == 0:
            continue
        m = mask
        orig_indices = {idx}
        # Eliminate using existing pivots
        while m != 0:
            # Find highest set bit
            hsb = m.bit_length() - 1
            if hsb in pivot_cols:
                row = rows[pivot_cols[hsb]]
                m ^= row[0]
                orig_indices = orig_indices ^ row[1]
            else:
                break
        
        if m == 0:
            continue
        
        hsb = m.bit_length() - 1
        pivot_cols[hsb] = len(rows)
        rows.append((m, orig_indices))
    
    # Now try to represent target_mask
    if target_mask == 0:
        return []
    
    m = target_mask
    result_indices = set()
    while m != 0:
        hsb = m.bit_length() - 1
        if hsb not in pivot_cols:
            return None
        row = rows[pivot_cols[hsb]]
        m ^= row[0]
        result_indices = result_indices ^ row[1]
    
    return list(result_indices)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Generate primes up to a reasonable bound
    max_search = 300
    primes = _sieve(max_search)
    prime_to_bit = {p: i for i, p in enumerate(primes)}
    num_bits = len(primes)
    
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
    
    # Incremental Gaussian elimination: add numbers one by one,
    # check when target_mask becomes representable
    # We want the smallest M such that target is in span of masks[211..M]
    
    # Build row-reduced echelon form incrementally
    rows = []
    pivot_cols = {}  # col -> row index
    
    best_raw = 10000
    best_solution = [224, 240]  # Known good: raw=30
    
    # We need to track which original numbers contributed to each row
    # rows[i] = (mask, frozenset of original number indices)
    
    for M in range(211, max_search + 1):
        if time.time() > deadline - 10:
            break
        
        mask = masks[M]
        
        # Try to add this vector to the row space
        m = mask
        orig_set = frozenset([M])
        
        while m != 0:
            hsb = m.bit_length() - 1
            if hsb in pivot_cols:
                row = rows[pivot_cols[hsb]]
                m ^= row[0]
                orig_set = orig_set ^ row[1]
            else:
                break
        
        if m != 0:
            hsb = m.bit_length() - 1
            pivot_cols[hsb] = len(rows)
            rows.append((m, orig_set))
        
        # Check if target_mask is now representable
        m2 = target_mask
        result_set = frozenset()
        representable = True
        while m2 != 0:
            hsb = m2.bit_length() - 1
            if hsb not in pivot_cols:
                representable = False
                break
            row = rows[pivot_cols[hsb]]
            m2 ^= row[0]
            result_set = result_set ^ row[1]
        
        if representable:
            # We found a solution with max element M
            selected = sorted(result_set)
            raw = M - 210
            if raw < best_raw:
                if len(selected) <= 100:
                    if _verify_solution(selected):
                        best_raw = raw
                        best_solution = selected
                        # Don't break - there might be a better one? No, M is increasing
                        # Actually M is the maximum, so this is the best for this M
                        # But we can stop since we're looking for minimum M
                        break
    
    # If we didn't find anything in the incremental search, fall back
    if best_raw >= 10000:
        best_solution = [224, 240]
        best_raw = 30
    
    # Ensure types
    result = [int(x) for x in best_solution]
    return result

# EVOLVE-BLOCK-END
