"""Meet-in-the-middle for small nonzero signed harmonic subsums."""
import time
import sys
import bisect
from math import gcd

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 75  # safety margin before 90s kill
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # LCM(1..15)
    lcm15 = 1
    for i in range(1, 16):
        lcm15 = lcm(lcm15, i)
    
    # LCM(16..30)
    lcm16_30 = 1
    for i in range(16, 31):
        lcm16_30 = lcm(lcm16_30, i)
    
    lcm_total = lcm(lcm15, lcm16_30)
    
    def gen_half_ints(start, count, lcm_val):
        """Generate (int_sum, base3_bits) where actual sum = int_sum / lcm_val.
        Terms are start, start+1, ..., start+count-1.
        base3_bits: digit 0=-1, 1=0, 2=+1, with first term as most significant."""
        results = [(0, 0)]
        for i in range(start, start + count):
            inv_int = lcm_val // i
            new_results = []
            for s, bits in results:
                new_results.append((s - inv_int, bits * 3 + 0))
                new_results.append((s, bits * 3 + 1))
                new_results.append((s + inv_int, bits * 3 + 2))
            results = new_results
        return results
    
    t0 = time.time()
    # Generate first half (terms 1..15)
    half_a = gen_half_ints(1, 15, lcm15)
    t1 = time.time()
    print(f"Half A: {len(half_a)} entries, {t1-t0:.1f}s", file=sys.stderr)
    
    if time.time() > deadline - 30:
        return [0]*28 + [1, -1]
    
    # Sort by integer sum
    half_a.sort(key=lambda x: x[0])
    sa_ints = [x[0] for x in half_a]
    t2 = time.time()
    print(f"Sorted A: {t2-t1:.1f}s", file=sys.stderr)
    
    if time.time() > deadline - 25:
        return [0]*28 + [1, -1]
    
    # Generate second half (terms 16..30)
    half_b = gen_half_ints(16, 15, lcm16_30)
    t3 = time.time()
    print(f"Half B: {len(half_b)} entries, {t3-t2:.1f}s", file=sys.stderr)
    
    if time.time() > deadline - 15:
        return [0]*28 + [1, -1]
    
    # For each sb in half_b, find sa in half_a minimizing |sa*lcm16_30 + sb*lcm15|
    # target sa = -sb * lcm15 / lcm16_30
    best_total = None
    best_sa_idx = None
    best_sb_idx = None
    
    sa_len = len(sa_ints)
    max_b = len(half_b)
    batch_size = 500000
    
    processed = 0
    while processed < max_b:
        end = min(processed + batch_size, max_b)
        for idx in range(processed, end):
            sb_int, sb_bits = half_b[idx]
            
            # target sa as float
            target = -sb_int * lcm15 / lcm16_30
            
            pos = bisect.bisect_left(sa_ints, target)
            
            for p in [pos, pos-1]:
                if 0 <= p < sa_len:
                    sa_int = sa_ints[p]
                    total_num = sa_int * lcm16_30 + sb_int * lcm15
                    if total_num == 0:
                        continue
                    abs_t = abs(total_num)
                    if best_total is None or abs_t < best_total:
                        best_total = abs_t
                        best_sa_idx = p
                        best_sb_idx = idx
        
        processed = end
        if time.time() > deadline - 5:
            break
    
    if best_total is None:
        return [0]*28 + [1, -1]
    
    # Reconstruct coefficients
    a_bits = half_a[best_sa_idx][1]
    b_bits = half_b[best_sb_idx][1]
    
    coeffs = [0] * 30
    
    # Half A: terms 1..15, bits encoded with term 1 as most significant
    for i in range(1, 16):
        digit = (a_bits // (3 ** (15 - i))) % 3
        coeffs[i - 1] = digit - 1
    
    # Half B: terms 16..30, bits encoded with term 16 as most significant
    for i in range(16, 31):
        local_i = i - 16
        digit = (b_bits // (3 ** (14 - local_i))) % 3
        coeffs[i - 1] = digit - 1
    
    # Verify
    from fractions import Fraction
    total = Fraction(0)
    for i in range(30):
        if coeffs[i] != 0:
            total += Fraction(coeffs[i], i + 1)
    
    if total == 0:
        return [0]*28 + [1, -1]
    
    print(f"Result: {total} = {total.numerator}/{total.denominator}", file=sys.stderr)
    return coeffs

# EVOLVE-BLOCK-END
