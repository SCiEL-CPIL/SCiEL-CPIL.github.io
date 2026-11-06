"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 82  # safety margin
    
    n = 30
    L = 1
    for i in range(1, n + 1):
        L = L * i // __import__('math').gcd(L, i)
    
    # Integer weights: w_i = L // i
    weights = [L // i for i in range(1, n + 1)]
    
    # The minimum non-zero |sum| is at least 1/L (since sum = integer / L)
    # So we need to find integer coefficients c_i in {-1,0,1} such that
    # |sum c_i * w_i| is minimized and non-zero.
    
    # Meet-in-the-middle: split into two halves
    half = n // 2  # 15
    weights_A = weights[:half]  # first 15
    weights_B = weights[half:]  # last 15
    
    # Generate all possible sums for half A (3^15 = 14348907)
    # For each combination, store the sum and the coefficients
    # We only care about the minimum absolute non-zero sum
    
    def gen_sums(weights_part):
        """Generate all possible sums and their coefficient tuples."""
        m = len(weights_part)
        results = {}  # sum -> list of coeff tuples (keep all to find exact match)
        
        # Use iterative approach
        current = [(0, ())]
        for w in weights_part:
            new_current = []
            for s, coeffs in current:
                new_current.append((s - w, coeffs + (-1,)))
                new_current.append((s, coeffs + (0,)))
                new_current.append((s + w, coeffs + (1,)))
            current = new_current
        return current
    
    # This is 3^15 which is ~14M, too many to store all with coeff tuples
    # Better: just store sums in a dict for lookup, and separately track min abs
    
    # Actually, for meet-in-the-middle, we need:
    # For each sum_a in A, we want sum_b such that sum_a + sum_b = k where k is small
    # So we build a dict for B: sum_b -> list of coeff tuples
    
    def gen_sums_dict(weights_part):
        """Generate dict: sum -> list of coeff tuples."""
        m = len(weights_part)
        current = [(0, ())]
        for w in weights_part:
            new_current = []
            for s, coeffs in current:
                new_current.append((s - w, coeffs + (-1,)))
                new_current.append((s, coeffs + (0,)))
                new_current.append((s + w, coeffs + (1,)))
            current = new_current
        
        d = {}
        for s, coeffs in current:
            if s not in d:
                d[s] = []
            d[s].append(coeffs)
        return d
    
    # Build dict for B
    if time.time() > deadline - 30:
        return [0] * 28 + [1, -1]
    
    sums_B_dict = gen_sums_dict(weights_B)
    
    if time.time() > deadline - 20:
        return [0] * 28 + [1, -1]
    
    # Now for A, iterate through all sums and for each, check if there's a B sum
    # that makes the total close to 0 (but non-zero)
    # We want to minimize |sum_a + sum_b| subject to sum_a + sum_b != 0
    
    best_abs = float('inf')
    best_coeffs = None
    
    # To be efficient, sort the keys of B and for each a, look for b closest to -a
    B_keys = sorted(sums_B_dict.keys())
    
    import bisect
    
    # Generate A sums one at a time (don't store all)
    def iterate_A_sums(weights_part):
        """Generator that yields (sum, coeffs) for all combinations."""
        m = len(weights_part)
        current = [(0, ())]
        for w in weights_part:
            new_current = []
            for s, coeffs in current:
                new_current.append((s - w, coeffs + (-1,)))
                new_current.append((s, coeffs + (0,)))
                new_current.append((s + w, coeffs + (1,)))
            current = new_current
        for s, coeffs in current:
            yield s, coeffs
    
    count = 0
    for a_sum, a_coeffs in iterate_A_sums(weights_A):
        count += 1
        if count % 1000000 == 0:
            if time.time() > deadline - 5:
                break
        
        # Find b closest to -a_sum
        target = -a_sum
        idx = bisect.bisect_left(B_keys, target)
        
        # Check idx-1, idx, and maybe idx+1
        for i in [idx - 1, idx, idx + 1]:
            if 0 <= i < len(B_keys):
                b_sum = B_keys[i]
                total = a_sum + b_sum
                if total == 0:
                    continue  # We need non-zero
                abs_total = abs(total)
                if abs_total < best_abs:
                    # Get a specific B coeff tuple
                    b_coeffs = sums_B_dict[b_sum][0]
                    full_coeffs = list(a_coeffs) + list(b_coeffs)
                    # Verify
                    val = sum(c * w for c, w in zip(full_coeffs, weights))
                    if val == total and total != 0:
                        best_abs = abs_total
                        best_coeffs = full_coeffs
                    if best_abs == 1:
                        # Can't do better than 1/L
                        break
        if best_abs == 1:
            break
    
    if best_coeffs is not None:
        # Verify with exact Fraction arithmetic
        s = Fraction(0, 1)
        for i, c in enumerate(best_coeffs):
            if c != 0:
                s += Fraction(c, i + 1)
        if s != 0:
            return best_coeffs
    
    # Fallback
    return [0] * 28 + [1, -1]
# EVOLVE-BLOCK-END
