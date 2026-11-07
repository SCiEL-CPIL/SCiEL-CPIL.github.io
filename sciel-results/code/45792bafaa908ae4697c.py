"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from collections import defaultdict, deque

# EVOLVE-BLOCK-START
N = 210
MAX_VAL = 10210
MAX_SIZE = 100
DEADLINE = 88.0
SAFETY_MARGIN = 2.0

def squarefree_kernel(n):
    """Compute the squarefree kernel of n."""
    result = 1
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            count = 0
            while temp % d == 0:
                temp //= d
                count += 1
            if count % 2 == 1:
                result *= d
            d += 1
        else:
            d += 1
    if temp > 1:
        result *= temp
    return result

def is_perfect_square(n):
    """Check if n is a perfect square using integer square root."""
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN

    # Precompute squarefree kernels for all candidates in range (210, 10210]
    candidates = range(211, MAX_VAL + 1)

    # Group candidates by their squarefree kernel
    sf_cache = {}
    kernel_to_numbers = defaultdict(list)
    for i in candidates:
        k = squarefree_kernel(i)
        sf_cache[i] = k
        kernel_to_numbers[k].append(i)

    # For each kernel group, sort the numbers (they're already in increasing order
    # since we iterate i in increasing order, but let's be explicit)
    for k in kernel_to_numbers:
        kernel_to_numbers[k].sort()

    # Strategy 1: Find the best pair (a, b) such that sf(210 * a * b) == 1
    # This is equivalent to sf(210 * a) == sf(b) (when both are squarefree,
    # sf(x*y) == 1 iff sf(x) == sf(y))
    # More precisely: sf(210 * a * b) == 1 means 210*a*b is a perfect square.
    # Since 210 is squarefree, this means sf(210*a*b) = sf(210*a*b) must equal 1.
    # sf(210*a*b) = sf(sf(210*a) * sf(b))
    # For this to be 1, we need sf(210*a) == sf(b) (both squarefree).

    best_max = MAX_VAL
    best_pair = None

    # Iterate over possible max values M from 211 upward
    # For each M, the pool is {211, ..., M}
    # We want to find a, b in pool with a != b such that sf(210*a) == sf(b)
    # and minimize max(a, b) = M

    # Efficient approach: for each a in increasing order, compute target = sf(210*a)
    # and find the smallest b != a with sf(b) == target.
    # The max is max(a, b). We want to minimize this.

    sorted_candidates = sorted(candidates)

    # For efficiency, precompute target kernels
    # For each a, target = sf(210 * a)
    # We need b such that sf(b) == target and b != a, b in range.

    # Let's iterate a from 211 upward, and for each a, find the smallest valid b.
    # Track the best (min max(a,b)) pair found.

    for a in sorted_candidates:
        if time.time() > deadline:
            break
        target = squarefree_kernel(210 * a)
        if target in kernel_to_numbers:
            # Find the smallest b in kernel_to_numbers[target] that is different from a
            for b in kernel_to_numbers[target]:
                if b != a:
                    current_max = max(a, b)
                    if current_max < best_max:
                        best_max = current_max
                        best_pair = (a, b)
                    break  # We only need the smallest b for this a

    # Strategy 2: If no pair found, try a single number a such that 210*a is a square.
    # 210 = 2*3*5*7 (squarefree), so a must be of the form k^2 * 210.
    # Smallest a > 210: k=2 -> a = 4*210 = 840.
    if best_pair is None:
        a = 4 * 210  # 840
        if a <= MAX_VAL:
            best_max = a
            best_pair = (a,)

    # Strategy 3: If still no pair (unlikely), try triples or use the fallback.
    # But first, let's also try: for each b in increasing order, find smallest a != b
    # such that sf(210*a) == sf(b). This is symmetric but let's also check.
    # Actually, the loop above already covers this since we iterate all a.

    # Let's also try a different pairing: maybe we can find a pair where
    # a and b are both small. The loop above should find the best.

    # Verify the solution
    if best_pair is not None:
        if len(best_pair) == 2:
            a, b = best_pair
            product = 210 * a * b
            if not is_perfect_square(product):
                # Fallback to single number
                a = 4 * 210
                best_pair = (a,)
                best_max = a
        else:
            a = best_pair[0]
            product = 210 * a
            if not is_perfect_square(product):
                # Should not happen for a = 4*210, but fallback
                best_pair = (4 * 210,)
                best_max = 4 * 210

    # If best_pair is still None (shouldn't happen), use fallback
    if best_pair is None:
        best_pair = (4 * 210,)
        best_max = 4 * 210

    # Return the list of selected integers as Python ints
    result = [int(x) for x in best_pair]
    return result
# EVOLVE-BLOCK-END
