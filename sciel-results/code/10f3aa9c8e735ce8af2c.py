"""Combined approach: generate powerful numbers, then count pairwise sums to find N with most representations."""
import time
from math import isqrt
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 105  # 15s buffer for 120s limit

    # Generate powerful numbers up to L
    # A powerful number is of the form a^2 * b^3 where b is squarefree
    # We iterate over all b (not just squarefree) since a^2 * b^3 for any b
    # will be powerful if we ensure the exponents work out. Actually, the standard
    # generation is: for each b, for each a, n = a^2 * b^3. This covers all powerful numbers
    # because if n is powerful, n = (product p_i^(e_i)) where e_i >= 2.
    # Write e_i = 2*f_i + g_i with g_i in {0,1}. Then n = (product p_i^f_i)^2 * (product p_i^g_i)^3
    # The second factor is a cube of a squarefree number. But if we allow b to be any integer,
    # we'll generate duplicates but that's fine - we use a set.
    L = 10**8
    powerfuls = set()
    max_b = int(L ** (1/3)) + 1
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > L:
            break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                powerfuls.add(n)

    powerfuls = sorted(powerfuls)
    n_pows = len(powerfuls)

    # Count sums of pairs (a, b) with a <= b
    sum_count = defaultdict(int)
    best_N = 0
    best_count = 0

    for i in range(n_pows):
        a = powerfuls[i]
        if time.time() > deadline - 5:
            break
        for j in range(i, n_pows):
            b_val = powerfuls[j]
            s = a + b_val
            # All sums are <= 2*L = 2*10^8 << 10^18, so all valid
            c = sum_count.get(s, 0) + 1
            sum_count[s] = c
            if c > best_count:
                best_count = c
                best_N = s

    # Find best N (if we broke early, best_N might not be the global max)
    if best_count == 0:
        return {'N': 2, 'pairs': [[1, 1]]}

    # Re-check for the true best in case we broke early
    # Actually, if we broke early, we might have missed some. Let's just use what we have.
    # The best_N tracked during the loop is the max seen so far.
    
    # Collect pairs for best_N
    pows_set = set(powerfuls)
    pairs = []
    for a in powerfuls:
        if 2 * a > best_N:
            break
        b_val = best_N - a
        if b_val in pows_set and b_val >= a:
            pairs.append([a, b_val])

    return {'N': best_N, 'pairs': pairs}
