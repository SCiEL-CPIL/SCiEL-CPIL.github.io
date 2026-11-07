"""Square-product extension for 210: find distinct ints in (210,10210]
whose product * 210 is a perfect square, minimizing max - 210."""

import math
import time

N = 210
MAX_N = 10210
MAX_K = 100
SMALL_PRIMES = [2, 3, 5, 7]


def _is_square(n):
    r = math.isqrt(n)
    return r * r == n


def _sieve_primes(limit):
    """Generate all primes up to limit."""
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]


def _factorize(n, primes):
    """Factor n using trial division by primes. Returns dict prime->exponent."""
    d = {}
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            d[p] = e
    if n > 1:
        d[n] = 1
    return d


def construct():
    t_start = time.time()
    DEADLINE = t_start + 70.0  # safety margin under 90s kill

    # Generate primes up to MAX_N for trial division
    primes = _sieve_primes(MAX_N)
    prime_set = set(primes)

    # Precompute parity vectors for all candidates 211..MAX_N
    # parity: (small_mask: 4 bits for 2,3,5,7; frozenset of large primes with odd exponent)
    parity = {}
    for n in range(211, MAX_N + 1):
        d = _factorize(n, primes)
        mask = 0
        for i, p in enumerate(SMALL_PRIMES):
            if d.get(p, 0) & 1:
                mask |= (1 << i)
        large = frozenset(p for p, e in d.items() if p not in SMALL_PRIMES and e & 1)
        parity[n] = (mask, large)

    # Target: 210 = 2^1 * 3^1 * 5^1 * 7^1, so we need XOR of all selected = (0b1111, frozenset())
    TARGET_MASK = 0b1111
    TARGET_LARGE = frozenset()

    # --- Strategy: find a small set (ideally 2-4 elements) that satisfies the condition ---
    # For 2 elements a, b: parity(a) XOR parity(b) must equal target
    # i.e., mask_a XOR mask_b == TARGET_MASK and large_a XOR large_b == TARGET_LARGE
    # So large_a == large_b (they must be the same set of large primes)
    # and mask_a XOR mask_b == 0b1111

    # Group candidates by their large-prime set
    from collections import defaultdict
    by_large = defaultdict(list)
    for n in range(211, MAX_N + 1):
        m, lg = parity[n]
        by_large[lg].append((m, n))

    # For each group with the same large set, find pairs where masks XOR to TARGET_MASK
    # This gives us 2-element solutions
    best_val = None
    best_set = None

    for lg, lst in by_large.items():
        if time.time() > DEADLINE:
            break
        # For each mask, we need another element with mask XOR TARGET_MASK in the same group
        # Build dict: mask -> list of numbers
        mask_to_nums = defaultdict(list)
        for m, n in lst:
            mask_to_nums[m].append(n)

        for m, nums in mask_to_nums.items():
            if time.time() > DEADLINE:
                break
            target_m = m ^ TARGET_MASK
            if target_m in mask_to_nums:
                # Find the pair with minimum max
                for a in nums:
                    for b in mask_to_nums[target_m]:
                        if a != b:
                            val = max(a, b) - 210
                            if best_val is None or val < best_val:
                                best_val = val
                                best_set = (a, b)
                # Only need the best pair from this mask pair
                break  # move to next group

    # Also try single element: parity(n) must equal target exactly
    # mask == 0b1111 and large == frozenset()
    for n in range(211, MAX_N + 1):
        if time.time() > DEADLINE:
            break
        m, lg = parity[n]
        if m == TARGET_MASK and lg == TARGET_LARGE:
            val = n - 210
            if best_val is None or val < best_val:
                best_val = val
                best_set = (n,)

    # If no 1 or 2 element solution found, try 3 elements
    # a XOR b XOR c = target
    # This is harder; skip for now and use a greedy approach as fallback
    if best_set is None:
        # Greedy: pick smallest elements that reduce the "distance" to target
        # Represent current XOR state
        cur_mask = 0
        cur_large = frozenset()
        selected = []

        # Distance = popcount(cur_mask XOR TARGET_MASK) + len(cur_large XOR TARGET_LARGE)
        def _dist(cm, cl):
            return bin(cm ^ TARGET_MASK).count('1') + len(cl ^ TARGET_LARGE)

        # Try to build a solution greedily
        for n in range(211, MAX_N + 1):
            if time.time() > DEADLINE:
                break
            if len(selected) >= MAX_K:
                break
            m, lg = parity[n]
            new_mask = cur_mask ^ m
            new_large = cur_large ^ lg
            if _dist(new_mask, new_large) < _dist(cur_mask, cur_large):
                cur_mask = new_mask
                cur_large = new_large
                selected.append(n)

        if _dist(cur_mask, cur_large) == 0 and selected:
            best_set = tuple(selected)
            best_val = max(selected) - 210

    # Validate the best solution
    if best_set is not None:
        # Verify
        prod = N
        for x in best_set:
            prod *= x
        if _is_square(prod) and len(set(best_set)) == len(best_set) and all(
            isinstance(x, int) and not isinstance(x, bool) and 210 < x <= MAX_N for x in best_set
        ):
            return [int(x) for x in best_set]

    # Fallback: 840 = 4*210, so 210*840 = 210^2*4 = (210*2)^2 is a perfect square
    return [840]


# Quick self-test
if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Result: {result}")
    print(f"Max - 210: {max(result) - 210}")
    prod = 210
    for x in result:
        prod *= x
    print(f"Product is square: {_is_square(prod)}")
    print(f"Time: {t1 - t0:.2f}s")
