import math
import time
from collections import defaultdict
from bisect import bisect_left

N = 210
MAX_N = 10210
SMALL_PRIMES = [2, 3, 5, 7]

def _is_square(n):
    r = math.isqrt(n)
    return r * r == n

def _sieve_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]

def construct():
    t_start = time.time()
    DEADLINE = t_start + 80.0

    def elapsed():
        return time.time() - t_start

    # Precompute squarefree part for every number up to MAX_N
    sf = list(range(MAX_N + 1))
    for i in range(2, int(MAX_N ** 0.5) + 1):
        i2 = i * i
        for j in range(i2, MAX_N + 1, i2):
            sf[j] //= i2

    # Precompute parity signature for each candidate
    # Signature: (mask4, frozenset_of_large_odd_exponents)
    # where mask4 captures parity of exponents of 2,3,5,7
    # and the frozenset captures primes > 7 with odd exponent
    parity = {}
    for n in range(211, MAX_N + 1):
        sfv = sf[n]
        # Extract parity of 2,3,5,7 from sfv
        mask = 0
        temp = sfv
        for i, p in enumerate(SMALL_PRIMES):
            if temp % p == 0:
                mask |= (1 << i)
                temp //= p
        # Remaining primes > 7 with odd exponent (sfv is squarefree, so all exponents are 0 or 1)
        large = frozenset(temp) if temp > 1 else frozenset()
        parity[n] = (mask, large)

    TARGET_MASK = 0b1111  # 210 = 2*3*5*7, all exponents odd
    TARGET_LARGE = frozenset()

    # Build lookup: signature -> sorted list of numbers with that signature
    parity_dict = defaultdict(list)
    for n in range(211, MAX_N + 1):
        sig = parity[n]
        parity_dict[sig].append(n)
    for key in parity_dict:
        parity_dict[key].sort()

    best_val = None
    best_set = None

    def try_update(val, s):
        nonlocal best_val, best_set
        if best_val is None or val < best_val:
            best_val = val
            best_set = tuple(s)

    # k=1: check if any single number has target signature
    for n in range(211, MAX_N + 1):
        if elapsed() > DEADLINE:
            break
        if parity[n] == (TARGET_MASK, TARGET_LARGE):
            try_update(n - 210, (n,))
            break  # first one is smallest

    # k=2: for each a, find b with complementary signature
    # Use binary search on the answer to prune
    # First pass: find best k=2 solution
    for a in range(211, MAX_N + 1):
        if elapsed() > DEADLINE:
            break
        if best_val is not None and a - 210 >= best_val:
            break
        ma, la = parity[a]
        target_m = TARGET_MASK ^ ma
        target_l = TARGET_LARGE ^ la
        target_key = (target_m, target_l)
        if target_key in parity_dict:
            nums = parity_dict[target_key]
            idx = bisect_left(nums, a + 1)
            if idx < len(nums):
                b = nums[idx]
                val = b - 210
                try_update(val, (a, b))

    # Now we have a best_val. Use it to prune further.
    # k=3: for each a, for each b > a, check if c exists with target signature
    # Prune: a, b, c all < 210 + best_val
    if best_val is not None:
        limit3 = 210 + best_val - 1  # strict: max must be < 210 + best_val
    else:
        limit3 = MAX_N

    # Precompute for each signature the list of numbers <= limit3
    # Actually, we'll just use bisect on the sorted lists
    for a in range(211, min(limit3, MAX_N) + 1):
        if elapsed() > DEADLINE:
            break
        if a - 210 >= best_val:
            break
        ma, la = parity[a]
        # For b, we need b > a and b < 210 + best_val
        b_limit = min(limit3, MAX_N)
        for b in range(a + 1, b_limit + 1):
            if elapsed() > DEADLINE:
                break
            if b - 210 >= best_val:
                break
            mb, lb = parity[b]
            target_m = TARGET_MASK ^ ma ^ mb
            target_l = TARGET_LARGE ^ la ^ lb
            target_key = (target_m, target_l)
            if target_key in parity_dict:
                nums = parity_dict[target_key]
                idx = bisect_left(nums, b + 1)
                if idx < len(nums):
                    c = nums[idx]
                    if c < 210 + best_val:
                        val = c - 210
                        try_update(val, (a, b, c))

    # k=4: pair-pair matching
    # Build pair signatures for pairs within the current limit
    if best_val is not None:
        limit4 = 210 + best_val - 1
    else:
        limit4 = MAX_N

    # Build pair_dict: combined_signature -> list of (max_val, a, b)
    pair_dict = defaultdict(list)
    for a in range(211, min(limit4, MAX_N) + 1):
        if elapsed() > DEADLINE:
            break
        if a - 210 >= best_val:
            break
        ma, la = parity[a]
        for b in range(a + 1, min(limit4, MAX_N) + 1):
            if elapsed() > DEADLINE:
                break
            if b - 210 >= best_val:
                break
            mb, lb = parity[b]
            pm = ma ^ mb
            pl = la ^ lb
            pair_dict[(pm, pl)].append((b, a, b))  # max is b since b > a

    # For each pair signature, find complement
    for key, pairs in list(pair_dict.items()):
        if elapsed() > DEADLINE:
            break
        pm, pl = key
        comp_m = TARGET_MASK ^ pm
        comp_l = TARGET_LARGE ^ pl
        comp_key = (comp_m, comp_l)
        if comp_key not in pair_dict:
            continue
        comp_pairs = pair_dict[comp_key]
        for max1, a, b in pairs:
            if elapsed() > DEADLINE:
                break
            if max1 - 210 >= best_val:
                continue
            for max2, c, d in comp_pairs:
                if elapsed() > DEADLINE:
                    break
                overall_max = max(max1, max2)
                if overall_max - 210 >= best_val:
                    continue
                if len({a, b, c, d}) == 4:
                    val = overall_max - 210
                    try_update(val, (a, b, c, d))

    # Verify and return
    if best_set is not None:
        prod = N
        for x in best_set:
            prod *= x
        if (_is_square(prod)
                and len(set(best_set)) == len(best_set)
                and all(isinstance(x, int) and not isinstance(x, bool) and 210 < x <= MAX_N for x in best_set)):
            return [int(x) for x in best_set]

    # Fallback: known valid solution
    return [840]

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
