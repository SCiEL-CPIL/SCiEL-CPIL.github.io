import time
import random
from math import gcd
from itertools import combinations

N = 128
DEADLINE_SAFETY = 82.0  # stay well under 90s

# ---------- Precomputation ----------
_LCM = [[0] * (N + 1) for _ in range(N + 1)]
for _a in range(1, N + 1):
    for _b in range(_a, N + 1):
        _v = _a * _b // gcd(_a, _b)
        _LCM[_a][_b] = _v
        _LCM[_b][_a] = _v

# conflict graph: x,y conflict if some z makes lcm equal on all three pairs
CONFLICT = [0] * (N + 1)  # bitmask
for _x in range(1, N + 1):
    for _y in range(_x + 1, N + 1):
        lx = _LCM[_x][_y]
        # z must divide both x and y (since lcm(x,z)|lcm(x,y) and lcm(y,z)|lcm(x,y)
        # and lcm(x,z) must equal lx which is multiple of x -> z divides x; same for y)
        # enumerate common divisors of x and y
        g = gcd(_x, _y)
        for _z in range(1, N + 1):
            if _z == _x or _z == _y:
                continue
            if _x % _z == 0 and _y % _z == 0:
                if _LCM[_x][_z] == lx and _LCM[_y][_z] == lx:
                    CONFLICT[_x] |= 1 << _y
                    CONFLICT[_y] |= 1 << _x
                    break

# For each element, the set of elements that conflict with it
# (bitmask already computed)

# ---------- Greedy with bitmask ----------
def _greedy(order, mask=0):
    for x in order:
        bit = 1 << x
        if mask & bit:
            continue
        if (mask & CONFLICT[x]) == 0:
            mask |= bit
    return mask

def _mask_to_set(mask):
    return [x for x in range(1, N + 1) if (mask >> x) & 1]

# ---------- Local search (add + swap) ----------
def _local_search(mask, deadline, max_rounds=200):
    size = bin(mask).count('1')
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        # try adding
        for x in range(1, N + 1):
            bit = 1 << x
            if (mask & bit) == 0 and (mask & CONFLICT[x]) == 0:
                mask |= bit
                size += 1
                improved = True
                break
        if improved:
            continue
        # try swap: remove one, add another
        for a in range(1, N + 1):
            abit = 1 << a
            if (mask & abit) == 0:
                continue
            without = mask & ~abit
            for x in range(1, N + 1):
                bit = 1 << x
                if (without & bit) == 0 and (without & CONFLICT[x]) == 0:
                    mask = without | bit
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return mask, bin(mask).count('1')

# ---------- Orderings ----------
def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def _num_divisors(n):
    c = 0
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            c += 1
            if i != n // i:
                c += 1
    return c

def _max_prime_factor(n):
    if n < 2:
        return 1
    mpf = 1
    d = 2
    while d * d <= n:
        while n % d == 0:
            mpf = d
            n //= d
        d += 1
    if n > 1:
        mpf = max(mpf, n)
    return mpf

def _radical(n):
    r = 1
    d = 2
    while d * d <= n:
        if n % d == 0:
            r *= d
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        r *= n
    return r

def _build_orderings(rng):
    all_el = list(range(1, N + 1))
    orderings = []

    # 1. Primes first
    primes = [x for x in all_el if _is_prime(x)]
    non_primes = [x for x in all_el if not _is_prime(x)]
    orderings.append(primes + non_primes)

    # 2. Fewer divisors first
    orderings.append(sorted(all_el, key=lambda x: _num_divisors(x)))

    # 3. Reverse
    orderings.append(list(reversed(all_el)))

    # 4. Larger max prime factor first
    orderings.append(sorted(all_el, key=lambda x: -_max_prime_factor(x)))

    # 5. Smaller radical first (fewer distinct primes)
    orderings.append(sorted(all_el, key=lambda x: _radical(x)))

    # 6. Larger radical first
    orderings.append(sorted(all_el, key=lambda x: -_radical(x)))

    # 7-14. Random
    for _ in range(8):
        orderings.append(rng.sample(all_el, N))

    # 15-18. Greedy-conflict-based: sort by conflict-degree
    deg = [bin(CONFLICT[x]).count('1') for x in range(N + 1)]
    orderings.append(sorted(all_el, key=lambda x: deg[x]))       # low degree first
    orderings.append(sorted(all_el, key=lambda x: -deg[x]))      # high degree first

    # 19. Powers of 2 first (these rarely conflict)
    pows2 = [2**k for k in range(1, 8) if 2**k <= N]
    rest = [x for x in all_el if x not in pows2]
    orderings.append(pows2 + rest)

    return orderings

def construct():
    start = time.time()
    deadline = start + DEADLINE_SAFETY
    rng = random.Random(12345)

    best_mask = 0
    best_size = 0

    # Phase 1: many greedy runs with diverse orderings
    orderings = _build_orderings(rng)
    for order in orderings:
        if time.time() > deadline:
            break
        m = _greedy(order)
        s = bin(m).count('1')
        if s > best_size:
            best_size = s
            best_mask = m

    # Phase 2: local search polish on best
    if time.time() < deadline:
        m2, s2 = _local_search(best_mask, deadline)
        if s2 > best_size:
            best_size = s2
            best_mask = m2

    # Phase 3: multi-start local search with random perturbations
    for _ in range(40):
        if time.time() > deadline:
            break
        # start from a fresh random-order greedy
        order = rng.sample(list(range(1, N + 1)), N)
        m = _greedy(order)
        # perturb: remove 1-3 random elements
        lst = _mask_to_set(m)
        if lst:
            for _ in range(rng.randint(1, 3)):
                if lst:
                    lst.remove(rng.choice(lst))
            m = 0
            for x in lst:
                m |= 1 << x
        m2, s2 = _local_search(m, deadline, max_rounds=100)
        if s2 > best_size:
            best_size = s2
            best_mask = m2

    # Phase 4: final local search on the best
    if time.time() < deadline:
        m2, s2 = _local_search(best_mask, deadline, max_rounds=500)
        if s2 > best_size:
            best_size = s2
            best_mask = m2

    result = _mask_to_set(best_mask)
    if not result:
        result = [1]
    return result
