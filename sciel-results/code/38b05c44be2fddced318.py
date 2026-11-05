"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
# Prime exponent vectors over primes up to 128. A consecutive-block product is
# uniquely identified by the sum of the exponent vectors of its members, so
# distinctness of all block products == distinctness of these sums.
_PRIMES = []
for _n in range(2, 129):
    _d = _n
    _p = 2
    while _p * _p <= _d:
        if _d % _p == 0:
            _PRIMES.append(_p)
            while _d % _p == 0:
                _d //= _p
        _p += 1
    if _d > 1:
        _PRIMES.append(_d)
_PRIMES = sorted(set(_PRIMES))

_EXP = {}
_NUM_PRIMES = len(_PRIMES)
for _x in range(2, 129):
    _d = _x
    _vec = [0] * _NUM_PRIMES
    for _idx, _p in enumerate(_PRIMES):
        if _p * _p > _d:
            break
        _c = 0
        while _d % _p == 0:
            _d //= _p
            _c += 1
        _vec[_idx] = _c
    if _d > 1:
        # _d is a prime factor larger than sqrt(original _x), so it must be in _PRIMES
        # Find its index
        for _idx, _p in enumerate(_PRIMES):
            if _p == _d:
                _vec[_idx] += 1
                break
    _EXP[_x] = tuple(_vec)


def _all_products(values):
    """Return set of all nonempty consecutive block products (integer products)."""
    n = len(values)
    prods = set()
    for i in range(n):
        prod = 1
        for j in range(i, n):
            prod *= values[j]
            prods.add(prod)
    return prods


def _can_add(values, products, x):
    """Check if x can be appended to values (strictly increasing, no product collision)."""
    if values and x <= values[-1]:
        return False
    tails = [x]
    prod = x
    for a in reversed(values):
        prod *= a
        tails.append(prod)
    if products.intersection(tails):
        return False
    return True


def _greedy_pass(values, products, start_from=2):
    """Continue greedy from current state."""
    values = list(values)
    products = set(products)
    for x in range(start_from, 129):
        if _can_add(values, products, x):
            tails = [x]
            prod = x
            for a in reversed(values):
                prod *= a
                tails.append(prod)
            values.append(x)
            products.update(tails)
    return values, products


def _build_prefix(vecs):
    """Build prefix sum tuples for fast block-sum extraction."""
    n = len(vecs)
    if n == 0:
        return []
    num_primes = len(vecs[0])
    P = [None] * (n + 1)
    P[0] = [0] * num_primes
    for i in range(n):
        prev = P[i]
        curr = vecs[i]
        P[i + 1] = [prev[k] + curr[k] for k in range(num_primes)]
    # Convert to tuples for set operations
    return [tuple(p) for p in P]


def _count_collisions(prefix):
    """Count number of duplicate block sums given prefix sums."""
    n = len(prefix) - 1
    if n <= 0:
        return 0
    seen = set()
    collisions = 0
    for i in range(n):
        pi = prefix[i]
        for j in range(i, n):
            pj1 = prefix[j + 1]
            s = tuple(pj1[k] - pi[k] for k in range(len(pi)))
            if s in seen:
                collisions += 1
            else:
                seen.add(s)
    return collisions


def _is_valid_vectors(vecs):
    """Check if all nonempty consecutive block sums of exponent vectors are distinct."""
    n = len(vecs)
    if n == 0:
        return True
    prefix = _build_prefix(vecs)
    return _count_collisions(prefix) == 0


def _local_search(values, deadline):
    """Local search using vector sums for fast validity checks."""
    values = list(values)
    best = len(values)
    best_values = list(values)

    def update_vecs(vals):
        return [_EXP[v] for v in vals]

    while time.time() < deadline:
        improved = False
        n = len(values)

        # 1. Try Insertions
        used_set = set(values)
        for i in range(n + 1):
            if time.time() > deadline:
                return best_values
            lo = (values[i - 1] + 1) if i > 0 else 2
            hi = (values[i] - 1) if i < n else 128
            for v in range(lo, hi + 1):
                if v in used_set:
                    continue
                test_vals = values[:i] + [v] + values[i:]
                test_vecs = update_vecs(test_vals)
                if _is_valid_vectors(test_vecs):
                    values = test_vals
                    used_set.add(v)
                    if len(values) > best:
                        best = len(values)
                        best_values = list(values)
                    improved = True
                    break

        # 2. Try Replacements
        for i in range(len(values)):
            if time.time() > deadline:
                return best_values
            lo = (values[i - 1] + 1) if i > 0 else 2
            hi = (values[i + 1] - 1) if i < len(values) - 1 else 128
            for v in range(lo, hi + 1):
                if v == values[i]:
                    continue
                test_vals = values[:i] + [v] + values[i + 1:]
                test_vecs = update_vecs(test_vals)
                if _is_valid_vectors(test_vecs):
                    values = test_vals
                    improved = True
                    break
            if improved:
                break

        if not improved:
            break

    return best_values


def _simulated_annealing(values, deadline):
    """SA that can traverse invalid states to escape local optima."""
    values = list(values)
    best = len(values)
    best_values = list(values)

    def get_vecs(vals):
        return [_EXP[v] for v in vals]

    def get_collisions_and_prefix(vals):
        vecs = get_vecs(vals)
        prefix = _build_prefix(vecs)
        return _count_collisions(prefix), prefix

    current_collisions, _ = get_collisions_and_prefix(values)
    T = 5.0
    T_min = 0.01
    cooling = 0.995

    random.seed(12345)

    while time.time() < deadline and T > T_min:
        n = len(values)
        # Choose a random move
        move_type = random.random()
        if move_type < 0.4 and n < 127:
            # Insert
            i = random.randint(0, n)
            lo = (values[i - 1] + 1) if i > 0 else 2
            hi = (values[i] - 1) if i < n else 128
            if lo > hi:
                T *= cooling
                continue
            v = random.randint(lo, hi)
            if v in set(values):
                T *= cooling
                continue
            test_vals = values[:i] + [v] + values[i:]
        elif move_type < 0.8 and n > 1:
            # Replace
            i = random.randint(0, n - 1)
            lo = (values[i - 1] + 1) if i > 0 else 2
            hi = (values[i + 1] - 1) if i < n - 1 else 128
            if lo > hi:
                T *= cooling
                continue
            v = random.randint(lo, hi)
            if v in set(values):
                T *= cooling
                continue
            test_vals = values[:i] + [v] + values[i + 1:]
        elif n >= 2:
            # Swap adjacent
            i = random.randint(0, n - 2)
            test_vals = list(values)
            test_vals[i], test_vals[i + 1] = test_vals[i + 1], test_vals[i]
            # Must maintain strictly increasing
            if test_vals[i] >= test_vals[i + 1]:
                T *= cooling
                continue
        else:
            T *= cooling
            continue

        new_collisions, _ = get_collisions_and_prefix(test_vals)
        delta_c = new_collisions - current_collisions
        delta_len = len(test_vals) - len(values)

        # Acceptance: prefer fewer collisions and more length
        # Energy = collisions - 10*length (so reducing collisions and increasing length both lower energy)
        energy_delta = delta_c - 10 * delta_len

        if energy_delta <= 0:
            values = test_vals
            current_collisions = new_collisions
            if len(values) > best:
                best = len(values)
                best_values = list(values)
        else:
            # Accept with probability exp(-energy_delta / T)
            import math
            if random.random() < math.exp(-energy_delta / T):
                values = test_vals
                current_collisions = new_collisions

        T *= cooling

    return best_values


def _final_polish(values, deadline):
    """Final high-precision polish: try all possible insertions and replacements exhaustively."""
    values = list(values)
    best = len(values)
    best_values = list(values)

    def get_vecs(vals):
        return [_EXP[v] for v in vals]

    # Repeatedly try to insert anywhere until no improvement
    changed = True
    while changed and time.time() < deadline:
        changed = False
        n = len(values)
        used_set = set(values)
        for i in range(n + 1):
            if time.time() > deadline:
                break
            lo = (values[i - 1] + 1) if i > 0 else 2
            hi = (values[i] - 1) if i < n else 128
            for v in range(lo, hi + 1):
                if v in used_set:
                    continue
                test_vals = values[:i] + [v] + values[i:]
                test_vecs = get_vecs(test_vals)
                if _is_valid_vectors(test_vecs):
                    values = test_vals
                    used_set.add(v)
                    changed = True
                    if len(values) > best:
                        best = len(values)
                        best_values = list(values)
                    break

    # Try replacements to potentially open up insertion space
    for _round in range(3):
        if time.time() > deadline:
            break
        n = len(values)
        for i in range(n):
            if time.time() > deadline:
                break
            lo = (values[i - 1] + 1) if i > 0 else 2
            hi = (values[i + 1] - 1) if i < n - 1 else 128
            for v in range(lo, hi + 1):
                if v == values[i]:
                    continue
                test_vals = values[:i] + [v] + values[i + 1:]
                test_vecs = get_vecs(test_vals)
                if _is_valid_vectors(test_vecs):
                    # Check if this replacement allows any new insertion
                    values = test_vals
                    # Try to insert after this change
                    n2 = len(values)
                    used_set2 = set(values)
                    for j in range(n2 + 1):
                        if time.time() > deadline:
                            break
                        lo2 = (values[j - 1] + 1) if j > 0 else 2
                        hi2 = (values[j] - 1) if j < n2 else 128
                        for v2 in range(lo2, hi2 + 1):
                            if v2 in used_set2:
                                continue
                            test_vals2 = values[:j] + [v2] + values[j:]
                            test_vecs2 = get_vecs(test_vals2)
                            if _is_valid_vectors(test_vecs2):
                                values = test_vals2
                                if len(values) > best:
                                    best = len(values)
                                    best_values = list(values)
                                break
                        else:
                            continue
                        break
                    break

    return best_values


def construct():
    deadline = time.time() + 85.0  # Safety margin before the 90s kill

    # Strategy 1: Standard greedy
    values1, products1 = [], set()
    for x in range(2, 129):
        if _can_add(values1, products1, x):
            tails = [x]
            prod = x
            for a in reversed(values1):
                prod *= a
                tails.append(prod)
            values1.append(x)
            products1.update(tails)
    best_values = list(values1)
    best_len = len(values1)

    # Strategy 2: Greedy with different starting points (skip some small values)
    for skip_start in range(2, 10):
        if time.time() > deadline:
            break
        values_try, products_try = [], set()
        for x in range(2, 129):
            if x < skip_start and x != 2:
                continue
            if _can_add(values_try, products_try, x):
                tails = [x]
                prod = x
                for a in reversed(values_try):
                    prod *= a
                    tails.append(prod)
                values_try.append(x)
                products_try.update(tails)
        if len(values_try) > best_len:
            best_len = len(values_try)
            best_values = list(values_try)

    # Strategy 3: Randomized multi-start greedy
    random.seed(42)
    for trial in range(20):
        if time.time() > deadline:
            break
        values_try, products_try = [], set()
        candidates = list(range(2, 129))
        random.shuffle(candidates)
        for x in sorted(candidates):
            if _can_add(values_try, products_try, x):
                tails = [x]
                prod = x
                for a in reversed(values_try):
                    prod *= a
                    tails.append(prod)
                values_try.append(x)
                products_try.update(tails)
        if len(values_try) > best_len:
            best_len = len(values_try)
            best_values = list(values_try)

    # Apply local search to the best found
    if time.time() < deadline:
        best_values = _local_search(best_values, deadline)
        best_len = len(best_values)

    # Apply simulated annealing to escape local optima
    if time.time() < deadline:
        sa_values = _simulated_annealing(best_values, deadline)
        if len(sa_values) > best_len:
            best_len = len(sa_values)
            best_values = sa_values

    # Final polish pass
    if time.time() < deadline:
        best_values = _final_polish(best_values, deadline)

    return best_values
# EVOLVE-BLOCK-END
