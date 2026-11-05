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
for _x in range(2, 129):
    _d = _x
    _vec = []
    for _p in _PRIMES:
        if _p * _p > _d:
            break
        _c = 0
        while _d % _p == 0:
            _d //= _p
            _c += 1
        _vec.append(_c)
    if _d > 1:
        _vec.append(1)
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


def _local_search(values, products, deadline):
    """Local search using the prime-exponent-vector encoding as a fast, exact
    distinctness surrogate. Tries remove-and-reinsert and insert-new moves."""
    values = list(values)
    best = len(values)
    best_values = list(values)
    prefix = [()]
    for v in values:
        prefix.append(prefix[-1] + _EXP[v])
    n = len(values)
    m = n + 1
    diff = [None] * (m * m)

    def _distinct():
        for i in range(n):
            pe = prefix[i]
            base = i * m
            for j in range(i, n):
                s = pe
                for k in range(j + 1, n):
                    s = s + _EXP[k]
                diff[base + k] = s
        for a in range(m * m):
            d = diff[a]
            if d is not None:
                for b in range(a + 1, m * m):
                    if diff[b] == d:
                        return False
        return True

    for _ in range(4):  # Multiple passes
        improved = False

        # Remove-and-reinsert: drop one element, re-greedy the whole range.
        for i in range(n):
            if time.time() > deadline:
                break
            vals_without = values[:i] + values[i + 1:]
            prods_without = _all_products(vals_without)
            new_vals, new_prods = _greedy_pass(vals_without, prods_without, start_from=2)
            if len(new_vals) > best:
                best = len(new_vals)
                best_values = list(new_vals)
                values = list(new_vals)
                prefix = [()]
                for v in values:
                    prefix.append(prefix[-1] + _EXP[v])
                n = len(values)
                m = n + 1
                diff = [None] * (m * m)
                improved = True

        # Insert-new: try inserting a fresh value at each gap.
        used_set = set(values)
        for i in range(len(values) + 1):
            if time.time() > deadline:
                break
            lo = (values[i - 1] + 1) if i > 0 else 2
            hi = (values[i] - 1) if i < len(values) else 128
            for v in range(lo, hi + 1):
                if v in used_set:
                    continue
                test_vals = values[:i] + [v] + values[i:]
                test_prods = _all_products(test_vals)
                if len(test_prods) == len(test_vals) * (len(test_vals) + 1) // 2:
                    new_vals, new_prods = _greedy_pass(test_vals, test_prods, start_from=128)
                    if len(new_vals) > best:
                        best = len(new_vals)
                        best_values = list(new_vals)
                        values = list(new_vals)
                        prefix = [()]
                        for v in values:
                            prefix.append(prefix[-1] + _EXP[v])
                        n = len(values)
                        m = n + 1
                        diff = [None] * (m * m)
                        used_set = set(values)
                        improved = True
        if not improved:
            break

    return best_values


def construct():
    deadline = time.time() + 55.0  # Safety margin before the 60s kill

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
        products_best = _all_products(best_values)
        best_values = _local_search(best_values, products_best, deadline)

    return best_values
# EVOLVE-BLOCK-END
