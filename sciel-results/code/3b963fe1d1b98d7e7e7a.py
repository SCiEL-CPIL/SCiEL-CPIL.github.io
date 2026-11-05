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
            while _d % _p == 0:
                _d //= _p
        _p += 1
    if _d > 1:
        _PRIMES.append(_d)
_PRIMES = sorted(set(_PRIMES))

_EXP = {}
_NUM_PRIMES = len(_PRIMES)
_prime_to_idx = {p: i for i, p in enumerate(_PRIMES)}
for _x in range(2, 129):
    _d = _x
    _vec = [0] * _NUM_PRIMES
    for _p in _PRIMES:
        if _p * _p > _d:
            break
        if _d % _p == 0:
            _c = 0
            while _d % _p == 0:
                _d //= _p
                _c += 1
            _vec[_prime_to_idx[_p]] = _c
    if _d > 1:
        _vec[_prime_to_idx[_d]] = 1
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


def _is_valid_vectors(vecs):
    """Check if all nonempty consecutive block sums of exponent vectors are distinct."""
    n = len(vecs)
    if n == 0:
        return True
    
    num_primes = len(vecs[0])
    P = [None] * (n + 1)
    P[0] = [0] * num_primes
    
    for i in range(n):
        prev = P[i]
        curr = vecs[i]
        P[i+1] = [prev[k] + curr[k] for k in range(num_primes)]
        
    seen = set()
    for i in range(n):
        for j in range(i, n):
            s_list = [P[j+1][k] - P[i][k] for k in range(num_primes)]
            s_tuple = tuple(s_list)
            if s_tuple in seen:
                return False
            seen.add(s_tuple)
    return True

def _check_insertion(values, vecs, pos, v):
    """Check if inserting v at position pos maintains validity. Returns True if valid."""
    n = len(values)
    if pos == n:
        # Append case: only need to check new blocks ending at the new element
        if values and v <= values[-1]:
            return False
        new_vec = _EXP[v]
        # Check all blocks ending at the new position
        prod_vec = list(new_vec)
        seen = set()
        for k in range(n, -1, -1):
            if k < n:
                prod_vec = [prod_vec[j] + vecs[k][j] for j in range(_NUM_PRIMES)]
            # Check if this sum already exists in any block
            # We need to check against all existing block sums
            # This is expensive, so we'll just do full validation for simplicity
            pass
        # Fall back to full validation
        test_vecs = vecs[:pos] + [new_vec] + vecs[pos:]
        return _is_valid_vectors(test_vecs)
    else:
        # Insertion in middle
        if pos > 0 and v <= values[pos-1]:
            return False
        if pos < n and v >= values[pos]:
            return False
        new_vec = _EXP[v]
        test_vecs = vecs[:pos] + [new_vec] + vecs[pos:]
        return _is_valid_vectors(test_vecs)

def _local_search(values, deadline):
    """Local search using vector sums for fast validity checks."""
    values = list(values)
    vecs = [_EXP[v] for v in values]
    best = len(values)
    best_values = list(values)
    
    def update_vecs(vals):
        return [_EXP[v] for v in vals]

    while time.time() < deadline:
        improved = False
        n = len(values)
        
        # 1. Try Insertions - try all possible insertion points and values
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
                    vecs = test_vecs
                    used_set.add(v)
                    if len(values) > best:
                        best = len(values)
                        best_values = list(values)
                    improved = True
                    break
        
        # 2. Try Replacements
        if time.time() > deadline:
            return best_values
        n = len(values)
        for i in range(n):
            if time.time() > deadline:
                return best_values
            lo = (values[i - 1] + 1) if i > 0 else 2
            hi = (values[i + 1] - 1) if i < n - 1 else 128
            for v in range(lo, hi + 1):
                if v == values[i]:
                    continue
                test_vals = values[:i] + [v] + values[i+1:]
                test_vecs = update_vecs(test_vals)
                if _is_valid_vectors(test_vecs):
                    values = test_vals
                    vecs = test_vecs
                    improved = True
                    break
        
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

    # Strategy 4: Beam search with width 50
    if time.time() < deadline:
        beam = [(list(values1), set(products1))]
        beam_width = 50
        for _ in range(3):  # 3 beam search iterations
            new_beam = []
            for vals, prods in beam:
                n = len(vals)
                used = set(vals)
                for i in range(n + 1):
                    if time.time() > deadline:
                        break
                    lo = (vals[i-1] + 1) if i > 0 else 2
                    hi = (vals[i] - 1) if i < n else 128
                    for v in range(lo, hi + 1):
                        if v in used:
                            continue
                        test_vals = vals[:i] + [v] + vals[i:]
                        test_vecs = [_EXP[x] for x in test_vals]
                        if _is_valid_vectors(test_vecs):
                            # Compute products for this candidate
                            test_prods = set()
                            for k in range(len(test_vals)):
                                prod = 1
                                for m in range(k, len(test_vals)):
                                    prod *= test_vals[m]
                                    test_prods.add(prod)
                            new_beam.append((test_vals, test_prods))
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            # Keep top beam_width by length
            new_beam.sort(key=lambda x: len(x[0]), reverse=True)
            beam = new_beam[:beam_width]
            if beam and len(beam[0][0]) > best_len:
                best_len = len(beam[0][0])
                best_values = list(beam[0][0])
            if not beam:
                break

    # Apply local search to the best found
    if time.time() < deadline:
        best_values = _local_search(best_values, deadline)

    return best_values
# EVOLVE-BLOCK-END
