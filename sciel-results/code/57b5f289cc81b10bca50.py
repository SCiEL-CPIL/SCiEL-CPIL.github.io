"""Combined program: prime-exponent vector model with branch-and-bound search."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

_P0 = time.time()
_DEADLINE = _P0 + 85.0  # safety margin under 90s

# All primes in 2..40
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
_PRIME_SET = set(_PRIMES)
_NUM_PRIMES = len(_PRIMES)


def _factorize(n):
    """Return tuple of (prime, exponent) for n's factorization."""
    factors = []
    for p in _PRIMES:
        if p * p > n:
            break
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            factors.append((p, e))
    if n > 1:
        factors.append((n, 1))
    return tuple(factors)


def _exponent_vector(n):
    """Return a tuple of exponents for n over the prime list."""
    vec = [0] * _NUM_PRIMES
    for i, p in enumerate(_PRIMES):
        while n % p == 0:
            vec[i] += 1
            n //= p
    if n > 1:
        # Should not happen for n <= 40 since we have all primes up to 37
        pass
    return tuple(vec)


def _is_square_free(n):
    """Check if n is square-free."""
    for p in _PRIMES:
        if p * p > n:
            break
        if n % (p * p) == 0:
            return False
    return True


def _can_add_vec(prods_vecs, x_vec):
    """True iff adding x_vec keeps all subset product exponent vectors distinct."""
    for pv in prods_vecs:
        new_vec = tuple(pv[i] + x_vec[i] for i in range(_NUM_PRIMES))
        if new_vec in prods_vecs:
            return False
    return True


def _try_add_vec(prods_vecs, x_vec):
    """Try to add x_vec; returns (success, new_prods_vecs)."""
    if not _can_add_vec(prods_vecs, x_vec):
        return False, prods_vecs
    new_prods = set(prods_vecs)
    for pv in prods_vecs:
        new_prods.add(tuple(pv[i] + x_vec[i] for i in range(_NUM_PRIMES)))
    return True, new_prods


def _verify_result(selected):
    """Verify that all subset products are distinct using actual integer products."""
    prods = {1}
    for x in selected:
        new_prods = set(prods)
        for p in prods:
            new_prods.add(p * x)
        if len(new_prods) != len(prods) * 2:
            return False
        prods = new_prods
    return True


def construct():
    if time.time() > _DEADLINE:
        return [2]

    # All primes in 2..40
    primes = list(_PRIMES)

    # Non-square-free composites in 2..40 (candidates to add)
    non_sqfree_composites = []
    for c in range(4, 41):
        if c in _PRIME_SET:
            continue
        if not _is_square_free(c):
            non_sqfree_composites.append(c)
    # non_sqfree_composites = [4,8,9,12,16,18,20,24,25,27,28,32,36,40]

    # Start with all primes
    selected = list(primes)
    prods_vecs = {tuple([0] * _NUM_PRIMES)}
    for x in primes:
        xv = _exponent_vector(x)
        _, prods_vecs = _try_add_vec(prods_vecs, xv)

    # Greedy ascending
    best_selected = selected[:]
    best_prods = prods_vecs.copy()

    for x in sorted(non_sqfree_composites):
        if time.time() > _DEADLINE:
            break
        if len(best_selected) >= 18:
            break
        xv = _exponent_vector(x)
        ok, new_prods = _try_add_vec(best_prods, xv)
        if ok:
            best_selected.append(x)
            best_prods = new_prods

    if len(best_selected) >= 18:
        if _verify_result(best_selected[:18]):
            return best_selected[:18]

    # Greedy descending
    selected = list(primes)
    prods_vecs = {tuple([0] * _NUM_PRIMES)}
    for x in primes:
        xv = _exponent_vector(x)
        _, prods_vecs = _try_add_vec(prods_vecs, xv)
    for x in sorted(non_sqfree_composites, reverse=True):
        if time.time() > _DEADLINE:
            break
        if len(selected) >= 18:
            break
        xv = _exponent_vector(x)
        ok, new_prods = _try_add_vec(prods_vecs, xv)
        if ok:
            selected.append(x)
            prods_vecs = new_prods
    if len(selected) > len(best_selected):
        best_selected = selected
        best_prods = prods_vecs

    if len(best_selected) >= 18:
        if _verify_result(best_selected[:18]):
            return best_selected[:18]

    # Branch-and-bound: try all combinations of 6 from the 14 non-sqfree composites
    # C(14,6) = 3003, very manageable
    for combo in combinations(non_sqfree_composites, 6):
        if time.time() > _DEADLINE:
            break
        selected = list(primes) + list(combo)
        # Verify all subset products distinct using exponent vectors
        prods_vecs = {tuple([0] * _NUM_PRIMES)}
        valid = True
        for x in selected:
            xv = _exponent_vector(x)
            if not _can_add_vec(prods_vecs, xv):
                valid = False
                break
            new_prods = set(prods_vecs)
            for pv in prods_vecs:
                new_prods.add(tuple(pv[i] + xv[i] for i in range(_NUM_PRIMES)))
            prods_vecs = new_prods
        if valid:
            if _verify_result(selected):
                return selected

    # If we still haven't found 18, return the best we have
    if len(best_selected) >= 2:
        if _verify_result(best_selected):
            return best_selected

    # Fallback: return a known valid small set
    return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
# EVOLVE-BLOCK-END
