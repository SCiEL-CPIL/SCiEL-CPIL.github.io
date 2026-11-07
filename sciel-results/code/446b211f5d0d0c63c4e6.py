"""Rewritten solver: maximum subset of 2..256 with no a | b*c for distinct a,b,c."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from itertools import combinations
import random

DEADLINE = 80.0  # safety margin from 90s wall-clock

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _factorize(n):
    """Return prime factorization as list of (prime, exponent)."""
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            cnt = 0
            while n % d == 0:
                n //= d
                cnt += 1
            factors.append((d, cnt))
        d += 1
    if n > 1:
        factors.append((n, 1))
    return factors

def _a_divides_bc(a_factors, b_factors, c_factors):
    """Check if a divides b*c using factor exponents."""
    # Build exponent map for b*c
    bc_exp = {}
    for p, e in b_factors:
        bc_exp[p] = bc_exp.get(p, 0) + e
    for p, e in c_factors:
        bc_exp[p] = bc_exp.get(p, 0) + e
    # Check if a's exponents are all <= bc's exponents
    for p, e in a_factors:
        if bc_exp.get(p, 0) < e:
            return False
    return True

def _greedy_build(factors_map, candidates):
    """Greedy: add elements one by one if they don't create violations."""
    selected = []
    for x in sorted(candidates):
        fx = factors_map[x]
        ok = True
        # Check with all pairs in selected
        for i, y in enumerate(selected):
            fy = factors_map[y]
            for z in selected[i+1:]:
                fz = factors_map[z]
                # a=x, b=y, c=z
                if _a_divides_bc(fx, fy, fz):
                    ok = False
                    break
                # a=y, b=x, c=z
                if _a_divides_bc(fy, fx, fz):
                    ok = False
                    break
                # a=z, b=x, c=y
                if _a_divides_bc(fz, fx, fy):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            selected.append(x)
    return selected

def _try_add(selected, x, factors_map):
    """Check if x can be added to selected without violations."""
    fx = factors_map[x]
    for i, y in enumerate(selected):
        fy = factors_map[y]
        for z in selected[i+1:]:
            fz = factors_map[z]
            if _a_divides_bc(fx, fy, fz):
                return False
            if _a_divides_bc(fy, fx, fz):
                return False
            if _a_divides_bc(fz, fx, fy):
                return False
    return True

def _local_search(selected, factors_map, all_elements, time_limit):
    """Improve via adding elements and swap moves."""
    best = list(selected)
    best_size = len(best)
    start = time.time()
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        # Try adding each unselected element
        unselected = [x for x in all_elements if x not in best]
        for x in unselected:
            if time.time() - start > time_limit:
                break
            if _try_add(best, x, factors_map):
                best.append(x)
                best_size += 1
                improved = True
                # Restart to find more additions
                break
        # Try swap: remove one, add two
        if not improved and best_size < len(all_elements) - 1:
            for i in range(len(best)):
                if time.time() - start > time_limit:
                    break
                removed = best.pop(i)
                # Try to add two elements
                unselected = [x for x in all_elements if x not in best]
                for a in unselected:
                    if time.time() - start > time_limit:
                        break
                    if _try_add(best, a, factors_map):
                        best.append(a)
                        unselected2 = [x for x in all_elements if x not in best]
                        for b in unselected2:
                            if time.time() - start > time_limit:
                                break
                            if _try_add(best, b, factors_map):
                                best.append(b)
                                if len(best) > best_size:
                                    best_size = len(best)
                                    improved = True
                                break
                        if len(best) <= best_size:
                            best.remove(a)
                if not improved:
                    best.insert(i, removed)
        if not improved:
            break
    return best

def _validate(result):
    """Verify the set satisfies the constraint."""
    for a in result:
        for i, b in enumerate(result):
            if b == a:
                continue
            for c in result:
                if c == a or c == b:
                    continue
                if (b * c) % a == 0:
                    return False
    return True

def construct():
    start_time = time.time()
    n = 256
    
    # Precompute factorizations
    factors_map = {}
    for x in range(2, n + 1):
        factors_map[x] = _factorize(x)
    
    # Start with primes (they can never divide b*c for b,c < a if a is prime > max(b,c))
    # Actually primes can divide b*c if one of b or c is a multiple of that prime
    # But for large primes, this is less likely. Let's use greedy from sorted candidates.
    
    # Strategy: prioritize larger numbers first (they're less likely to divide products)
    candidates = list(range(2, n + 1))
    
    # Greedy: try larger numbers first
    selected = _greedy_build(factors_map, sorted(candidates, reverse=True))
    
    # Local search to improve
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 5:
        selected = _local_search(selected, factors_map, candidates, remaining - 5)
    
    # Verify
    result = sorted(selected)
    result = [int(x) for x in result]
    
    if not _validate(result):
        # Fallback: just primes
        result = [x for x in range(2, n + 1) if _is_prime(x)]
        result = [int(x) for x in result]
    
    return result
# EVOLVE-BLOCK-END
