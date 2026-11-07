import math
import time
from itertools import combinations

P = {'n': 210, 'maximum_window': 10210, 'maximum_size': 100}

# Precompute prime list up to sqrt(10210) ~ 101
_primes_cache = None
def _get_primes(limit):
    global _primes_cache
    if _primes_cache is None or _primes_cache[0] < limit:
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit+1, i):
                    sieve[j] = False
        _primes_cache = (limit, [i for i in range(2, limit+1) if sieve[i]])
    return _primes_cache[1]


def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n


def _squarefree_kernel(n):
    """Return the squarefree part of n (product of primes with odd exponent)."""
    k = n
    d = 2
    kernel = 1
    while d * d <= k:
        if k % d == 0:
            exp = 0
            while k % d == 0:
                k //= d
                exp += 1
            if exp % 2 == 1:
                kernel *= d
        d += 1 if d == 2 else 2
    if k > 1:
        kernel *= k
    return kernel


def _parity_set(n):
    """Return the set of primes with odd exponent in n's factorization.
    Used for XOR-based matching."""
    k = n
    pset = set()
    primes = _get_primes(10210)
    for d in primes:
        if d * d > k:
            break
        if k % d == 0:
            exp = 0
            while k % d == 0:
                k //= d
                exp += 1
            if exp % 2 == 1:
                pset.add(d)
    if k > 1:
        pset.add(k)
    return frozenset(pset)


def _xor_pset(p1, p2):
    """Symmetric difference of two frozensets."""
    return frozenset(p1.symmetric_difference(p2))


def _validate(lst, n=210, max_window=10210, max_size=100):
    """Check all constraints."""
    if not isinstance(lst, (list, tuple)):
        return False
    if len(lst) < 1 or len(lst) > max_size:
        return False
    if len(set(lst)) != len(lst):
        return False
    for x in lst:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if x <= n or x > max_window:
            return False
    prod = n
    for x in lst:
        prod *= x
    return _is_square(prod)


def _find_best_pair(start, end, target_pset, deadline):
    """Find a pair (a, b) in [start, end] with a < b, both > n,
    such that parity_set(a) XOR parity_set(b) == target_pset.
    Minimize max(a, b)."""
    n = P['n']
    if start <= n:
        start = n + 1
    if end > P['maximum_window']:
        end = P['maximum_window']
    if start >= end:
        return None, None

    # Precompute parity sets for all numbers in range
    psets = {}
    for v in range(start, end + 1):
        if time.time() > deadline:
            break
        psets[v] = _parity_set(v)

    if not psets:
        return None, None

    # Build a map from parity set to list of values (sorted)
    pset_to_vals = {}
    for v, ps in psets.items():
        if ps not in pset_to_vals:
            pset_to_vals[ps] = []
        pset_to_vals[ps].append(v)

    # For each value a, find b such that pset(b) = pset(a) XOR target_pset
    best = None
    for a in sorted(psets.keys()):
        if time.time() > deadline:
            break
        pa = psets[a]
        needed = _xor_pset(pa, target_pset)
        if needed in pset_to_vals:
            for b in pset_to_vals[needed]:
                if b == a:
                    continue
                mx = max(a, b)
                if best is None or mx < best[0]:
                    best = (mx, [a, b])
                # Since we want min max, and b is in sorted order,
                # the first valid b >= a gives smallest max for this a
                # But we need b != a, so check next if b == a
                if b > a:
                    break
                # b < a, check if there's a larger b
                # Actually we just need any b != a with minimal max
                # Let's continue to find the best b for this a
        # Optimization: if a alone is already larger than current best max, stop
        if best is not None and a >= best[0]:
            break

    if best is not None:
        return best
    return None, None


def _find_best_triple(start, end, target_pset, deadline):
    """Find a triple (a, b, c) in [start, end] with distinct values,
    such that parity_set(a) XOR parity_set(b) XOR parity_set(c) == target_pset.
    Minimize max(a, b, c)."""
    n = P['n']
    if start <= n:
        start = n + 1
    if end > P['maximum_window']:
        end = P['maximum_window']
    if start >= end:
        return None, None

    # Limit search space
    max_range = 3000
    end_eff = min(end, start + max_range)

    psets = {}
    for v in range(start, end_eff + 1):
        if time.time() > deadline:
            break
        psets[v] = _parity_set(v)

    # Build map from parity set to list of values
    pset_to_vals = {}
    for v, ps in psets.items():
        if ps not in pset_to_vals:
            pset_to_vals[ps] = []
        pset_to_vals[ps].append(v)

    vals = sorted(psets.keys())
    best = None

    # For pairs (a, b), check if there exists c with pset(c) = pset(a)^pset(b)^target
    for i, a in enumerate(vals):
        if time.time() > deadline:
            break
        pa = psets[a]
        if best is not None and a >= best[0]:
            break
        for j in range(i + 1, len(vals)):
            b = vals[j]
            if best is not None and b >= best[0]:
                break
            pb = psets[b]
            needed = _xor_pset(_xor_pset(pa, pb), target_pset)
            if needed in pset_to_vals:
                for c in pset_to_vals[needed]:
                    if c == a or c == b:
                        continue
                    mx = max(a, b, c)
                    if best is None or mx < best[0]:
                        best = (mx, [a, b, c])
                    # Found a valid c; since we want min max and c is from list,
                    # take the smallest valid c
                    if c > b:
                        break

    if best is not None:
        return best
    return None, None


def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # safety margin before 90s kill

    n = P['n']
    target_pset = _parity_set(n)  # {2, 3, 5, 7}

    # Baseline: single element 210 * 4 = 840
    baseline_val = 210 * 4
    best_val = baseline_val
    best_list = [baseline_val]

    # Try single elements: find smallest x > n such that sf(x) == sf(n)
    # x = 210 * s^2, s >= 2
    for s in range(2, 100):
        cand = n * s * s
        if cand > P['maximum_window']:
            break
        if cand > n:
            v = cand - n
            if v < best_val:
                best_val = v
                best_list = [cand]
            break  # smallest s gives smallest cand

    # Direct search: for each candidate max M from n+1 upward, check if there
    # exists a subset of {n+1, ..., M} whose product * n is a perfect square.
    # Start with small M and increase until we find a solution or time runs out.
    
    # First, try very small windows with exhaustive pair/triple search
    # Precompute parity sets up to a reasonable limit
    max_precompute = min(2000, P['maximum_window'])
    psets_cache = {}
    for v in range(n + 1, max_precompute + 1):
        psets_cache[v] = _parity_set(v)
    
    # Try to find a single element with parity == target
    # Already handled above via 210*s^2
    
    # Try pairs: for each pair (a,b) with a < b <= M, check parity(a)^parity(b) == target
    # We want to minimize M = max(a,b)
    # Strategy: iterate M from n+2 upward, for each M check if there's a in [n+1, M-1]
    # such that parity(a) ^ parity(M) == target
    
    # Build inverse lookup: for a given target_pset and one element's parity,
    # what parity does the other element need?
    
    # Efficient approach: group numbers by their parity set
    # For a target M, we need some a < M with parity(a) = parity(M) ^ target_pset
    
    # Let's try a different approach: iterate M from small to large
    # For each M, check if any a in [n+1, M-1] has the right parity
    
    # Pre-build parity groups as we go
    parity_groups = {}  # frozenset -> list of values
    
    found = False
    for M in range(n + 2, max_precompute + 1):
        if time.time() > deadline - 5:
            break
        
        pM = psets_cache[M]
        needed = _xor_pset(pM, target_pset)
        
        # Check if any a in parity_groups[needed] has a < M
        if needed in parity_groups:
            for a in parity_groups[needed]:
                if a < M:
                    lst = [a, M]
                    if _validate(lst):
                        v = M - n
                        if v < best_val:
                            best_val = v
                            best_list = list(lst)
                        found = True
                        break
        
        if found and best_val < 20:
            break
        
        # Add M to its parity group
        if pM not in parity_groups:
            parity_groups[pM] = []
        parity_groups[pM].append(M)
    
    # If we didn't find a great pair, try triples in small window
    if time.time() < deadline - 10 and best_val > 30:
        # Try triples: for M, check if there exist a, b < M with
        # parity(a) ^ parity(b) ^ parity(M) == target
        # i.e., parity(a) ^ parity(b) == parity(M) ^ target
        
        # For each M, we need a pair (a,b) from [n+1, M-1] with combined parity = parity(M)^target
        # This is equivalent to: is there a pair in the prefix [n+1, M-1] with XOR = needed?
        
        # We can precompute all pair XORs as we extend M
        # But that's O(M^2). Instead, for each M, check pairs in the prefix.
        
        # Alternative: for each M, iterate a in [n+1, M-1], compute needed_b = parity(a) ^ parity(M) ^ target
        # and check if needed_b exists in parity_groups with value < M and != a
        
        triple_found = False
        for M in range(n + 3, min(n + 300, max_precompute + 1)):
            if time.time() > deadline - 5:
                break
            
            pM = psets_cache[M]
            
            for a in range(n + 1, M):
                if time.time() > deadline - 5:
                    break
                pa = psets_cache[a]
                needed_b = _xor_pset(_xor_pset(pa, pM), target_pset)
                
                if needed_b in parity_groups:
                    for b in parity_groups[needed_b]:
                        if b < M and b != a:
                            lst = [a, b, M]
                            if _validate(lst):
                                v = M - n
                                if v < best_val:
                                    best_val = v
                                    best_list = list(lst)
                                triple_found = True
                                break
                if triple_found:
                    break
            
            if triple_found and best_val < 15:
                break
            
            # Add M to parity groups (for future M values)
            if pM not in parity_groups:
                parity_groups[pM] = []
            parity_groups[pM].append(M)
    
    # Final validation
    if not _validate(best_list):
        best_list = [n * 4]

    return [int(x) for x in best_list]
