"""Model-and-optimize: parameterize the set and optimize (branch-and-bound) its size
against the constraint 'for all distinct a,b,c in S: a does not divide b*c'."""

import math
import time

_P = {'n': 256}
_DEADLINE = 85.0  # wall-clock safety margin (evaluator kills at 90s)
_START = time.time()


def _elapsed():
    return time.time() - _START


def _precompute(n):
    """Precompute, for each k in 2..n, the list of multiples of k in 2..n,
    and for each k, the list of d that divide k (d in 2..n, d!=k).
    Then a 'divides' test a | (b*c) is expensive; instead we precompute, for each pair (b,c),
    the set of a in 2..n that divide b*c. But b*c up to 256*256=65536; a divides b*c iff
    a | lcm-related... We precompute divisors of every integer up to 65536."""
    n = n
    max_prod = n * n
    # divisors up to max_prod
    divs = [[] for _ in range(max_prod + 1)]
    for d in range(2, n + 1):
        for m in range(d, max_prod + 1, d):
            divs[m].append(d)
    # For quick access: a divides (b*c)  <=>  a in divs[b*c]
    return n, divs


def construct():
    n = _P['n']
    n, divs = _precompute(n)
    # universe
    uni = list(range(2, n + 1))

    # Precompute conflict sets: for each x, the set of pairs (a,b) in uni x uni (a<b)
    # such that adding x to a set containing both a and b would violate the constraint.
    # Specifically, conflict[x] is a set of frozensets {a,b} where x cannot coexist with both a and b.
    # But more usefully: for can_add(x, cur), we need to check for each pair (a,b) in cur:
    #   a | x*b, b | x*a, or x | a*b
    # Precompute for each x, a list of (a, b, type) triples indicating conflicts.
    # Actually, let's precompute: for each x and each a, the set of b such that conflict exists.
    # conflict[x][a] = set of b where x,a,b form an invalid triple.
    
    # For speed, precompute:
    # bad[x][a] = set of b (b != x, b != a) such that {x,a,b} is invalid
    # This is O(n^3) = 256^3 ~ 16M which is fine.
    
    # Actually, let's precompute a 3D boolean or use a different approach.
    # For can_add(x, cur): for each a in cur, for each b in cur (a<b), check if x conflicts with pair (a,b).
    # Precompute: conflict_pair[x][a][b] = True if {x,a,b} is invalid.
    # But that's n^3 booleans = 16M, which is ~16MB, acceptable.
    
    # Let's use a different approach: for each x, precompute a list of (a, b) pairs that conflict with x.
    # Then can_add(x, cur) checks if any such pair has both elements in cur.
    
    # For each x, compute all pairs (a, b) with a < b, a != x, b != x, such that {x,a,b} is invalid.
    # Store as a set of tuples for fast lookup.
    
    conflict_pairs = {}  # x -> set of (a, b) where a < b, a != x, b != x, and {x,a,b} invalid
    
    for x in uni:
        cp = set()
        for i, a in enumerate(uni):
            if a == x:
                continue
            for b in uni[i+1:]:
                if b == x:
                    continue
                # Check if {x, a, b} is invalid: any of x|a*b, a|x*b, b|x*a
                if x in divs[a * b] or a in divs[x * b] or b in divs[x * a]:
                    cp.add((a, b) if a < b else (b, a))
        conflict_pairs[x] = cp

    def can_add_fast(x, curset):
        """Check if x can be added to curset without violating constraints."""
        cp = conflict_pairs[x]
        # For each conflicting pair (a,b), check if both a and b are in curset
        for a, b in cp:
            if a in curset and b in curset:
                return False
        return True

    # Greedy initial solution
    def greedy(order=None):
        S = []
        Sset = set()
        if order is None:
            order = uni
        for x in order:
            if can_add_fast(x, Sset):
                S.append(x)
                Sset.add(x)
        return S

    # Try different orderings for greedy
    best_greedy = []
    best_greedy_set = set()
    
    # Order 1: natural
    S1 = greedy(uni)
    if len(S1) > len(best_greedy):
        best_greedy = S1
        best_greedy_set = set(S1)
    
    # Order 2: primes first
    def is_prime(k):
        if k < 2:
            return False
        for d in range(2, int(math.isqrt(k)) + 1):
            if k % d == 0:
                return False
        return True
    
    primes = [x for x in uni if is_prime(x)]
    composites = [x for x in uni if not is_prime(x)]
    S2 = greedy(primes + composites)
    if len(S2) > len(best_greedy):
        best_greedy = S2
        best_greedy_set = set(S2)
    
    # Order 3: by number of prime factors (ascending) - "simpler" numbers first
    def num_prime_factors(k):
        count = 0
        d = 2
        while d * d <= k:
            while k % d == 0:
                count += 1
                k //= d
            d += 1
        if k > 1:
            count += 1
        return count
    
    ordered_by_pf = sorted(uni, key=num_prime_factors)
    S3 = greedy(ordered_by_pf)
    if len(S3) > len(best_greedy):
        best_greedy = S3
        best_greedy_set = set(S3)
    
    # Order 4: descending by value
    S4 = greedy(list(reversed(uni)))
    if len(S4) > len(best_greedy):
        best_greedy = S4
        best_greedy_set = set(S4)
    
    cur = best_greedy[:]
    curset = set(cur)
    
    import random
    rng = random.Random(12345)
    
    best = cur[:]
    best_set = curset.copy()
    
    def local_search(cur, curset, budget):
        nonlocal best, best_set
        improved = True
        while improved and _elapsed() < budget:
            improved = False
            missing = [x for x in uni if x not in curset]
            rng.shuffle(missing)
            for x in missing:
                if _elapsed() > budget:
                    break
                if can_add_fast(x, curset):
                    cur.append(x)
                    curset.add(x)
                    improved = True
                    if len(cur) > len(best):
                        best = cur[:]
                        best_set = curset.copy()
        
        # Try swap moves: remove one element, try to add two
        if _elapsed() < budget:
            for _ in range(50):  # limited swap attempts
                if _elapsed() > budget:
                    break
                if len(cur) < 2:
                    break
                # Pick a random element to remove
                idx = rng.randint(0, len(cur) - 1)
                removed = cur.pop(idx)
                curset.discard(removed)
                
                # Try to add as many as possible
                missing = [x for x in uni if x not in curset]
                rng.shuffle(missing)
                added_any = False
                for x in missing:
                    if _elapsed() > budget:
                        break
                    if can_add_fast(x, curset):
                        cur.append(x)
                        curset.add(x)
                        added_any = True
                
                if len(cur) > len(best):
                    best = cur[:]
                    best_set = curset.copy()
                
                # If we didn't improve, put the removed element back
                if not added_any or len(cur) <= len(best_greedy):
                    cur.append(removed)
                    curset.add(removed)
        
        return cur, curset

    # Run local search with restarts
    for restart in range(500):
        if _elapsed() > _DEADLINE - 3:
            break
        
        # Start from best or perturbed best
        if restart == 0:
            cur = best[:]
            curset = set(cur)
        else:
            cur = best[:]
            curset = set(cur)
            # Perturb: remove 1-3 random elements
            k = min(len(cur), rng.randint(1, 3))
            for _ in range(k):
                if cur:
                    idx = rng.randint(0, len(cur) - 1)
                    r = cur.pop(idx)
                    curset.discard(r)
        
        local_search(cur, curset, _DEADLINE - 3)
        
        if len(cur) > len(best):
            best = cur[:]
            best_set = curset.copy()
    
    result = best
    if not result:
        result = [2]
    
    # Verify
    def is_valid_set(S):
        L = len(S)
        Sset = set(S)
        for i in range(L):
            a = S[i]
            for j in range(i + 1, L):
                b = S[j]
                for k in range(j + 1, L):
                    c = S[k]
                    if a in divs[b * c] or b in divs[a * c] or c in divs[a * b]:
                        return False
        return True
    
    if not is_valid_set(result):
        # Rebuild safely
        result = []
        rs = set()
        for x in uni:
            if can_add_fast(x, rs):
                result.append(x)
                rs.add(x)
    
    return [int(x) for x in result]
