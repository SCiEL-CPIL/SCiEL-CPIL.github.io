import time

N = 5000
DEADLINE = 300.0  # safety margin vs the 330s kill


def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def _is_admissible(res, primes):
    L = len(res)
    if L == 0:
        return True
    present = {}
    for p in primes:
        if p > L:
            break
        seen = set()
        for x in res:
            seen.add(x % p)
        if len(seen) == p:
            return False
    return True


def construct():
    start = time.time()

    def now():
        return time.time()

    primes = _primes(N)

    # Per-prime residue counts and number of *present* (nonzero) residues.
    counts = {p: [0] * p for p in primes}
    present = {p: 0 for p in primes}
    in_set = [True] * (N + 1)

    # Initialise from the full set [0, N].
    for x in range(N + 1):
        for p in primes:
            r = x % p
            if counts[p][r] == 0:
                present[p] += 1
            counts[p][r] += 1

    L = N + 1

    def remove(x):
        nonlocal L
        in_set[x] = False
        L -= 1
        for p in primes:
            r = x % p
            counts[p][r] -= 1
            if counts[p][r] == 0:
                present[p] -= 1

    def add(x):
        nonlocal L
        in_set[x] = True
        L += 1
        for p in primes:
            r = x % p
            if counts[p][r] == 0:
                present[p] += 1
            counts[p][r] += 1

    # ---- Phase 1: greedy removal until admissible ----
    # For the smallest violating prime, remove one element from each residue class
    # to break the "all residues present" condition with minimal element loss.
    while now() - start < DEADLINE - 40:
        viol = None
        for p in primes:
            if p > L:
                break
            if present[p] == p:  # all residues present -> must break this prime
                viol = p
                break
        if viol is None:
            break
        # Remove one element from each residue class of viol.
        # Choose elements that are "least costly": prefer elements whose removal
        # doesn't break other primes' constraints (i.e., elements in residue classes
        # that still have count > 1 for all primes <= L).
        to_remove = []
        for r in range(viol):
            # Find best element in residue class r for prime viol
            best_x = -1
            best_cost = float('inf')
            for x in range(N + 1):
                if not in_set[x] or x % viol != r:
                    continue
                # Cost: how many other primes p' <= L would have this residue class
                # drop to 0 if we remove x? (i.e., counts[p'][x%p'] == 1)
                cost = 0
                for p2 in primes:
                    if p2 > L:
                        break
                    r2 = x % p2
                    if counts[p2][r2] == 1:
                        cost += 1
                if cost < best_cost:
                    best_cost = cost
                    best_x = x
                    if best_cost == 0:
                        break
            if best_x >= 0:
                to_remove.append(best_x)
        for x in to_remove:
            if in_set[x]:
                remove(x)

    # ---- Phase 2: add-back local improvement ----
    # Adding x is safe iff no prime p <= L+1 has exactly one missing residue
    # equal to x % p.
    def safe(x):
        limit = L + 1
        for p in primes:
            if p > limit:
                break
            if present[p] == p - 1 and counts[p][x % p] == 0:
                return False
        return True

    # Multiple passes with randomized ordering to escape local optima.
    import random
    rng = random.Random(42)
    for _ in range(1000):
        if now() - start > DEADLINE - 25:
            break
        added = False
        # Try all candidates in a shuffled order
        candidates = [x for x in range(N + 1) if not in_set[x]]
        rng.shuffle(candidates)
        for x in candidates:
            if in_set[x]:
                continue
            if safe(x):
                add(x)
                added = True
        if not added:
            break

    res = [x for x in range(N + 1) if in_set[x]]

    # ---- Final safety: ensure admissibility before returning ----
    if not _is_admissible(res, primes):
        # Rebuild tracking state from res and finish a greedy removal.
        for p in primes:
            counts[p] = [0] * p
            present[p] = 0
        in_set = [False] * (N + 1)
        for x in res:
            in_set[x] = True
            for p in primes:
                r = x % p
                if counts[p][r] == 0:
                    present[p] += 1
                counts[p][r] += 1
        L = len(res)
        while now() - start < DEADLINE - 5:
            viol = None
            for p in primes:
                if p > L:
                    break
                if present[p] == p:
                    viol = p
                    break
            if viol is None:
                break
            c = counts[viol]
            m = min(c)
            r = c.index(m)
            for x in range(N + 1):
                if in_set[x] and x % viol == r:
                    remove(x)
        res = [x for x in range(N + 1) if in_set[x]]

    return res
