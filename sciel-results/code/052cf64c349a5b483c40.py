import time
import numpy as np

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
    # Only primes up to N+1 matter for admissibility
    relevant_primes = [p for p in primes if p <= N + 1]
    n_rel = len(relevant_primes)

    # Precompute residue lists: for each prime index, list of numbers per residue
    residue_lists = []  # residue_lists[i][r] = list of x in [0,N] with x % p_i == r
    # Precompute x % p for each relevant prime as a numpy array
    prime_residue_arrays = []  # prime_residue_arrays[i] = np.array([x % p_i for x in range(N+1)])
    
    for p in relevant_primes:
        arr = np.arange(N + 1) % p
        prime_residue_arrays.append(arr)
        rl = [[] for _ in range(p)]
        for x in range(N + 1):
            rl[x % p].append(x)
        residue_lists.append(rl)

    # Per-prime residue counts and number of *present* (nonzero) residues.
    counts = [np.zeros(p, dtype=np.int32) for p in relevant_primes]
    present = [0] * n_rel
    in_set = np.ones(N + 1, dtype=bool)

    # Initialise from the full set [0, N].
    for i, p in enumerate(relevant_primes):
        arr = prime_residue_arrays[i]
        # Count residues
        for r in range(p):
            counts[i][r] = np.sum(arr == r)
        present[i] = p  # all residues present initially

    L = N + 1

    def remove(x):
        nonlocal L
        in_set[x] = False
        L -= 1
        for i, p in enumerate(relevant_primes):
            r = x % p
            counts[i][r] -= 1
            if counts[i][r] == 0:
                present[i] -= 1

    def add(x):
        nonlocal L
        in_set[x] = True
        L += 1
        for i, p in enumerate(relevant_primes):
            r = x % p
            if counts[i][r] == 0:
                present[i] += 1
            counts[i][r] += 1

    # ---- Phase 1: greedy removal until admissible ----
    # For the smallest violating prime, drop the minimum-count residue class.
    while now() - start < DEADLINE - 40:
        viol_idx = None
        for i, p in enumerate(relevant_primes):
            if p > L:
                break
            if present[i] == p:  # all residues present -> must break this prime
                viol_idx = i
                break
        if viol_idx is None:
            break
        p = relevant_primes[viol_idx]
        c = counts[viol_idx]
        r = int(np.argmin(c))
        # Get all numbers with this residue that are still in the set
        candidates = residue_lists[viol_idx][r]
        for x in candidates:
            if in_set[x]:
                remove(x)

    # ---- Phase 2: add-back local improvement ----
    # Adding x is safe iff no prime p <= L+1 has exactly one missing residue
    # equal to x % p.
    def safe(x):
        limit = L + 1
        for i, p in enumerate(relevant_primes):
            if p > limit:
                break
            if present[i] == p - 1 and counts[i][x % p] == 0:
                return False
        return True

    # Multiple passes: each pass may unlock further additions.
    for _ in range(500):
        if now() - start > DEADLINE - 25:
            break
        added = False
        for x in range(N + 1):
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
        for i in range(n_rel):
            p = relevant_primes[i]
            counts[i] = np.zeros(p, dtype=np.int32)
            present[i] = 0
        in_set = np.zeros(N + 1, dtype=bool)
        for x in res:
            in_set[x] = True
            for i, p in enumerate(relevant_primes):
                r = x % p
                if counts[i][r] == 0:
                    present[i] += 1
                counts[i][r] += 1
        L = len(res)
        while now() - start < DEADLINE - 5:
            viol_idx = None
            for i, p in enumerate(relevant_primes):
                if p > L:
                    break
                if present[i] == p:
                    viol_idx = i
                    break
            if viol_idx is None:
                break
            p = relevant_primes[viol_idx]
            c = counts[viol_idx]
            r = int(np.argmin(c))
            candidates = residue_lists[viol_idx][r]
            for x in candidates:
                if in_set[x]:
                    remove(x)
        res = [x for x in range(N + 1) if in_set[x]]

    return res
