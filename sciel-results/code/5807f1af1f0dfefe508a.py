import time
import random

N = 5000
_DEADLINE = time.time() + 320.0  # 330s wall-clock, 10s safety margin
_RNG = random.Random(20240517)

# EVOLVE-BLOCK-START
def _primes(limit):
    """Return list of primes <= limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, ok in enumerate(sieve) if ok]


def _build_primes_by_len():
    """primes_by_len[k] = list of primes <= k, for k in 0..N+1."""
    primes = _primes(N + 1)
    pb = [[] for _ in range(N + 2)]
    idx = 0
    for k in range(2, N + 2):
        while idx < len(primes) and primes[idx] <= k:
            pb[k].append(primes[idx])
            idx += 1
    return pb


def _greedy_initial(primes_by_len):
    """Start from full range, repair admissibility by removing the least-populated
    residue class for each offending prime (bitmask state for speed)."""
    points = list(range(N + 1))
    present = [True] * (N + 1)
    active = N + 1

    # precompute residue for each prime
    # we recompute as needed; store per-prime residue arrays lazily
    residue_cache = {}

    def residues(p):
        if p not in residue_cache:
            residue_cache[p] = [v % p for v in range(N + 1)]
        return residue_cache[p]

    # initial counts per prime
    # We'll maintain counts[p][r] and covered_mask[p]
    # Start: build for all primes up to N+1
    all_primes = primes_by_len[N + 1]
    counts = {}
    covered = {}
    for p in all_primes:
        cnt = [0] * p
        r = residues(p)
        for v in range(N + 1):
            cnt[r[v]] += 1
        counts[p] = cnt
        mask = 0
        for rr in range(p):
            if cnt[rr] > 0:
                mask |= (1 << rr)
        covered[p] = mask

    # repair loop
    while True:
        current_len = active
        if current_len < 2:
            break
        offending = None
        for p in primes_by_len[current_len]:
            if covered[p] == (1 << p) - 1:
                offending = p
                break
        if offending is None:
            break
        p = offending
        cnt = counts[p]
        r = residues(p)
        # find least-populated class
        min_count = min(cnt)
        target = cnt.index(min_count)
        # remove all elements with residue == target
        for v in range(N + 1):
            if present[v] and r[v] == target:
                present[v] = False
                active -= 1
                # update all primes' counts/masks
                for q in all_primes:
                    rv = v % q
                    counts[q][rv] -= 1
                    if counts[q][rv] == 0:
                        covered[q] &= ~(1 << rv)
    return points, present, active


def _local_search(points, present, active, primes_by_len, residue_cache):
    """Swap-based local search to improve tuple size / robustness.
    A swap removes element a and adds element b (b not present, a present).
    Accept if it keeps admissibility and increases a robustness score.
    """
    all_primes = primes_by_len[N + 1]
    # build counts/covered from current present
    counts = {}
    covered = {}
    for p in all_primes:
        if p not in residue_cache:
            residue_cache[p] = [v % p for v in range(N + 1)]
        r = residue_cache[p]
        cnt = [0] * p
        for v in range(N + 1):
            if present[v]:
                cnt[r[v]] += 1
        counts[p] = cnt
        mask = 0
        for rr in range(p):
            if cnt[rr] > 0:
                mask |= (1 << rr)
        covered[p] = mask

    def robustness_score():
        # min over offending-adjacent primes of min class count
        # We want to maximize the minimum "slack". Define slack for prime p (p<=len)
        # as min_r counts[p][r]. Admissibility requires some r with count 0.
        # For robustness, maximize the min over p<=len of (min count among classes)
        # but that's 0 when admissible. Instead, measure how "safe" we are:
        # the number of empty classes per prime. More empty = more robust.
        score = 0
        L = active
        for p in primes_by_len[L]:
            mask = covered[p]
            # number of empty classes
            empty = p - bin(mask).count('1')
            score += empty
        return score

    best_score = robustness_score()
    present_list = present[:]
    active_list = active
    best_present = present[:]
    best_active = active

    # candidate additions: elements not present
    # We'll do random swaps with a time budget
    start = time.time()
    time_budget = 250.0  # leave margin
    # Precompute residue for all primes for quick access
    # Do a number of passes
    not_present = [v for v in range(N + 1) if not present[v]]
    # We'll randomly pick a to remove and b to add
    # To be admissible after swap: for every p<=len', some class empty.
    # Removing a decreases counts, adding b increases counts.
    # Simplest: try swap, recompute affected, check admissibility.
    # For speed, only consider swaps that keep len the same (remove+add).
    # Admissibility check: after swap, for each p<=len, covered mask != full.
    # We can update masks incrementally.

    # Precompute for each v the list of (p, residue) it affects
    # v affects prime p if p <= N+1 (all). residue = v % p.
    # For a swap (remove a, add b), for each p:
    #   counts[p][a%p] -= 1, counts[p][b%p] += 1
    #   update covered[p] accordingly.
    # Then check admissibility for p <= len (len unchanged).
    # Also the robustness score changes.

    # For speed, precompute v % p for all p in a dict per v? Too big.
    # Instead compute on the fly; p list is small (~669 primes up to 5001).
    # Each swap touches ~669 primes. With 300s budget, can do ~100k swaps.

    # We'll iterate: random a from present, random b from not_present.
    # To bias toward improvements, we can also try "add only" if len can grow,
    # but len growth requires the tuple to become admissible at new len, which
    # is harder. Stick with same-len swaps to improve robustness, then try
    # single-add to grow len.

    # Actually, the primary objective is to maximize len. The greedy already
    # minimizes removals. Local search with same-len swaps won't increase len.
    # To increase len, we need to add an element and possibly remove fewer
    # elsewhere. That's a different move. Let's do:
    #   Move 1: add b (not present) -> len+1, check admissibility at len+1.
    #   Move 2: remove a -> len-1 (never helpful for size, skip).
    #   Move 3: swap a->b (len same) to set up for an add.
    #
    # Strategy: repeatedly try to add a random not-present element; if it
    # breaks admissibility, try to fix by removing a different element
    # (net len same or +1). This is complex.
    #
    # Simpler high-value move: try to add b. If admissible at new len, accept
    # (len grows). If not, find an a to remove such that after add+remove
    # (len same) we're admissible AND robustness improves, then later retry add.
    #
    # Given time, I'll implement: random add attempts, and random swaps to
    # improve robustness (which may enable future adds). Track best len.

    best_len = active
    # We'll maintain global best (present, active) by len then robustness.
    def better(p1, a1, s1, p2, a2, s2):
        if a1 != a2:
            return a1 > a2
        return s1 > s2

    # Start
    cur_present = present[:]
    cur_active = active
    cur_score = best_score
    best_present = cur_present[:]
    best_active = cur_active
    best_score = cur_score

    # Precompute residue for all primes once
    for p in all_primes:
        if p not in residue_cache:
            residue_cache[p] = [v % p for v in range(N + 1)]

    # counts and covered for current state
    cur_counts = {p: counts[p][:] for p in all_primes}
    cur_covered = {p: covered[p] for p in all_primes}

    def apply_add(b, counts_d, covered_d):
        """Return (ok, new_counts, new_covered) after adding b. Modifies in place
        then we revert. Actually we'll do trial: compute deltas."""
        pass

    # For trial, we'll copy affected counts. To be fast, we mutate and revert.
    def trial_add(b):
        # add b
        for p in all_primes:
            r = b % p
            c = cur_counts[p][r]
            if c == 0:
                cur_covered[p] |= (1 << r)
            cur_counts[p][r] = c + 1
        new_len = cur_active + 1
        # check admissibility at new_len
        admissible = True
        for p in primes_by_len[new_len]:
            if cur_covered[p] == (1 << p) - 1:
                admissible = False
                break
        # revert
        for p in all_primes:
            r = b % p
            cur_counts[p][r] -= 1
            if cur_counts[p][r] == 0:
                cur_covered[p] &= ~(1 << r)
        return admissible

    def trial_swap(a, b):
        """Remove a, add b (len same). Return (admissible, score_delta)."""
        # apply
        for p in all_primes:
            ra = a % p
            rb = b % p
            if ra == rb:
                # same class, no change
                continue
            # decrement ra
            ca = cur_counts[p][ra]
            if ca == 1:
                cur_covered[p] &= ~(1 << ra)
            cur_counts[p][ra] = ca - 1
            # increment rb
            cb = cur_counts[p][rb]
            if cb == 0:
                cur_covered[p] |= (1 << rb)
            cur_counts[p][rb] = cb + 1
        # check admissibility at same len
        admissible = True
        L = cur_active
        for p in primes_by_len[L]:
            if cur_covered[p] == (1 << p) - 1:
                admissible = False
                break
        # compute score delta (robustness)
        # We'll compute new score and compare; but for speed, approximate:
        # revert and return admissible; caller decides.
        # revert
        for p in all_primes:
            ra = a % p
            rb = b % p
            if ra == rb:
                continue
            cur_counts[p][ra] += 1
            if cur_counts[p][ra] == 1:
                cur_covered[p] |= (1 << ra)
            cur_counts[p][rb] -= 1
            if cur_counts[p][rb] == 0:
                cur_covered[p] &= ~(1 << rb)
        return admissible

    def do_add(b):
        nonlocal cur_active, cur_score, best_active, best_score, best_present
        for p in all_primes:
            r = b % p
            c = cur_counts[p][r]
            if c == 0:
                cur_covered[p] |= (1 << r)
            cur_counts[p][r] = c + 1
        cur_present[b] = True
        cur_active += 1
        # update score
        # recompute score (cheap enough occasionally)
        cur_score = robustness_score()
        if better(cur_present, cur_active, cur_score, best_present, best_active, best_score):
            best_present = cur_present[:]
            best_active = cur_active
            best_score = cur_score

    def do_swap(a, b):
        nonlocal cur_score, best_active, best_score, best_present
        for p in all_primes:
            ra = a % p
            rb = b % p
            if ra == rb:
                continue
            ca = cur_counts[p][ra]
            if ca == 1:
                cur_covered[p] &= ~(1 << ra)
            cur_counts[p][ra] = ca - 1
            cb = cur_counts[p][rb]
            if cb == 0:
                cur_covered[p] |= (1 << rb)
            cur_counts[p][rb] = cb + 1
        cur_present[a] = False
        cur_present[b] = True
        cur_score = robustness_score()
        if better(cur_present, cur_active, cur_score, best_present, best_active, best_score):
            best_present = cur_present[:]
            best_active = cur_active
            best_score = cur_score

    # Main loop
    add_attempts = 0
    swap_attempts = 0
    while time.time() - start < time_budget:
        # Try adds
        if not_present:
            b = _RNG.choice(not_present)
            if trial_add(b):
                do_add(b)
                not_present.remove(b)
                add_attempts += 1
                continue
        # Try swaps to improve robustness
        # pick random a present, random b not present
        if not_present and cur_active > 0:
            a = _RNG.randrange(N + 1)
            while not cur_present[a]:
                a = _RNG.randrange(N + 1)
            b = _RNG.choice(not_present)
            if trial_swap(a, b):
                # accept swap if it improves robustness
                # compute score after swap (trial already reverted, so recompute)
                # We'll just accept if admissible (to explore)
                do_swap(a, b)
                swap_attempts += 1
        else:
            break

    # Restore best
    return best_present, best_active


def construct():
    primes_by_len = _build_primes_by_len()
    residue_cache = {}

    # Phase 1: greedy initial
    points, present, active = _greedy_initial(primes_by_len)
    # rebuild residue_cache (it was local to greedy)
    residue_cache = {}

    # Phase 2: local search
    if time.time() < _DEADLINE - 20:
        present, active = _local_search(points, present, active, primes_by_len, residue_cache)

    # Build result
    result = [v for v in range(N + 1) if present[v]]
    return result
# EVOLVE-BLOCK-END
