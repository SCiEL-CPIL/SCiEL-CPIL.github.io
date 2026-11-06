import math
import random
import time
from fractions import Fraction

# ---- primes up to 1000 (fixed) ----
def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start:n + 1:step] = [False] * len(sieve[start:n + 1:step])
    return [i for i, v in enumerate(sieve) if v]

_PRIMES = _primes_upto(1000)
# invertible residues per prime: 1..p-1
_INV = {p: list(range(1, p)) for p in _PRIMES}
P_INDEX = {p: i for i, p in enumerate(_PRIMES)}
NUM_PRIMES = len(_PRIMES)
# reciprocal cost as Fraction
_COST = {p: Fraction(1, p) for p in _PRIMES}
_BUDGET = Fraction(1, 1)


def _coverage_length(assign):
    """assign: list of residues (0..p-1) per prime; 0 means 'not used'.
    Return largest L such that every 1..L is covered by some class."""
    covered = [False] * 20000  # enough for L up to 10^5 cap
    for idx, r in enumerate(assign):
        if r == 0:
            continue
        p = _PRIMES[idx]
        for n in range(r, 20000, p):
            covered[n] = True
    # find largest L with all 1..L covered
    L = 0
    for n in range(1, 20000):
        if covered[n]:
            L = n
        else:
            break
    return L


def _total_cost(assign):
    c = Fraction(0)
    for idx, r in enumerate(assign):
        if r != 0:
            c += _COST[_PRIMES[idx]]
    return c


def _is_valid(assign):
    return _total_cost(assign) <= _BUDGET


# ---------- Greedy construction (randomized) ----------
def _greedy_build(seed_rng, target_L):
    """Try to build a covering of 1..target_L via greedy set-cover-like selection.
    Returns (assign, cost, L) or None."""
    # coverage count per n for 1..target_L
    cov = [0] * (target_L + 1)
    assign = [0] * NUM_PRIMES
    cost = Fraction(0)
    # worklist of primes to consider (shuffled)
    order = list(range(NUM_PRIMES))
    seed_rng.shuffle(order)

    # We'll iteratively pick a (p, r) that maximally extends the covered prefix.
    # To be efficient, we track the current covered prefix length.
    def covered_prefix(cov, tl):
        L = 0
        for n in range(1, tl + 1):
            if cov[n] > 0:
                L = n
            else:
                break
        return L

    # precompute for each prime the list of residues and the n's they cover within 1..target_L
    # For speed, for each prime p, the residues r in 1..p-1 cover arithmetic progressions.
    # We'll evaluate candidate (p, r) by how many new prefix integers it covers.
    # To keep it fast, we only consider primes where p <= target_L + something.
    active = [i for i in order if _PRIMES[i] <= target_L + 1]
    # shuffle active
    seed_rng.shuffle(active)

    # We maintain for each prime whether already used.
    used = [False] * NUM_PRIMES
    # candidate pool: for each active prime, list of residues (shuffled)
    cand = {}
    for i in active:
        p = _PRIMES[i]
        rs = _INV[p][:]
        seed_rng.shuffle(rs)
        cand[i] = rs

    # Greedy loop: while there is a gap, pick (p, r) that covers the first gap and extends most.
    # To bound runtime, limit iterations.
    max_iter = 400
    for _ in range(max_iter):
        L = covered_prefix(cov, target_L)
        if L >= target_L:
            break
        # first gap
        gap = L + 1
        if gap > target_L:
            break
        # find candidate (p, r) with p not used, r in cand[i], r == gap mod p (i.e. r in 1..p-1 and gap % p == r)
        # Actually class (p, r) covers n where n % p == r. For it to cover gap, need gap % p == r and r in 1..p-1.
        best = None
        best_score = -1
        for i in active:
            if used[i]:
                continue
            p = _PRIMES[i]
            r = gap % p
            if r == 0:
                # residue 0 not allowed (0..p-1 with 0 meaning not used; but class (p,0) is valid too?
                # Task says 0 <= r < p. But we use 0 to mean "not used". Conflict.
                # We'll only use residues 1..p-1 to avoid confusion. So skip r==0.
                continue
            if r not in cand[i]:
                continue
            # score: how far the prefix extends if we add this class
            # simulate coverage extension
            # count consecutive covered from gap onward given existing cov + this class
            # quick estimate: extend while (cov[n] > 0) or (n % p == r)
            n = gap
            ext = 0
            while n <= target_L:
                if cov[n] > 0 or (n % p == r):
                    ext += 1
                    n += 1
                else:
                    break
            if ext > best_score:
                best_score = ext
                best = (i, r)
        if best is None:
            break
        i, r = best
        p = _PRIMES[i]
        # check cost
        newcost = cost + _COST[p]
        if newcost > _BUDGET:
            # skip this prime entirely
            used[i] = True
            continue
        # apply
        assign[i] = r
        used[i] = True
        cost = newcost
        # update coverage
        for n in range(r, target_L + 1, p):
            cov[n] += 1
        # remove this residue from cand
        try:
            cand[i].remove(r)
        except ValueError:
            pass
        # if no residues left, mark used
        if not cand[i]:
            used[i] = True

    L = covered_prefix(cov, target_L)
    if L == 0:
        return None
    return assign, cost, L


# ---------- Local search: single-move (change residue of a used prime) ----------
def _local_search(assign, cost, target_L, seed_rng, time_limit, max_steps=20000):
    """Hill-climb: try changing residue of a used prime to improve coverage prefix.
    Returns (assign, cost, L)."""
    start = time.time()
    # coverage array
    cov = [0] * (target_L + 1)
    for idx, r in enumerate(assign):
        if r != 0:
            p = _PRIMES[idx]
            for n in range(r, target_L + 1, p):
                cov[n] += 1

    def prefix(cov, tl):
        L = 0
        for n in range(1, tl + 1):
            if cov[n] > 0:
                L = n
            else:
                break
        return L

    L = prefix(cov, target_L)
    used = [i for i, r in enumerate(assign) if r != 0]
    if not used:
        return assign, cost, L

    steps = 0
    while time.time() - start < time_limit and steps < max_steps:
        steps += 1
        # pick a random used prime
        i = used[seed_rng.randrange(len(used))]
        p = _PRIMES[i]
        old_r = assign[i]
        # candidate new residue: try residues that cover the current first gap, else random
        gap = L + 1
        if gap <= target_L:
            r_new = gap % p
            if r_new == 0 or r_new == old_r:
                # pick another random residue
                r_new = seed_rng.randrange(1, p)
                if r_new == old_r:
                    r_new = (r_new % (p - 1)) + 1
        else:
            r_new = seed_rng.randrange(1, p)
            if r_new == old_r:
                r_new = (r_new % (p - 1)) + 1
        # compute delta coverage quickly
        # remove old, add new
        # To keep fast, only accept if it doesn't reduce coverage of 1..L and extends beyond L
        # We'll do a bounded check.
        # temporary apply
        # remove old_r
        for n in range(old_r, target_L + 1, p):
            cov[n] -= 1
        # add new
        for n in range(r_new, target_L + 1, p):
            cov[n] += 1
        newL = prefix(cov, target_L)
        if newL >= L:
            L = newL
            assign[i] = r_new
        else:
            # revert
            for n in range(old_r, target_L + 1, p):
                cov[n] += 1
            for n in range(r_new, target_L + 1, p):
                cov[n] -= 1
    return assign, cost, L


# ---------- Add a new class (pick an unused prime + residue) ----------
def _try_add_class(assign, cost, target_L, seed_rng):
    """Try to add a new (p, r) class that improves coverage, within budget.
    Returns (assign, cost, improved) """
    cov = [0] * (target_L + 1)
    for idx, r in enumerate(assign):
        if r != 0:
            p = _PRIMES[idx]
            for n in range(r, target_L + 1, p):
                cov[n] += 1

    def prefix(cov, tl):
        L = 0
        for n in range(1, tl + 1):
            if cov[n] > 0:
                L = n
            else:
                break
        return L

    L = prefix(cov, target_L)
    gap = L + 1
    if gap > target_L:
        return assign, cost, False

    # find unused primes
    used_set = set(i for i, r in enumerate(assign) if r != 0)
    candidates = [i for i in range(NUM_PRIMES) if i not in used_set and _PRIMES[i] <= target_L + 1]
    if not candidates:
        return assign, cost, False
    seed_rng.shuffle(candidates)

    best = None
    bestL = L
    for i in candidates:
        p = _PRIMES[i]
        if cost + _COST[p] > _BUDGET:
            continue
        r = gap % p
        if r == 0:
            continue
        # simulate
        newL = L
        n = gap
        while n <= target_L:
            if cov[n] > 0 or (n % p == r):
                newL = n
                n += 1
            else:
                break
        if newL > bestL:
            bestL = newL
            best = (i, r)
        if newL >= target_L:
            break
    if best is None:
        return assign, cost, False
    i, r = best
    p = _PRIMES[i]
    assign[i] = r
    cost += _COST[p]
    return assign, cost, True


def construct():
    start = time.time()
    deadline = start + 135  # safety margin before 150s kill
    seed = 12345
    best_assign = None
    best_L = 0

    # Start with the seed program's construction as initial solution
    # The seed: i-th prime from 13 upward covers integer i
    primes_from_13 = [p for p in _PRIMES if p >= 13]
    seed_assign = [0] * NUM_PRIMES
    for i, p in enumerate(primes_from_13):
        idx = P_INDEX[p]
        seed_assign[idx] = (i + 1) % p
    seed_cost = _total_cost(seed_assign)
    seed_L = _coverage_length(seed_assign)
    if _is_valid(seed_assign):
        best_assign = seed_assign[:]
        best_L = seed_L
    else:
        # fallback: just use primes 13..
        # but cost might exceed; trim
        best_assign = [0] * NUM_PRIMES
        best_L = 0

    rng = random.Random(seed)

    # Phase 1: Greedy builds targeting increasing L
    target = max(best_L + 1, 1000)
    while time.time() < deadline - 15:
        res = _greedy_build(rng, target)
        if res is not None:
            assign, cost, L = res
            if L > best_L:
                best_assign = assign[:]
                best_L = L
            # local search to improve
            assign2, cost2, L2 = _local_search(assign, cost, target, rng, time_limit=5, max_steps=5000)
            if L2 > best_L:
                best_assign = assign2[:]
                best_L = L2
        # increase target
        target = min(target + 300, 100000)
        if target >= 100000:
            break

    # Phase 2: Refine best solution with local search + add class
    if best_assign is not None:
        target = best_L + 500
        target = min(target, 100000)
        while time.time() < deadline - 5:
            # add class
            assign, cost, improved = _try_add_class(best_assign, _total_cost(best_assign), target, rng)
            if improved:
                L = _coverage_length(assign)
                if L > best_L:
                    best_L = L
                    target = min(L + 500, 100000)
            # local search
            assign, cost, L = _local_search(assign, cost, target, rng, time_limit=8, max_steps=20000)
            if L > best_L:
                best_L = L
                target = min(L + 500, 100000)
            if target >= 100000:
                break

    # Final validation
    if best_assign is not None and _is_valid(best_assign):
        L = _coverage_length(best_assign)
        best_L = L
        return [[_PRIMES[i], r] for i, r in enumerate(best_assign) if r != 0]
    else:
        # fallback to seed
        return [[p, (i + 1) % p] for i, p in enumerate(primes_from_13)]
