"""Prime residue covering with reciprocal budget 1."""
import time
from fractions import Fraction

DEADLINE = 150.0  # wall-clock guard (evaluator kills at 150s)
CAP = 100000

# --- Precompute primes up to 1000 and their reciprocals (exact) ---
def _sieve(n):
    is_p = [True] * (n + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_p[i]:
            for j in range(i * i, n + 1, i):
                is_p[j] = False
    return [i for i in range(2, n + 1) if is_p[i]]

PRIMES = _sieve(1000)
PINDEX = {p: i for i, p in enumerate(PRIMES)}
# reciprocal as Fraction
RECIP = {p: Fraction(1, p) for p in PRIMES}

# Bitset: a big int where bit n is 1 iff n is covered.
# We cover numbers 1..CAP.  A class (p, r) covers n with n ≡ r (mod p).
# Build a mask for each (p, r) lazily and cache.
_MASK_CACHE = {}

def _mask(p, r):
    key = (p, r)
    m = _MASK_CACHE.get(key)
    if m is not None:
        return m
    # numbers n in 1..CAP with n % p == r
    # first n >= 1
    if r == 0:
        first = p
    else:
        first = r
    if first < 1:
        first += p
    if first > CAP:
        m = 0
    else:
        # set bits first, first+p, first+2p, ... <= CAP
        # Build via shifting: bit i set for i = first + k*p
        # We'll build by loop but p can be small; use a fast method:
        # create a pattern of length p with bit r set, repeat.
        # Simpler: loop step p.
        m = 0
        n = first
        while n <= CAP:
            m |= (1 << n)
            n += p
    _MASK_CACHE[key] = m
    return m

def _coverage_L(used):
    """used: dict p -> r. Return largest L s.t. 1..L all covered."""
    m = 0
    for p, r in used.items():
        m |= _mask(p, r)
    # find first zero bit starting at 1
    # L = number of consecutive 1s from bit 1
    # Check bit 1,2,...
    # Efficient: find first zero bit at or above 1.
    # (~m) has 1s where m has 0. We want lowest set bit of (~m) that is >= 1.
    inv = (~m) & ((1 << (CAP + 1)) - 1)
    # lowest set bit position
    if inv & 1:
        # bit 0 is zero in m? bit 0 corresponds to n=0, irrelevant.
        # we start at n=1
        pass
    # find lowest set bit of inv that is >=1
    # clear bit 0
    inv1 = inv & ~1
    if inv1 == 0:
        return CAP
    # lowest set bit
    lsb = inv1 & (-inv1)
    pos = lsb.bit_length() - 1
    # pos is the first uncovered n >= 1
    return pos - 1

def _budget_ok(used):
    s = Fraction(0)
    for p in used:
        s += RECIP[p]
    return s <= 1

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE - 2.0  # safety margin

    # Greedy seed: pick primes to maximize coverage with budget 1.
    # Start with a known-good baseline: primes 13..~ (as seed) but we'll
    # build greedily by coverage gain per cost.
    used = {}
    # budget remaining
    budget = Fraction(1)

    # We'll do a greedy: repeatedly add (p,r) that gives max new coverage
    # among 1..some window, per unit cost.
    # To keep it fast, evaluate coverage on a smaller window first, say 1..2000.
    WIN = 2000

    def _mask_win(p, r, W=WIN):
        # restricted to 1..W
        if r == 0:
            first = p
        else:
            first = r
        if first < 1:
            first += p
        if first > W:
            return 0
        m = 0
        n = first
        while n <= W:
            m |= (1 << n)
            n += p
        return m

    current_cover = 0
    # candidate primes: all primes <=1000
    # We'll consider adding each prime with best residue.
    # For speed, iterate a few rounds.
    for _round in range(60):
        if time.time() > deadline:
            break
        best_gain = 0
        best_pr = None
        # For each unused prime, find residue r maximizing new bits in window
        for p in PRIMES:
            if p in used:
                continue
            # try residues; to limit cost, try a few: 0..min(p-1, 64) plus random?
            # Actually residue choice matters; we'll scan all r if p small else sample.
            if p <= 64:
            # small primes: scan all
                range_r = range(p)
            else:
                range_r = range(0, p, max(1, p // 32))
            for r in range_r:
                m = _mask_win(p, r)
                gain = bin(m & ~current_cover).count("1")
                # cost = 1/p
                # value = gain * p  (gain per unit cost approx)
                val = gain * p
                if val > best_gain:
                    best_gain = val
                    best_pr = (p, r)
        if best_pr is None or best_gain <= 0:
            break
        p, r = best_pr
        # check budget
        if budget - RECIP[p] < 0:
            continue
        used[p] = r
        budget -= RECIP[p]
        current_cover |= _mask_win(p, r)

    # Now local search: try to swap a class for another to extend coverage.
    # Objective: maximize L (coverage of 1..L).
    def _score(used):
        return _coverage_L(used)

    best_used = dict(used)
    best_score = _score(best_used)

    # Local search steps:
    # 1) Try to change residue of an existing prime.
    # 2) Try to swap one prime for another.
    # Use bitset for speed on full CAP.
    # Precompute masks for current used on full CAP.
    def _full_mask(p, r):
        return _mask(p, r)

    # We'll maintain a dict of masks.
    # Represent used as dict p->r.
    # For local search, evaluate _coverage_L which is O(#primes) big-int ORs + one bit op.

    steps = 0
    while time.time() < deadline:
        steps += 1
        improved = False

        # --- Residue change ---
        for p in list(used.keys()):
            if time.time() > deadline:
                break
            old_r = used[p]
            # try new residues; sample if large
            if p <= 64:
                cands = range(p)
            else:
                cands = [0, 1, p - 1] + list(range(0, p, max(1, p // 16)))
            for r in cands:
                if r == old_r:
                    continue
                used[p] = r
                s = _score(used)
                if s > best_score:
                    best_score = s
                    best_used = dict(used)
                    improved = True
                    break
                else:
                    used[p] = old_r
            # reset to old if not changed
            used[p] = old_r

        # --- Swap: remove one, add another ---
        if not improved:
            for p_remove in list(used.keys()):
                if time.time() > deadline:
                    break
                # temporarily remove
                r_remove = used.pop(p_remove)
                # try adding each unused prime with best residue
                for p_add in PRIMES:
                    if p_add in used:
                        continue
                    if RECIP[p_add] > budget + RECIP[p_remove]:
                        continue
                    # best residue for p_add given current coverage
                    # evaluate a few residues
                    if p_add <= 64:
                        cands = range(p_add)
                    else:
                        cands = [0, 1, p_add - 1] + list(range(0, p_add, max(1, p_add // 16)))
                    best_r = None
                    best_s = -1
                    for r in cands:
                        used[p_add] = r
                        s = _score(used)
                        if s > best_s:
                            best_s = s
                            best_r = r
                    # remove p_add
                    if p_add in used:
                        del used[p_add]
                    if best_r is not None and best_s > best_score:
                        used[p_add] = best_r
                        best_score = best_s
                        best_used = dict(used)
                        improved = True
                        break
                # restore p_remove
                used[p_remove] = r_remove
                if improved:
                    break

        if not improved:
            # no improvement in this pass; try a random perturbation to escape local min
            if used:
                # pick a random prime in used, change its residue randomly
                import random
                p = random.choice(list(used.keys()))
                old_r = used[p]
                new_r = random.randrange(p)
                used[p] = new_r
                s = _score(used)
                if s >= best_score:
                    best_used = dict(used)
                    best_score = s
                else:
                    used[p] = old_r
            else:
                break

    # Final: return best_used
    # Ensure validity
    if not _budget_ok(best_used):
        # fallback to seed
        primes = [p for p in range(13, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
        return [[p, (i + 1) % p] for i, p in enumerate(primes)]

    return [[p, r] for p, r in best_used.items()]
