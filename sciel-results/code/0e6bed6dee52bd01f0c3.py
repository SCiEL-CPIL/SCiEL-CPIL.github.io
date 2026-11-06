"""Find a maximal dissociated (all reciprocal subset sums distinct) subset of [1,22]."""
import time
import random
import itertools
from fractions import Fraction

# ---------- exact verification ----------
def _is_dissociated(S):
    """True iff all 2^|S| reciprocal subset sums are pairwise distinct (incl. empty sum)."""
    n = len(S)
    if n == 0:
        return True
    # base values
    base = [Fraction(1, s) for s in S]
    # subset sums via iterative doubling (exact, no floating error)
    sums = [Fraction(0)]
    for b in base:
        # add b to each existing sum
        sums = sums + [x + b for x in sums]
    # check all distinct
    return len(set(sums)) == len(sums)


def _is_dissociated_fast(S):
    """Faster check using integer-scaled sums to avoid Fraction overhead when possible.
    We scale by L = lcm(S) so each 1/s becomes L/s (integer). Then subset sums are integers.
    """
    n = len(S)
    if n == 0:
        return True
    # compute lcm
    from math import gcd
    L = 1
    for s in S:
        L = L * s // gcd(L, s)
    vals = [L // s for s in S]
    # subset sums as integers
    sums = [0]
    for v in vals:
        sums = sums + [x + v for x in sums]
    return len(set(sums)) == len(sums)


# ---------- search ----------
def construct():
    t0 = time.time()
    DEADLINE = t0 + 80.0  # safety margin before 90s kill

    # Precompute all candidate subsets of [1,22] grouped by size for quick access.
    universe = list(range(1, 23))

    # Start with a known good seed (9 elements) and try to grow.
    # We'll do a randomized local search: maintain a dissociated set, try adding
    # a random element, if it breaks dissociation try removing one to repair, etc.

    best = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_len = len(best)

    rng = random.Random(12345)

    # Helper: try to add element e to set S; return True if S+e is dissociated
    def can_add(S, e):
        if e in S:
            return False
        return _is_dissociated_fast(S + [e])

    # Helper: greedy grow from a starting set
    def greedy_grow(start):
        S = list(start)
        S.sort()
        # try adding each candidate not in S, in random order, repeatedly
        changed = True
        while changed:
            changed = False
            candidates = [x for x in universe if x not in S]
            rng.shuffle(candidates)
            for e in candidates:
                if time.time() > DEADLINE - 5:
                    return S
                if can_add(S, e):
                    S.append(e)
                    S.sort()
                    changed = True
        return S

    # Helper: local search with add/remove moves
    def local_search(start, max_iters=2000):
        S = list(start)
        S.sort()
        cur_len = len(S)
        it = 0
        while it < max_iters and time.time() < DEADLINE - 5:
            it += 1
            # Try add
            candidates = [x for x in universe if x not in S]
            rng.shuffle(candidates)
            added = False
            for e in candidates:
                if can_add(S, e):
                    S.append(e)
                    S.sort()
                    cur_len += 1
                    added = True
                    break
            if added:
                continue
            # Try remove one, then add two (2-opt style)
            if len(S) >= 3:
                # pick a random element to remove
                idx = rng.randrange(len(S))
                removed = S[idx]
                S2 = S[:idx] + S[idx+1:]
                # try to add two elements to S2
                cands = [x for x in universe if x not in S2 and x != removed]
                rng.shuffle(cands)
                improved = False
                for i in range(min(len(cands), 30)):
                    e1 = cands[i]
                    if can_add(S2, e1):
                        S3 = S2 + [e1]
                        S3.sort()
                        # try add one more
                        cands2 = [x for x in universe if x not in S3]
                        rng.shuffle(cands2)
                        for j in range(min(len(cands2), 30)):
                            e2 = cands2[j]
                            if can_add(S3, e2):
                                S4 = S3 + [e2]
                                S4.sort()
                                S = S4
                                cur_len = len(S)
                                improved = True
                                break
                        if not improved:
                            # maybe just one add
                            S = S3
                            cur_len = len(S)
                            improved = True
                        break
                if improved:
                    continue
            # Try swap: remove one, add one
            if len(S) >= 2:
                idx = rng.randrange(len(S))
                removed = S[idx]
                S2 = S[:idx] + S[idx+1:]
                cands = [x for x in universe if x not in S2 and x != removed]
                rng.shuffle(cands)
                for e in cands:
                    if can_add(S2, e):
                        S = S2 + [e]
                        S.sort()
                        break
        return S

    # Try multiple random restarts
    restarts = 0
    while time.time() < DEADLINE - 10:
        restarts += 1
        # Random starting set of size 8-11
        k = rng.randint(8, 11)
        start = rng.sample(universe, k)
        # Greedy grow
        S = greedy_grow(start)
        if len(S) > best_len:
            best = S
            best_len = len(S)
        # Local search
        S = local_search(S, max_iters=500)
        if len(S) > best_len:
            best = S
            best_len = len(S)
        # If we reached 12 or more, try to verify and maybe push further
        if best_len >= 12:
            # Do a focused search to try for 13
            S = greedy_grow(best)
            if len(S) > best_len:
                best = S
                best_len = len(S)
            S = local_search(S, max_iters=1000)
            if len(S) > best_len:
                best = S
                best_len = len(S)

    # Final verification of best
    if not _is_dissociated(best):
        # Fallback to known good
        best = [1, 2, 3, 5, 7, 11, 13, 17, 19]

    return best
