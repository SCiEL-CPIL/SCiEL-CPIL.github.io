"""Sumset/difference-set exponent lower bound.

Strategy (same family as the seed, refined):
  * Build large near-Sidon / near-Golomb sets (greedy + Welch + Mian-Chowla).
  * Score with a FAST bitset scorer (Python ints as bitmasks -> C-speed shifts).
  * Refine with a SMOOTH-SURROGATE local search: a continuous soft-max penalty
    for sum/difference collisions (log-sum-exp) guides moves, with exact
    bitset scoring only when a move looks promising.
  * Final exact local polish + prime scaling + truncation.
"""
import time
import math
import random
import numpy as np

_T0 = time.time()
DEADLINE = _T0 + 138.0  # hard kill is 150s; keep a 12s safety margin

def _time_left():
    return DEADLINE - time.time()

# ---------------------------------------------------------------------------
# Fast scoring via bitmasks (Python ints as arbitrary-precision bitsets)
# ---------------------------------------------------------------------------
def _bit_popcount(x):
    return x.bit_count()

def _score_set(U):
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = U[-1]
    if max_u == 0 or n < 2:
        return 1.0

    # sumset: for each a, OR of (mask << a); bit k set iff some pair sums to k
    smask = 0
    for a in U:
        smask |= (mask << a) if (mask := _mk(U)) else 0
    # (recompute mask once)
    mask = 0
    for v in U:
        mask |= (1 << v)
    smask = 0
    for a in U:
        smask |= (mask << a)
    sumset_size = _bit_popcount(smask)

    # difference set: shift mask by max_u, convolve with mask
    dmask = 0
    for a in U:
        dmask |= (mask << (a + max_u))
    diffset_size = _bit_popcount(dmask)

    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def _mk(U):
    m = 0
    for v in U:
        m |= (1 << v)
    return m

def _score_set2(U):
    """Clean bitset scorer (used everywhere)."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = U[-1]
    if max_u == 0 or n < 2:
        return 1.0
    mask = 0
    for v in U:
        mask |= (1 << v)
    smask = 0
    for a in U:
        smask |= (mask << a)
    sumset_size = _bit_popcount(smask)
    dmask = 0
    for a in U:
        dmask |= (mask << (a + max_u))
    diffset_size = _bit_popcount(dmask)
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

# ---------------------------------------------------------------------------
# Smooth surrogate: soft-max collision penalty (log-sum-exp)
#   For a candidate value x, we measure how many existing pairs already produce
#   the sums x+a and the differences x-a (a in U).  A smooth penalty
#   log(1 + sum exp(w*count)) rewards adding x with few collisions.
# ---------------------------------------------------------------------------
def _smooth_pen(U, x, w=1.0):
    """Return a smooth 'collision cost' for adding x to set U (lower=better).
    Uses a soft count of how many sums x+a and diffs x-a collide with existing
    pairwise sums/differences.  Approximate but smooth and cheap."""
    # Build hash sets of existing sums and diffs (cached by caller for speed).
    pass  # replaced by caller-provided sets

def _build_sumdiff(U):
    sums = set()
    diffs = set()
    for i in range(len(U)):
        a = U[i]
        for j in range(i, len(U)):
            sums.add(a + U[j])
        for j in range(len(U)):
            diffs.add(a - U[j])
    return sums, diffs

def _smooth_add_cost(U, sums, diffs, x, w=0.6):
    """Smooth cost for adding x: soft count of new sum/diff collisions."""
    c = 0.0
    for a in U:
        if x + a in sums:
            c += math.exp(w)
        if x - a in diffs:
            c += math.exp(w)
        if a - x in diffs:
            c += math.exp(w)
    return math.log(1.0 + c)

# ---------------------------------------------------------------------------
# Constructions
# ---------------------------------------------------------------------------
def _mian_chowla(n):
    seq = [0]
    sums = {0}
    for _ in range(1, n):
        x = seq[-1] + 1
        while any(x + a in sums for a in seq):
            x += 1
        for a in seq:
            sums.add(a + x)
        seq.append(x)
    return seq

def _welch(p):
    if p < 3:
        return [0]
    S = []
    for i in range(p):
        S.append(i * p + (i * i) % p)
    return sorted(set(S))

def _greedy_golomb(n, max_val=None, seed=0):
    if n <= 1:
        return [0]
    if max_val is None:
        max_val = n * n * 3
    rng = random.Random(seed)
    ruler = [0]
    diffs = set()
    for _ in range(1, n):
        lo = ruler[-1] + 1
        best = None
        for pos in range(lo, max_val + 1):
            ok = True
            for a in ruler:
                if (pos - a) in diffs:
                    ok = False
                    break
            if ok:
                best = pos
                break
        if best is None:
            for _ in range(3000):
                pos = rng.randint(lo, max_val)
                if all((pos - a) not in diffs for a in ruler):
                    best = pos
                    break
        if best is None:
            break
        for a in ruler:
            diffs.add(best - a)
        ruler.append(best)
    return ruler

def _greedy_sidon(n, max_val=None, seed=0):
    if n <= 1:
        return [0]
    if max_val is None:
        max_val = n * n * 4
    rng = random.Random(seed)
    seq = [0]
    sums = {0}
    for _ in range(1, n):
        lo = seq[-1] + 1
        best = None
        for pos in range(lo, max_val + 1):
            if all(a + pos not in sums for a in seq):
                best = pos
                break
        if best is None:
            for _ in range(3000):
                pos = rng.randint(lo, max_val)
                if all(a + pos not in sums for a in seq):
                    best = pos
                    break
        if best is None:
            break
        for a in seq:
            sums.add(a + best)
        seq.append(best)
    return seq

# ---------------------------------------------------------------------------
# Smooth-surrogate local search
# ---------------------------------------------------------------------------
def _local_search(U, max_iters=600, deadline=None, seed=0, w=0.6):
    if deadline is None:
        deadline = DEADLINE
    rng = random.Random(seed)
    U = sorted(set(U))
    best_score = _score_set2(U)
    best_U = U[:]
    max_u = U[-1] if U else 0
    upper = max_u + max(60, max_u // 4)

    for it in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        n = len(U)
        sums, diffs = _build_sumdiff(U)

        # ---- add a candidate chosen by smooth cost ----
        if n < 400 and rng.random() < 0.5:
            cands = []
            for _ in range(200):
                x = rng.randint(1, upper)
                if x in U:
                    continue
                cands.append((x, _smooth_add_cost(U, sums, diffs, x, w)))
            cands.sort(key=lambda t: t[1])
            for x, _ in cands[:3]:
                new_U = sorted(U + [x])
                s = _score_set2(new_U)
                if s > best_score + 1e-12:
                    best_score, best_U = s, new_U
                    improved = True
                    break

        # ---- remove a random nonzero element ----
        if not improved and n > 3 and rng.random() < 0.4:
            idx = rng.randint(1, n - 1)
            new_U = U[:idx] + U[idx + 1:]
            s = _score_set2(new_U)
            if s > best_score + 1e-12:
                best_score, best_U = s, new_U
                improved = True

        # ---- swap a random element to a nearby value ----
        if not improved and n > 2:
            idx = rng.randint(1, n - 1)
            old = U[idx]
            deltas = rng.sample([-60, -30, -15, -8, -4, -2, -1, 1, 2, 4, 8, 15, 30, 60, 100], 6)
            for d in deltas:
                nv = old + d
                if nv < 1 or nv > upper or nv in U:
                    continue
                new_U = sorted(U[:idx] + [nv] + U[idx + 1:])
                s = _score_set2(new_U)
                if s > best_score + 1e-12:
                    best_score, best_U = s, new_U
                    improved = True
                    break

        U = best_U[:]
        max_u = U[-1] if U else 0
        if not improved and it % 40 == 0:
            upper += max(50, max_u // 10)

    return best_U, best_score

def _local_search_exact(U, max_iters=400, deadline=None, seed=0):
    """Exact (non-smooth) local polish: add/remove/swap with full scoring."""
    if deadline is None:
        deadline = DEADLINE
    rng = random.Random(seed)
    U = sorted(set(U))
    best_score = _score_set2(U)
    best_U = U[:]
    max_u = U[-1] if U else 0
    upper = max_u + max(40, max_u // 6)

    for it in range(max_iters):
        if time.time() > deadline:
            break
        improved = False
        n = len(U)

        if n < 400 and rng.random() < 0.45:
            for _ in range(30):
                x = rng.randint(1, upper)
                if x in U:
                    continue
                new_U = sorted(U + [x])
                s = _score_set2(new_U)
                if s > best_score + 1e-12:
                    best_score, best_U = s, new_U
                    improved = True
                    break

        if not improved and n > 3 and rng.random() < 0.4:
            idx = rng.randint(1, n - 1)
            new_U = U[:idx] + U[idx + 1:]
            s = _score_set2(new_U)
            if s > best_score + 1e-12:
                best_score, best_U = s, new_U
                improved = True

        if not improved and n > 2:
            idx = rng.randint(1, n - 1)
            old = U[idx]
            for d in rng.sample([-30, -15, -8, -4, -2, -1, 1, 2, 4, 8, 15, 30], 5):
                nv = old + d
                if nv < 1 or nv > upper or nv in U:
                    continue
                new_U = sorted(U[:idx] + [nv] + U[idx + 1:])
                s = _score_set2(new_U)
                if s > best_score + 1e-12:
                    best_score, best_U = s, new_U
                    improved = True
                    break

        U = best_U[:]
        max_u = U[-1] if U else 0
        if not improved and it % 50 == 0:
            upper += max(40, max_u // 12)

    return best_U, best_score

def _scale(U, p):
    return sorted(p * v for v in U)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def construct():
    start = time.time()
    deadline = DEADLINE

    best_U = [0, 1, 3]
    best_score = _score_set2(best_U)

    def consider(U):
        nonlocal best_U, best_score
        if not U or 0 not in U:
            return
        U = sorted(set(U))
        s = _score_set2(U)
        if s > best_score:
            best_score = s
            best_U = U
        return s

    candidates = []

    # Mian-Chowla (known good small-span near-Sidon)
    for n in [8, 12, 16, 20, 24, 30]:
        if time.time() > deadline - 25:
            break
        candidates.append(_mian_chowla(n))

    # Welch Sidon for primes
    for p in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293]:
        if time.time() > deadline - 25:
            break
        candidates.append(_welch(p))

    # greedy Golomb rulers
    for n in [10, 20, 40, 80, 120, 180, 260, 360, 500, 700, 1000]:
        if time.time() > deadline - 25:
            break
        gr = _greedy_golomb(n, max_val=n * n * 3, seed=1)
        if len(gr) >= n:
            candidates.append(gr)

    # greedy Sidon
    for n in [10, 20, 40, 80, 150, 250, 400]:
        if time.time() > deadline - 25:
            break
        ss = _greedy_sidon(n, max_val=n * n * 4, seed=2)
        if len(ss) >= n:
            candidates.append(ss)

    # ---- score & rank ----
    scored = []
    for U in candidates:
        if time.time() > deadline - 25:
            break
        s = _score_set2(U)
        scored.append((s, U))
    scored.sort(key=lambda t: -t[0])

    for s, U in scored[:15]:
        if time.time() > deadline - 12:
            break
        consider(U)

    # ---- smooth-surrogate local search on top candidates ----
    for s, U in scored[:8]:
        if time.time() > deadline - 8:
            break
        iU, is_ = _local_search(U, max_iters=500, deadline=deadline, seed=7, w=0.6)
        if is_ > best_score:
            best_score = is_
            best_U = iU

    # ---- prime scaling of the best ----
    if time.time() < deadline - 6 and best_U:
        for p in [2, 3, 5, 7]:
            if time.time() > deadline - 4:
                break
            consider(_scale(best_U, p))

    # ---- final exact polish ----
    if time.time() < deadline - 3:
        iU, is_ = _local_search_exact(best_U, max_iters=600, deadline=deadline, seed=11)
        if is_ > best_score:
            best_score = is_
            best_U = iU

    return sorted(set(best_U))


if __name__ == "__main__":
    U = construct()
    print("size:", len(U), "max:", max(U) if U else 0, "score:", _score_set2(U))
