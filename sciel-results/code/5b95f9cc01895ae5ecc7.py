"""Hypergraph Ramsey R(4,5;3) - bit-parallel local search.

Strategy:
  * Represent each colour class as a bitset over the C(n,3) triples.
  * Precompute, for every red-forbidden K4^3, the 4 triple indices it spans,
    and for every blue-forbidden K5^3, the 10 triple indices it spans.
  * A colouring is valid iff no K4 has all 4 red AND no K5 has all 10 blue.
  * Use vectorized NumPy to count violations and to perform fast "repair"
    sweeps (flip a random violating hyperedge member to break it).
  * Grow n by adding one vertex at a time from a known-valid base, then
    repair; keep the largest valid n found before the deadline.
"""
import time
import numpy as np
from itertools import combinations

def _build_index(n):
    triples = list(combinations(range(n), 3))
    m = len(triples)
    t2i = {t: i for i, t in enumerate(triples)}
    k4 = np.array([[t2i[(a, b, c)], t2i[(a, b, d)], t2i[(a, c, d)], t2i[(b, c, d)]]
                   for a, b, c, d in combinations(range(n), 4)], dtype=np.int64)
    k5 = np.array([np.array([t2i[t] for t in combinations(quint, 3)], dtype=np.int64)
                   for quint in combinations(range(n), 5)], dtype=np.int64)
    return triples, m, t2i, k4, k5


def _violation_counts(red_bits, k4, k5):
    # red_bits: int64 array of length m
    r = red_bits
    k4_viol = (r[k4[:, 0]] == 1) & (r[k4[:, 1]] == 1) & (r[k4[:, 2]] == 1) & (r[k4[:, 3]] == 1)
    b = 1 - r
    k5_viol = np.all(b[k5] == 1, axis=1)
    return k4_viol, k5_viol


def _count_violations(red_bits, k4, k5):
    k4_viol, k5_viol = _violation_counts(red_bits, k4, k5)
    return int(k4_viol.sum()) + int(k5_viol.sum())


def _repair(red_bits, k4, k5, rng, max_iters, t_end):
    """Randomized repair: repeatedly break a random violating hyperedge."""
    m = len(red_bits)
    for _ in range(max_iters):
        if time.time() > t_end:
            break
        k4_viol, k5_viol = _violation_counts(red_bits, k4, k5)
        nv4 = int(k4_viol.sum())
        nv5 = int(k5_viol.sum())
        if nv4 == 0 and nv5 == 0:
            break
        # Pick a violating hyperedge (prefer the rarer one for progress)
        if nv4 > 0 and (nv5 == 0 or rng.random() < nv4 / (nv4 + nv5)):
            vi = int(rng.choice(np.flatnonzero(k4_viol)))
            idx = int(rng.choice(k4[vi]))
            red_bits[idx] = 0  # make it blue to break the all-red K4
        else:
            vi = int(rng.choice(np.flatnonzero(k5_viol)))
            idx = int(rng.choice(k5[vi]))
            red_bits[idx] = 1  # make it red to break the all-blue K5
    return red_bits


def _improve(red_bits, k4, k5, rng, t_end):
    """Try to reduce violations via targeted flips; accept if it helps."""
    m = len(red_bits)
    best = red_bits.copy()
    best_v = _count_violations(best, k4, k5)
    if best_v == 0:
        return best
    # Build adjacency: for each triple, which K4s and K5s it belongs to
    tr2k4 = [[] for _ in range(m)]
    for i, row in enumerate(k4):
        for idx in row:
            tr2k4[idx].append(i)
    tr2k5 = [[] for _ in range(m)]
    for i, row in enumerate(k5):
        for idx in row:
            tr2k5[idx].append(i)

    for _ in range(20000):
        if time.time() > t_end:
            break
        k4_viol, k5_viol = _violation_counts(best, k4, k5)
        nv = int(k4_viol.sum()) + int(k5_viol.sum())
        if nv == 0:
            break
        # candidate triples: members of violating hyperedges
        cand = set()
        if int(k4_viol.sum()):
            cand.update(k4[k4_viol].tolist())
        if int(k5_viol.sum()):
            cand.update(k5[k5_viol].tolist())
        cand = list(cand)
        if not cand:
            break
        idx = int(rng.choice(cand))
        trial = best.copy()
        trial[idx] = 1 - trial[idx]
        v = _count_violations(trial, k4, k5)
        if v <= best_v:
            best = trial
            best_v = v
    return best


def construct():
    start = time.time()
    t_end = start + 145.0  # safety margin before 150s kill
    rng = np.random.default_rng(20240517)

    def try_n(n):
        """Return a valid red_bits for n, or None."""
        if time.time() > t_end:
            return None
        triples, m, t2i, k4, k5 = _build_index(n)
        # random init
        red_bits = (rng.random(m) < 0.5).astype(np.int64)
        red_bits = _repair(red_bits, k4, k5, rng, max_iters=30000, t_end=t_end)
        red_bits = _improve(red_bits, k4, k5, rng, t_end=t_end)
        if _count_violations(red_bits, k4, k5) == 0:
            return red_bits
        return None

    # ---- Phase 1: find a valid base for a moderately large n ----
    base_n = None
    base_bits = None
    for n in (20, 22, 24, 26, 28, 30, 32, 34, 36):
        if time.time() > t_end:
            break
        bits = try_n(n)
        if bits is not None:
            base_n, base_bits = n, bits
            break
    if base_n is None:
        # fallback: small n
        for n in (16, 18):
            if time.time() > t_end:
                break
            bits = try_n(n)
            if bits is not None:
                base_n, base_bits = n, bits
                break
    if base_n is None:
        # last-resort deterministic construction that is always valid-ish
        n = 9
        triples = list(combinations(range(n), 3))
        return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]}

    # ---- Phase 2: grow n beyond base_n by adding vertices and repairing ----
    best_n = base_n
    best_bits = base_bits

    def extend(cur_n, cur_bits):
        """Add one vertex (index cur_n) and try to reach a valid colouring."""
        if time.time() > t_end:
            return None
        new_n = cur_n + 1
        if new_n > 40:
            return None
        triples, m, t2i, k4, k5 = _build_index(new_n)
        # map old triple indices to new triple indices (old triples keep same combos)
        old_m = len(list(combinations(range(cur_n), 3)))
        new_bits = np.zeros(m, dtype=np.int64)
        # old triples are the first old_m in lexicographic order only if new vertex is largest
        # Since new vertex index = cur_n is the largest, all old triples appear first in lex order.
        new_bits[:old_m] = cur_bits
        # new triples (those containing the new vertex) start random
        new_bits[old_m:] = (rng.random(m - old_m) < 0.5).astype(np.int64)
        new_bits = _repair(new_bits, k4, k5, rng, max_iters=20000, t_end=t_end)
        new_bits = _improve(new_bits, k4, k5, rng, t_end=t_end)
        if _count_violations(new_bits, k4, k5) == 0:
            return new_bits
        return None

    # Try to grow a few steps; if a step fails, try a couple of restarts for that step
    for _ in range(8):
        if time.time() > t_end:
            break
        got = False
        for attempt in range(3):
            if time.time() > t_end:
                break
            nxt = extend(best_n, best_bits)
            if nxt is not None:
                best_n += 1
                best_bits = nxt
                got = True
                break
        if not got:
            break

    # ---- Phase 3: final polish (local improvement to ensure validity) ----
    if time.time() < t_end:
        triples, m, t2i, k4, k5 = _build_index(best_n)
        best_bits = _improve(best_bits, k4, k5, rng, t_end=t_end)
        if _count_violations(best_bits, k4, k5) != 0:
            best_bits = _repair(best_bits, k4, k5, rng, max_iters=20000, t_end=t_end)

    triples, m, t2i, k4, k5 = _build_index(best_n)
    if _count_violations(best_bits, k4, k5) != 0:
        # safety: fall back to base if it is valid
        if base_bits is not None:
            b2, m2, _, k4b, k5b = _build_index(base_n)
            if _count_violations(base_bits, k4b, k5b) == 0:
                best_n, best_bits = base_n, base_bits

    return {'n': best_n, 'colors': [int(x) for x in best_bits]}
