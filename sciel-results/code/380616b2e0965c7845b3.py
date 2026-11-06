"""Randomized greedy packing + local search for linear 3-graphs on 16 points.

Goal: minimize the independence number (largest point set containing no triple)
subject to the linearity constraint (any two triples share at most one point).
"""
import time
import random
from itertools import combinations

# Hard wall-clock guard (evaluator kills at 150s; leave a safety margin).
_START = time.time()
_DEADLINE = _START + 140.0

N = 16
ALL_TRIPLES = list(combinations(range(N), 3))
# Precompute intersection sizes between every pair of triples.
_PAIR_INTER = {}
for i in range(len(ALL_TRIPLES)):
    a = ALL_TRIPLES[i]
    sa = set(a)
    for j in range(i + 1, len(ALL_TRIPLES)):
        b = ALL_TRIPLES[j]
        inter = len(sa & set(b))
        _PAIR_INTER[(i, j)] = inter
        _PAIR_INTER[(j, i)] = inter


def _alpha(triples):
    """Independence number: largest subset of points containing no triple."""
    if not triples:
        return N
    # Fast check for alpha <= 4: if every 5-subset contains a triple.
    triple_sets = [set(t) for t in triples]
    for s in combinations(range(N), 5):
        ok = False
        for ts in triple_sets:
            if len(ts & set(s)) == 3:
                ok = True
                break
        if not ok:
            return 5  # at least 5
    # Exact via DP over subsets (2^16 = 65536 states).
    size = 1 << N
    is_triple = [False] * size
    for ts in triple_sets:
        m = 0
        for p in ts:
            m |= 1 << p
        is_triple[m] = True
    # best[mask] = largest independent subset of mask.
    best = [0] * size
    for mask in range(1, size):
        if is_triple[mask]:
            continue
        low = mask & (-mask)
        rest = mask ^ low
        # Independent subset either excludes the lowest bit, or includes it
        # and excludes its two partners (mask out the whole triple if present).
        val_without = best[rest]
        val_with = 1 + best[rest]
        # If the lowest bit is part of a triple inside mask, including it
        # requires removing the other two points of that triple.
        lb = low.bit_length() - 1
        for ts in triple_sets:
            if lb in ts:
                others = 0
                for p in ts:
                    if p != lb:
                        others |= 1 << p
                if (mask & others) == others:
                    val_with = 1 + best[rest & ~others]
                    break
        best[mask] = max(val_without, val_with)
    return best[size - 1]


def _valid(triples):
    for i in range(len(triples)):
        for j in range(i + 1, len(triples)):
            if _PAIR_INTER[(i, j)] > 1:
                return False
    return True


def _greedy(rng, min_size=3, max_size=3):
    """Randomized greedy: pick random triples, keep if linear with chosen."""
    order = list(range(len(ALL_TRIPLES)))
    rng.shuffle(order)
    chosen = []
    for idx in order:
        if len(chosen) >= 60:
            break
        ok = True
        for c in chosen:
            if _PAIR_INTER[(idx, c)] > 1:
                ok = False
                break
        if ok:
            chosen.append(idx)
    return [ALL_TRIPLES[i] for i in chosen]


def _local_search(triples, rng):
    """Try to reduce alpha by removing a triple and re-adding a better one."""
    best = triples
    best_alpha = _alpha(triples)
    for _ in range(400):
        if time.time() > _DEADLINE:
            break
        # Try removing each triple and re-greedy filling.
        for ri in range(len(best)):
            cand = [t for k, t in enumerate(best) if k != ri]
            # Greedily add random triples to fill.
            order = list(range(len(ALL_TRIPLES)))
            rng.shuffle(order)
            added = 0
            for idx in order:
                if len(cand) >= len(best):
                    break
                ok = True
                for c in cand:
                    if _PAIR_INTER[(idx, c)] > 1:
                        ok = False
                        break
                if ok:
                    cand.append(idx)
                    added += 1
            a = _alpha(cand)
            if a < best_alpha:
                best = cand
                best_alpha = a
    return best, best_alpha


def construct():
    global _DEADLINE
    rng = random.Random(12345)
    best = None
    best_alpha = N + 1

    # Phase 1: many randomized greedy restarts.
    while time.time() < _DEADLINE - 10.0:
        t = _greedy(rng)
        if not t:
            continue
        a = _alpha(t)
        if a < best_alpha:
            best_alpha = a
            best = t
            if best_alpha <= 5:
                break
        # Occasional local search from a good seed.
        if a <= 6 and rng.random() < 0.3:
            t2, a2 = _local_search(t, rng)
            if a2 < best_alpha:
                best_alpha = a2
                best = t2
                if best_alpha <= 5:
                    break

    if best is None:
        # Fallback: seed-like construction (always valid).
        best = [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
    return best


# EVOLVE-BLOCK-END
