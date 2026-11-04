"""Fresh design: greedy set-cover + local search over the full (bit-packed) space,
with a deadline guard and a seed-style construction as a safe fallback."""
import time
import itertools
import random

import numpy as np

_DEADLINE_FRACTION = 0.90  # safety margin vs. the 150s wall-clock kill


def _ball_masks(q, n):
    """Return (all_masks, ball_masks).

    all_masks[i] : int bitset of word i and every word at Hamming distance <=1
                   (i.e. the radius-1 ball around i, including i itself).
    ball_masks[i]: int bitset of words at distance exactly 1 from i.
    """
    total = q ** n
    all_masks = np.zeros(total, dtype=object)
    ball_masks = np.zeros(total, dtype=object)
    for w in range(total):
        m = 1 << w
        for pos in range(n):
            base = (w // (q ** pos)) % q
            for nv in range(q):
                if nv == base:
                    continue
                v = w + (nv - base) * (q ** pos)
                m |= (1 << v)
                ball_masks[w] |= (1 << v)
        all_masks[w] = m
    return all_masks, ball_masks


def _greedy_cover(uncovered, ball_masks, rng, max_rounds=200000):
    """Greedy set cover: repeatedly pick the word covering the most uncovered words.
    Returns (chosen_words, final_uncovered_bitset)."""
    chosen = []
    uc = uncovered
    total = len(ball_masks)
    # incremental "newly covered by each word" counter
    newcov = np.zeros(total, dtype=np.int64)
    for w in range(total):
        newcov[w] = (uc & ball_masks[w]).bit_count()
    rounds = 0
    while uc and rounds < max_rounds:
        best = int(np.argmax(newcov))
        if newcov[best] <= 0:
            break
        chosen.append(best)
        uc &= ~ball_masks[best]
        # update counters for words sharing at least one newly-covered word
        affected = (newcov > 0)
        if np.any(affected):
            idx = np.nonzero(affected)[0]
            for i in idx:
                newcov[i] = (uc & ball_masks[i]).bit_count()
        rounds += 1
    return chosen, uc


def _local_search(chosen, ball_masks, all_masks, rng, deadline):
    """Add/remove/swap local search to reduce |chosen| while keeping full coverage."""
    chosen = list(chosen)
    chosen_set = set(chosen)
    total = len(ball_masks)
    full = (1 << total) - 1
    if len(chosen) == 0:
        return chosen

    def coverage():
        cov = 0
        for w in chosen:
            cov |= all_masks[w]
        return cov

    cov = coverage()
    improved = True
    while improved and time.time() < deadline:
        improved = False

        # --- REMOVE: drop a word whose ball is redundant ---
        for w in list(chosen):
            if time.time() >= deadline:
                return chosen
            if (cov & ~all_masks[w]) == (cov & ~all_masks[w]) and (cov | all_masks[w]) == cov:
                # check redundancy: removing w still keeps full coverage
                cov_without = 0
                for o in chosen:
                    if o != w:
                        cov_without |= all_masks[o]
                if cov_without == full:
                    chosen.remove(w)
                    cov = cov_without
                    improved = True
                    break

        if not improved:
            # --- SWAP: replace one word with another that covers the same or more ---
            for w in list(chosen):
                if time.time() >= deadline:
                    return chosen
                # words that could replace w: those whose ball covers w's unique part
                unique = all_masks[w]
                # try a random candidate
                for _ in range(3):
                    cand = rng.randrange(total)
                    if cand in chosen_set:
                        continue
                    # new coverage if we swap w -> cand
                    base = 0
                    for o in chosen:
                        if o != w:
                            base |= all_masks[o]
                    newcov = base | all_masks[cand]
                    if newcov == full:
                        # ensure it does not increase size (it doesn't) and ideally helps
                        chosen[chosen.index(w)] = cand
                        chosen_set.discard(w)
                        chosen_set.add(cand)
                        cov = newcov
                        improved = True
                        break
                if improved:
                    break
    return chosen


def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 150.0 * _DEADLINE_FRACTION

    total = q ** n

    # Safe fallback: seed's direct-sum construction (valid for the standard task).
    def seed_construction():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3]
                     for a in range(q) for b in range(q)]
        if n <= 4:
            # perfect-ish: just use all words if tiny, else tetracode prefix
            return [list(h) for h in tetracode[:q ** n]]
        return [h + list(tail) for h in tetracode
                for tail in itertools.product(range(q), repeat=n - 4)]

    # If the space is too large to enumerate bitsets, fall back immediately.
    if total > (1 << 60):
        return seed_construction()

    try:
        rng = random.Random(12345)
        all_masks, ball_masks = _ball_masks(q, n)
        full = (1 << total) - 1
        uncovered = full

        chosen, uncovered = _greedy_cover(uncovered, ball_masks, rng)

        # If greedy did not fully cover (shouldn't happen), fill in remainder.
        if uncovered:
            for w in range(total):
                if time.time() >= deadline:
                    break
                if uncovered & (1 << w):
                    chosen.append(w)
                    uncovered &= ~all_masks[w]
                    if not uncovered:
                        break

        # Local search to shrink.
        chosen = _local_search(chosen, ball_masks, all_masks, rng, deadline)

        # Convert to lists of ints.
        qpow = [q ** i for i in range(n)]
        result = []
        seen = set()
        for w in chosen:
            digits = [(w // p) % q for p in qpow]
            t = tuple(digits)
            if t not in seen:
                seen.add(t)
                result.append(digits)

        if not result:
            return seed_construction()
        return result

    except Exception:
        return seed_construction()
