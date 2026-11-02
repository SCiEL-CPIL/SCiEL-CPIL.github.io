"""Covering code K_3(7,1) via bitmask balls + greedy set cover + local search."""
import time
import random
from collections import deque


def _build_balls(q, n, radius):
    """Return list of bitsets (Python ints), one per word in {0..q-1}^n,
    where bitset[i] = set of words within Hamming distance radius of word i."""
    total = q ** n
    # Precompute one-position variants for each coordinate
    # For each word index, generate all neighbors at dist 1
    # Represent words in base-q via digits
    # We'll build the distance-1 ball for each word then include itself
    # Efficient: for each coordinate and each value, the words that differ only there

    # Precompute for each position p and value v: list of word indices that have value v at p
    pos_val = [[[] for _ in range(q)] for _ in range(n)]
    digits = [[0] * n for _ in range(total)]
    for i in range(total):
        x = i
        for p in range(n):
            d = x % q
            digits[i][p] = d
            x //= q
            pos_val[p][d].append(i)

    balls = [0] * total
    mask_all = (1 << total) - 1
    for i in range(total):
        b = 1 << i
        for p in range(n):
            old = digits[i][p]
            for v in range(q):
                if v == old:
                    continue
                # all words in pos_val[p][v] are at dist 1 from i
                for j in pos_val[p][v]:
                    b |= (1 << j)
        balls[i] = b
    return balls


def _covered_by(balls, chosen):
    cov = 0
    for i in chosen:
        cov |= balls[i]
    return cov


def _is_cover(balls, chosen, total):
    cov = _covered_by(balls, chosen)
    return cov == (1 << total) - 1


def construct(q, n, radius):
    t0 = time.time()
    DEADLINE = t0 + 110.0  # safety margin under 120s

    total = q ** n
    # Build balls (distance <= radius). For radius 1 this is O(total*n*q).
    balls = _build_balls(q, n, radius)
    full = (1 << total) - 1

    # Greedy set cover: repeatedly pick ball with max new coverage.
    # Maintain uncovered mask; candidate score = popcount(ball & uncovered).
    uncovered = full
    chosen = []
    # Precompute popcount via int.bit_count
    # To speed up, we maintain for each candidate its coverage of uncovered.
    # We'll do a simpler loop: each iteration scan all not-chosen for max gain.
    chosen_set = set()
    while uncovered:
        best_i = -1
        best_gain = 0
        for i in range(total):
            if i in chosen_set:
                continue
            gain = (balls[i] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_i = i
                if gain == uncovered.bit_count():
                    break
        if best_i == -1 or best_gain == 0:
            break
        chosen.append(best_i)
        chosen_set.add(best_i)
        uncovered &= ~balls[best_i]
        if time.time() > DEADLINE:
            break

    if uncovered:
        # Not fully covered yet; add more greedily but we may be over deadline.
        # Continue quickly without strict per-iteration timing until deadline.
        while uncovered and time.time() < DEADLINE:
            best_i = -1
            best_gain = 0
            for i in range(total):
                if i in chosen_set:
                    continue
                gain = (balls[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
            if best_i == -1 or best_gain == 0:
                break
            chosen.append(best_i)
            chosen_set.add(best_i)
            uncovered &= ~balls[best_i]

    # Local search phase
    # 1) Remove redundant words
    def try_remove_redundant(chosen, chosen_set, balls, total, full, deadline):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            cov = 0
            for i in chosen:
                cov |= balls[i]
            for i in list(chosen):
                if time.time() > deadline:
                    break
                rest = (cov ^ balls[i])  # cov minus i's contribution (i in cov)
                if rest == full:
                    chosen.remove(i)
                    chosen_set.discard(i)
                    cov = rest
                    changed = True
        return chosen, chosen_set

    chosen, chosen_set = try_remove_redundant(chosen, chosen_set, balls, total, full, DEADLINE)

    # 2) Swap: for each chosen i, try replacing with j (not chosen) that keeps cover
    #    and reduces... swap doesn't reduce count. Skip unless we want diversity.
    # Instead do: add-then-remove loop to try to find smaller cover.
    # 3) Add a random word, then remove redundants (may enable better removals).
    rng = random.Random(12345)
    for _ in range(40):
        if time.time() > DEADLINE - 5:
            break
        # add a random word
        j = rng.randrange(total)
        if j not in chosen_set:
            chosen.append(j)
            chosen_set.add(j)
            chosen, chosen_set = try_remove_redundant(chosen, chosen_set, balls, total, full, DEADLINE)

    # Final redundancy removal
    chosen, chosen_set = try_remove_redundant(chosen, chosen_set, balls, total, full, DEADLINE)

    # Convert indices back to words (base-q digits)
    words = []
    for i in chosen:
        x = i
        w = [0] * n
        for p in range(n):
            w[p] = x % q
            x //= q
        words.append(w)

    # Ensure distinct
    seen = set()
    out = []
    for w in words:
        t = tuple(w)
        if t not in seen:
            seen.add(t)
            out.append(w)
    return out
