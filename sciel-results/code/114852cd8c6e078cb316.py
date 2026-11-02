"""Greedy set-cover construction for q-ary covering codes."""
import time
import itertools
import numpy as np

_DEADLINE = 145.0  # safety margin before 150s kill


def construct(q, n, radius):
    t0 = time.time()
    total = q ** n

    # Represent each word as a base-q integer in [0, total)
    # Precompute the Hamming ball around each word
    # For efficiency, generate neighbors by single-position changes (radius 1)
    # or use general radius support

    # Build the cover: for each candidate c, the set of words within distance <= radius
    # We'll use a numpy array of uncovered flags, and for greedy we need
    # count of uncovered words in each candidate's ball.

    # Strategy: represent words as integers. Precompute for each word its ball as a list of ints.
    # For q=3, n=7: 2187 words, each ball size = 1 + 7*2 = 15 (radius 1). Total pairs ~32805.

    # Precompute digit decomposition for fast neighbor generation
    # word_to_digits and digits_to_word

    # For general radius, we need all words within Hamming distance <= radius.
    # For small n and q, we can enumerate all words and compute distances, but that's O(total^2).
    # Better: for each word, generate its ball by changing up to 'radius' positions.

    # Let's precompute the ball for each word as a frozenset of ints.
    # But 2187 balls of ~15-30 elements each is fine.

    # First, map int <-> tuple of digits
    def int_to_digits(x):
        d = []
        for _ in range(n):
            d.append(x % q)
            x //= q
        return d  # least significant first

    def digits_to_int(d):
        x = 0
        for i in range(n):
            x = x * q + d[i]
        return x

    # Precompute all words as digit tuples
    all_digits = [int_to_digits(w) for w in range(total)]

    # Precompute balls: ball[w] = list of word-indices within Hamming distance <= radius
    # For radius 1: self + single-position changes
    # For general radius: use itertools.combinations for positions, product for values

    balls = []
    for w in range(total):
        d = all_digits[w]
        covered = {w}
        if radius >= 1:
            # All words at Hamming distance exactly 1..radius
            # For radius 1:
            for pos in range(n):
                for val in range(q):
                    if val == d[pos]:
                        continue
                    nd = list(d)
                    nd[pos] = val
                    nw = digits_to_int(nd)
                    covered.add(nw)
            if radius >= 2:
                # Add distance-2 words
                for p1 in range(n):
                    for p2 in range(p1 + 1, n):
                        for v1 in range(q):
                            if v1 == d[p1]:
                                continue
                            for v2 in range(q):
                                if v2 == d[p2]:
                                    continue
                                nd = list(d)
                                nd[p1] = v1
                                nd[p2] = v2
                                nw = digits_to_int(nd)
                                covered.add(nw)
        balls.append(covered)

    # Greedy set cover
    uncovered = np.ones(total, dtype=bool)
    # uncovered_count for each candidate
    # We'll maintain an array of how many uncovered words each candidate covers
    cover_count = np.zeros(total, dtype=np.int32)
    for w in range(total):
        for u in balls[w]:
            cover_count[w] += 1

    chosen = []
    best_count = cover_count.copy()  # current uncovered coverage for each candidate

    while uncovered.any():
        if time.time() > _DEADLINE:
            break
        # Pick candidate with max uncovered coverage
        idx = int(np.argmax(best_count))
        if best_count[idx] == 0:
            break
        chosen.append(idx)
        # Mark all words in ball[idx] as covered
        for u in balls[idx]:
            if uncovered[u]:
                uncovered[u] = False
                # Decrease best_count for all candidates that cover u
                # This is expensive; instead, recompute lazily or use a different approach
        # For efficiency with small total, recompute best_count from scratch each iteration
        # is O(total * ball_size) per iteration. With ~100 iterations and total=2187, ball_size~15:
        # 100 * 2187 * 15 = 3.3M ops, fine.
        # But let's do it more carefully: only update candidates that cover the newly-covered words.
        # Actually, let's just recompute since it's simple and fast enough.
        # Recompute best_count
        best_count[:] = 0
        for w in range(total):
            cnt = 0
            for u in balls[w]:
                if uncovered[u]:
                    cnt += 1
            best_count[w] = cnt

    # If we ran out of time or got stuck, add random candidates to ensure coverage
    if uncovered.any() and time.time() < _DEADLINE - 5:
        # Add candidates greedily with a simpler method
        remaining = np.where(uncovered)[0]
        while remaining.size > 0 and time.time() < _DEADLINE - 2:
            # Pick a random uncovered word, add its best candidate
            u = int(remaining[np.random.randint(remaining.size)])
            # Find candidate with max uncovered coverage among those covering u
            # Simplify: just pick u itself as a candidate (it covers at least u)
            if u not in chosen:
                chosen.append(u)
                for v in balls[u]:
                    uncovered[v] = False
                remaining = np.where(uncovered)[0]

    # Convert chosen ints back to digit lists
    result = [all_digits[c] for c in chosen]

    # Ensure distinctness (should be by construction)
    seen = set()
    final = []
    for c in result:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            final.append(c)

    return final
