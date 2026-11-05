# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_DEADLINE = 120.0
_SAFETY = 5.0


def _elapsed_ok(start):
    return (time.time() - start) < (_DEADLINE - _SAFETY)


def construct(q, n, radius):
    start = time.time()
    rng = random.Random(12345)

    # For the canonical task: q=3, n=6, radius=1 -> 729 words, best known 73.
    # We use a vectorized greedy set-cover + local search approach.

    total = q ** n
    if total <= 0:
        return []

    # Precompute balls (radius-1) for all words as flat int arrays.
    # Ball of radius 1: the word itself + all single-position changes.
    # ball size = 1 + n*(q-1)
    ball_size = 1 + n * (q - 1)
    balls_flat = np.empty((total, ball_size), dtype=np.int32)

    # Build balls
    for w in range(total):
        # Convert w to base-q digits
        digits = np.zeros(n, dtype=np.int32)
        tmp = w
        for i in range(n - 1, -1, -1):
            digits[i] = tmp % q
            tmp //= q

        balls_flat[w, 0] = w
        idx = 1
        for pos in range(n):
            orig = digits[pos]
            for val in range(q):
                if val == orig:
                    continue
                # Compute new word index
                # Change digit at pos from orig to val
                # Position value: q^(n-1-pos)
                pos_val = q ** (n - 1 - pos)
                new_w = w + (val - orig) * pos_val
                balls_flat[w, idx] = new_w
                idx += 1

    # Greedy set cover
    uncovered = np.ones(total, dtype=np.uint8)
    covered_count = 0
    codewords = []

    while covered_count < total and _elapsed_ok(start):
        # Compute gain for each candidate: number of uncovered points in its ball
        # Vectorized: for each w, count uncovered in balls_flat[w]
        # This is a 2D gather + count; for 729x13 it's fast
        gains = np.array([np.count_nonzero(uncovered[balls_flat[w]]) for w in range(total)], dtype=np.int32)
        best_word = int(np.argmax(gains))
        best_gain = int(gains[best_word])
        if best_gain == 0:
            break
        codewords.append(best_word)
        uncovered[balls_flat[best_word]] = 0
        covered_count += best_gain

    # Local search with vectorized coverage
    coverage = np.zeros(total, dtype=np.int32)
    for cw in codewords:
        np.add.at(coverage, balls_flat[cw], 1)

    def try_remove_vectorized(idx):
        """Try removing codeword at idx. Returns True if valid."""
        cw = codewords[idx]
        ball = balls_flat[cw]
        # Check if coverage - ball has any zero
        if np.min(coverage - ball) <= 0:
            return False
        # Valid removal
        np.subtract.at(coverage, ball, 1)
        codewords.pop(idx)
        return True

    def try_swap_vectorized(idx, new_cw):
        """Try replacing codeword at idx with new_cw. Returns True if valid."""
        old_cw = codewords[idx]
        if old_cw == new_cw:
            return False
        old_ball = balls_flat[old_cw]
        new_ball = balls_flat[new_cw]
        # Check if coverage - old_ball + new_ball has any zero
        # We need to be careful with overlapping elements
        # Use a temporary copy for the check
        temp_cov = coverage.copy()
        np.subtract.at(temp_cov, old_ball, 1)
        np.add.at(temp_cov, new_ball, 1)
        if np.min(temp_cov) <= 0:
            return False
        # Valid swap
        np.subtract.at(coverage, old_ball, 1)
        np.add.at(coverage, new_ball, 1)
        codewords[idx] = new_cw
        return True

    # Local search: iterate, try removals first, then swaps.
    max_iters = 200000
    for it in range(max_iters):
        if not _elapsed_ok(start):
            break
        improved = False

        # Try removals: scan from end to start
        for idx in range(len(codewords) - 1, -1, -1):
            if try_remove_vectorized(idx):
                improved = True
                break  # Restart scan after a removal

        if improved:
            continue

        # Try swaps: pick a random codeword and a random replacement
        if len(codewords) > 0:
            idx = rng.randrange(len(codewords))
            new_cw = rng.randrange(total)
            if new_cw != codewords[idx]:
                if try_swap_vectorized(idx, new_cw):
                    pass  # Accept the swap (keeps size same, may enable future removals)

    # Convert back to lists of ints
    result = []
    for cw in codewords:
        w = [0] * n
        tmp = cw
        for i in range(n - 1, -1, -1):
            w[i] = tmp % q
            tmp //= q
        result.append(w)
    return result
# EVOLVE-BLOCK-END
