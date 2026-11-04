"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import math

# Deadline guard: evaluator kills at 150s; keep a safety margin.
_DEADLINE = 150.0
_SAFETY = 5.0  # seconds to leave before the hard kill


def _elapsed_ok(start):
    return (time.time() - start) < (_DEADLINE - _SAFETY)


def construct(q, n, radius):
    start = time.time()
    rng = random.Random(12345)

    # For the canonical task: q=3, n=6, radius=1 -> 729 words, best known 73.
    # We use a greedy set-cover / local-search approach.

    # Represent words as integers for speed.
    def word_to_int(w):
        v = 0
        for i in range(n):
            v = v * q + w[i]
        return v

    def int_to_word(v):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = v % q
            v //= q
        return w

    total = q ** n
    if total <= 0:
        return []

    # Precompute the "ball" of radius r around a word: all words within Hamming distance <= radius.
    # For q=3, n=6, radius=1: ball size = 1 + n*(q-1) = 1 + 6*2 = 13.
    # Generate all neighbors including self.
    def ball(w):
        """Return list of ints for all words within Hamming distance <= radius from word w (as int)."""
        # Convert to list
        wl = int_to_word(w)
        result = [w]
        if radius >= 1:
            for pos in range(n):
                orig = wl[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    wl[pos] = val
                    result.append(word_to_int(wl))
                wl[pos] = orig
        # For radius > 1, we'd need more, but for this task radius=1.
        return result

    # Greedy set cover: repeatedly pick the word that covers the most uncovered points.
    # Represent uncovered as a bitset or boolean array.
    # total = 729, so a boolean list of 729 is fine.

    uncovered = [True] * total
    covered_count = 0
    codewords = []

    # Precompute balls for all words? 729 * 13 = 9477 ints. Fine.
    all_balls = [ball(i) for i in range(total)]

    # Greedy: pick word with max coverage of uncovered.
    while covered_count < total and _elapsed_ok(start):
        best_word = -1
        best_gain = 0
        for w in range(total):
            # Count how many uncovered points are in ball(w)
            gain = 0
            for u in all_balls[w]:
                if uncovered[u]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_word = w
            if gain == total - covered_count:  # can't do better
                break
        if best_word < 0 or best_gain == 0:
            break
        # Add best_word
        codewords.append(best_word)
        for u in all_balls[best_word]:
            if uncovered[u]:
                uncovered[u] = False
                covered_count += 1

    # Now do local search: try to remove or swap codewords to reduce count while maintaining coverage.
    # We'll work with the list of codewords (ints).

    def is_covered_by(codeword_list, point):
        """Check if point is within radius of any codeword in codeword_list."""
        for cw in codeword_list:
            # Compute Hamming distance
            # Quick check: is point in ball(cw)?
            if point in all_balls[cw]:
                return True
        return False

    # For efficiency, maintain a coverage count: for each point, how many codewords cover it.
    # coverage[p] = number of codewords whose ball contains p.
    def build_coverage(cw_list):
        cov = [0] * total
        for cw in cw_list:
            for u in all_balls[cw]:
                cov[u] += 1
        return cov

    def try_remove(cw_list, coverage, idx):
        """Try removing codeword at idx. Return True if coverage remains valid (all >= 1)."""
        cw = cw_list[idx]
        for u in all_balls[cw]:
            coverage[u] -= 1
            if coverage[u] == 0:
                # Not valid; revert
                coverage[u] += 1
                return False
        # Valid removal
        cw_list.pop(idx)
        return True

    def try_swap(cw_list, coverage, idx, new_cw):
        """Try replacing codeword at idx with new_cw. Return True if valid."""
        old_cw = cw_list[idx]
        if old_cw == new_cw:
            return False
        # Temporarily remove old
        for u in all_balls[old_cw]:
            coverage[u] -= 1
        # Add new
        for u in all_balls[new_cw]:
            coverage[u] += 1
        # Check all covered
        for u in range(total):
            if coverage[u] == 0:
                # Revert
                for u2 in all_balls[new_cw]:
                    coverage[u2] -= 1
                for u2 in all_balls[old_cw]:
                    coverage[u2] += 1
                return False
        cw_list[idx] = new_cw
        return True

    # Local search: iterate, try removals first, then swaps.
    # Use simulated annealing for swaps to escape local minima.
    max_iters = 50000
    temp = 1.0
    cooling = 0.9995

    # Initial coverage
    coverage = build_coverage(codewords)

    for it in range(max_iters):
        if not _elapsed_ok(start):
            break
        improved = False

        # Try removals
        for idx in range(len(codewords) - 1, -1, -1):
            if try_remove(codewords, coverage, idx):
                improved = True
                # After removal, coverage is updated; continue trying other removals
                # But we need to be careful: after popping, indices shift.
                # Let's restart the removal scan for simplicity, but that's expensive.
                # Instead, break and restart the outer loop.
                break

        if improved:
            continue

        # Try swaps with simulated annealing
        if len(codewords) > 0:
            idx = rng.randrange(len(codewords))
            new_cw = rng.randrange(total)
            if new_cw != codewords[idx]:
                # Check if swap is valid
                if try_swap(codewords, coverage, idx, new_cw):
                    # Accept if it doesn't increase count (it doesn't, it's a swap)
                    # For SA, we might accept worse solutions, but here swaps keep size same.
                    # The goal is to enable future removals. So accept all valid swaps?
                    # That could loop. Let's use SA: accept with probability based on "energy".
                    # Define energy as number of points with coverage == 1 (fragile points).
                    # Lower energy is better (more robust, easier to remove).
                    # But computing energy each time is expensive.
                    # Alternative: just accept all valid swaps for a while, then try removals again.
                    pass
                else:
                    # Rejected
                    pass

        # Occasionally restart with a different random seed for diversity?
        # Keep it simple.

    # Convert back to lists of ints
    return [int_to_word(cw) for cw in codewords]
# EVOLVE-BLOCK-END
