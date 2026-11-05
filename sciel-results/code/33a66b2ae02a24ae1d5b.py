"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys
from collections import defaultdict


def _word_to_idx(word, q):
    idx = 0
    for v in word:
        idx = idx * q + v
    return idx


def _idx_to_word(idx, q, n):
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word


def _build_ball_indices(cw, q, n, radius):
    """Return a frozenset of indices for all words within Hamming distance radius of cw."""
    base = _word_to_idx(cw, q)
    covered = {base}
    if radius >= 1:
        for pos in range(n):
            orig = cw[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                lst = list(cw)
                lst[pos] = new_val
                covered.add(_word_to_idx(lst, q))
    return frozenset(covered)


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    total_words = q ** n

    # Build initial covering code using direct sum of ternary tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]

    # Precompute balls for all codewords
    balls = [_build_ball_indices(cw, q, n, radius) for cw in codewords]

    # Build coverage counts
    counts = [0] * total_words
    for ball in balls:
        for w in ball:
            counts[w] += 1

    # Track number of uncovered words (should be 0 for a valid covering)
    uncovered_count = sum(1 for c in counts if c == 0)

    def is_redundant(idx):
        for w in balls[idx]:
            if counts[w] <= 1:
                return False
        return True

    def remove_codeword(idx):
        """Remove codeword at index idx, updating counts and uncovered_count."""
        nonlocal uncovered_count
        for w in balls[idx]:
            if counts[w] == 1:
                uncovered_count += 1
            counts[w] -= 1
        codewords.pop(idx)
        balls.pop(idx)

    def add_codeword(word):
        """Add a new codeword, updating counts and uncovered_count. Returns new index."""
        nonlocal uncovered_count
        idx = len(codewords)
        codewords.append(word)
        ball = _build_ball_indices(word, q, n, radius)
        balls.append(ball)
        for w in ball:
            if counts[w] == 0:
                uncovered_count -= 1
            counts[w] += 1
        return idx

    # Phase 1: Greedy removal - repeatedly remove redundant codewords
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() > deadline:
                break
            if is_redundant(i):
                remove_codeword(i)
                improved = True
                break

    best_code = codewords[:]
    best_len = len(codewords)

    # Phase 2: Local search
    rng = random.Random(42)
    search_count = 0
    max_search_steps = 200000

    while time.time() < deadline and search_count < max_search_steps:
        search_count += 1
        m = len(codewords)
        if m <= 1:
            break

        # Strategy 1: Try to remove a random codeword
        idx = rng.randrange(m)
        if is_redundant(idx):
            remove_codeword(idx)
            if len(codewords) < best_len:
                best_code = codewords[:]
                best_len = len(codewords)
            continue

        # Strategy 2: Try to replace a codeword with a nearby one
        idx = rng.randrange(m)
        old_cw = codewords[idx]

        # Generate a random word at distance 1 or 2 from old_cw
        dist = rng.choice([1, 1, 2])
        new_cw = old_cw[:]
        positions = list(range(n))
        rng.shuffle(positions)
        for pos in positions[:dist]:
            old_val = new_cw[pos]
            new_val = rng.randrange(q)
            while new_val == old_val:
                new_val = rng.randrange(q)
            new_cw[pos] = new_val

        if new_cw == old_cw:
            continue

        # Check if swap maintains coverage
        old_ball = balls[idx]
        # Words only covered by old_cw would become uncovered
        would_break = False
        for w in old_ball:
            if counts[w] == 1:
                would_break = True
                break

        if not would_break:
            # Perform the swap
            for w in old_ball:
                counts[w] -= 1
            new_ball = _build_ball_indices(new_cw, q, n, radius)
            codewords[idx] = new_cw
            balls[idx] = new_ball
            for w in new_ball:
                counts[w] += 1

            # Try to remove a redundant codeword after the swap
            m2 = len(codewords)
            for _ in range(3):
                if len(codewords) <= 1:
                    break
                ridx = rng.randrange(len(codewords))
                if is_redundant(ridx):
                    remove_codeword(ridx)
                    if len(codewords) < best_len:
                        best_code = codewords[:]
                        best_len = len(codewords)

        # Strategy 3: Try to find two codewords whose union covers one's ball
        # Pick two random codewords and check if one can be removed
        if m >= 2:
            i1 = rng.randrange(m)
            i2 = rng.randrange(m)
            if i1 != i2 and is_redundant(i1):
                remove_codeword(i1)
                if len(codewords) < best_len:
                    best_code = codewords[:]
                    best_len = len(codewords)
            elif i1 != i2 and is_redundant(i2):
                remove_codeword(i2)
                if len(codewords) < best_len:
                    best_code = codewords[:]
                    best_len = len(codewords)

    # Final cleanup: remove any remaining redundant codewords
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() > deadline:
                break
            if is_redundant(i):
                remove_codeword(i)
                improved = True
                break

    if len(codewords) < best_len:
        best_code = codewords[:]
        best_len = len(codewords)

    return best_code
# EVOLVE-BLOCK-END
