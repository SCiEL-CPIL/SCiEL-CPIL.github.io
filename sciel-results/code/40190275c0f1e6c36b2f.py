"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys


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


def _build_ball(cw, q, n, radius):
    """Return a set of indices for all words within Hamming distance radius of cw."""
    covered = set()
    covered.add(_word_to_idx(cw, q))
    if radius >= 1:
        for pos in range(n):
            orig = cw[pos]
            for new_val in range(q):
                if new_val == orig:
                    continue
                lst = list(cw)
                lst[pos] = new_val
                covered.add(_word_to_idx(lst, q))
    return covered


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    total_words = q ** n

    # Build initial covering code using direct sum of ternary tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]

    # Precompute balls for all codewords
    balls = [_build_ball(cw, q, n, radius) for cw in codewords]

    # Build coverage counts
    counts = [0] * total_words
    for ball in balls:
        for w in ball:
            counts[w] += 1

    def is_covered():
        return all(c > 0 for c in counts)

    def is_redundant(idx):
        for w in balls[idx]:
            if counts[w] <= 1:
                return False
        return True

    # Phase 1: Greedy removal - repeatedly remove redundant codewords
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() > deadline:
                break
            if is_redundant(i):
                # Remove codeword i
                for w in balls[i]:
                    counts[w] -= 1
                codewords.pop(i)
                balls.pop(i)
                improved = True
                break

    best_code = codewords[:]
    best_len = len(codewords)

    # Phase 2: Local search with add-remove and swap operations
    # Use randomization to escape local minima

    def try_remove(idx):
        """Try to remove codeword idx. Returns True if successful."""
        if not is_redundant(idx):
            return False
        for w in balls[idx]:
            counts[w] -= 1
        codewords.pop(idx)
        balls.pop(idx)
        return True

    def try_add(word):
        """Add a word to the code. Returns the index of the new codeword."""
        idx = len(codewords)
        codewords.append(word)
        ball = _build_ball(word, q, n, radius)
        balls.append(ball)
        for w in ball:
            counts[w] += 1
        return idx

    def try_remove_then_add(idx, new_word):
        """Remove codeword idx and add new_word. Returns True if coverage maintained."""
        # Check if removal would break coverage
        for w in balls[idx]:
            if counts[w] <= 1:
                return False
        # Remove
        for w in balls[idx]:
            counts[w] -= 1
        old_ball = balls[idx]
        codewords[idx] = new_word
        new_ball = _build_ball(new_word, q, n, radius)
        balls[idx] = new_ball
        for w in new_ball:
            counts[w] += 1
        return True

    # Local search loop
    rng = random.Random(42)
    search_count = 0
    max_search_steps = 50000

    while time.time() < deadline and search_count < max_search_steps:
        search_count += 1
        m = len(codewords)
        if m <= 1:
            break

        # Strategy 1: Try to remove a random codeword
        if m > 1:
            idx = rng.randrange(m)
            if is_redundant(idx):
                for w in balls[idx]:
                    counts[w] -= 1
                codewords.pop(idx)
                balls.pop(idx)
                if len(codewords) < best_len:
                    best_code = codewords[:]
                    best_len = len(codewords)
                continue

        # Strategy 2: Try to replace a codeword with a nearby one
        idx = rng.randrange(m)
        old_cw = codewords[idx]

        # Generate a random word at distance 1 or 2 from old_cw
        dist = rng.choice([1, 2])
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

        # Try the swap
        if try_remove_then_add(idx, new_cw):
            if is_covered():
                if len(codewords) < best_len:
                    best_code = codewords[:]
                    best_len = len(codewords)
            else:
                # Undo - restore old state
                for w in balls[idx]:
                    counts[w] -= 1
                codewords[idx] = old_cw
                old_ball = _build_ball(old_cw, q, n, radius)
                balls[idx] = old_ball
                for w in old_ball:
                    counts[w] += 1

        # Strategy 3: Add a random word, then try to remove two
        if m > 2 and time.time() < deadline:
            random_word = [rng.randrange(q) for _ in range(n)]
            # Check if this word is not already in the code
            if random_word not in codewords:
                new_idx = try_add(random_word)
                # Try to remove two codewords
                removed_count = 0
                for _ in range(2):
                    if len(codewords) <= 2:
                        break
                    ridx = rng.randrange(len(codewords))
                    if is_redundant(ridx):
                        for w in balls[ridx]:
                            counts[w] -= 1
                        codewords.pop(ridx)
                        balls.pop(ridx)
                        removed_count += 1
                        if len(codewords) < best_len:
                            best_code = codewords[:]
                            best_len = len(codewords)
                # Remove the random word we added if it's still there
                # Find it - it should be at some index
                if random_word in codewords:
                    ridx2 = codewords.index(random_word)
                    for w in balls[ridx2]:
                        counts[w] -= 1
                    codewords.pop(ridx2)
                    balls.pop(ridx2)

    return best_code
# EVOLVE-BLOCK-END
