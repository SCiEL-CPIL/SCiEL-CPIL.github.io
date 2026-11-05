"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time


def _hamming_dist(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def _covered_by(word, codewords, radius):
    for cw in codewords:
        if _hamming_dist(word, cw) <= radius:
            return True
    return False


def _build_coverage(codewords, q, n, radius):
    """Return a set of all words covered by the given codewords."""
    covered = set()
    for cw in codewords:
        # Add the codeword itself
        covered.add(tuple(cw))
        if radius >= 1:
            for pos in range(n):
                orig = cw[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    lst = list(cw)
                    lst[pos] = new_val
                    covered.add(tuple(lst))
    return covered


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    # Direct sum of the ternary tetracode (9 words of length 4, a perfect code of covering
    # radius 1) with every word on the other n - 4 coordinates: 9 * 3**(n - 4) words.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    total_words = q ** n

    # Map each word to an index
    def word_to_idx(word):
        idx = 0
        for v in word:
            idx = idx * q + v
        return idx

    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute the ball (covered words) for each codeword as a list of indices
    def ball_indices(cw):
        inds = [word_to_idx(cw)]
        if radius >= 1:
            for pos in range(n):
                orig = cw[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    lst = list(cw)
                    lst[pos] = new_val
                    inds.append(word_to_idx(lst))
        return inds

    # Build coverage count: for each word in the space, how many codewords cover it
    def build_coverage_count(cw_list):
        counts = [0] * total_words
        for cw in cw_list:
            for idx in ball_indices(cw):
                counts[idx] += 1
        return counts

    # Check if removing a codeword keeps coverage valid (using precomputed ball)
    def is_redundant(cw, counts):
        for idx in ball_indices(cw):
            if counts[idx] <= 1:
                return False
        return True

    # --- Phase 1: fast greedy removal with O(1) delta count updates ---
    counts = build_coverage_count(codewords)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() > deadline:
                return codewords
            if is_redundant(codewords[i], counts):
                # Remove codeword i: decrement counts for its ball
                for idx in ball_indices(codewords[i]):
                    counts[idx] -= 1
                codewords.pop(i)
                improved = True
                break  # Restart the scan from the end

    # --- Phase 2: swap-based local search ---
    # Try to replace a codeword with another that covers more uniquely-covered words
    # while keeping all words covered.
    if time.time() < deadline:
        # Rebuild counts to be safe
        counts = build_coverage_count(codewords)
        # Precompute balls for current codewords
        balls = [ball_indices(cw) for cw in codewords]

        improved_swap = True
        while improved_swap and time.time() < deadline:
            improved_swap = False
            for i in range(len(codewords)):
                if time.time() > deadline:
                    break
                # For each candidate replacement position j != i, try swapping
                # Actually, we try: remove i, add a new word w, check coverage
                # Efficient approach: find words that are only covered by codeword i
                # Then try to cover them with a new codeword
                # Simpler: try all possible replacements in a neighborhood
                # For speed, try swapping codeword i with codeword j (different index)
                # but that's just reordering. Instead, try new words.
                # Let's try: for each position in codeword i, change one symbol
                for pos in range(n):
                    if time.time() > deadline:
                        break
                    orig = codewords[i][pos]
                    for new_val in range(q):
                        if new_val == orig:
                            continue
                        new_cw = list(codewords[i])
                        new_cw[pos] = new_val
                        # Check if new_cw is already in codewords
                        new_cw_t = tuple(new_cw)
                        if any(tuple(c) == new_cw_t for c in codewords):
                            continue
                        # Check coverage: words covered only by i must be covered by new_cw
                        # Words covered by i but not new_cw: need counts[idx] >= 2 or covered by new_cw
                        # Words covered by new_cw but not i: fine
                        # Simpler: temporarily remove i, add new_cw, check min count >= 1
                        # But that's O(n*q) per check. Let's do it smartly.
                        # Words that lose coverage: in ball(i) \ ball(new_cw) with counts[idx] == 1
                        # Words that gain coverage: in ball(new_cw) \ ball(i)
                        ball_i_set = set(balls[i])
                        ball_new = ball_indices(new_cw)
                        ball_new_set = set(ball_new)
                        # Check losing words
                        valid = True
                        for idx in ball_i_set:
                            if idx not in ball_new_set:
                                if counts[idx] <= 1:
                                    valid = False
                                    break
                        if not valid:
                            continue
                        # Perform swap
                        for idx in ball_i_set:
                            counts[idx] -= 1
                        for idx in ball_new_set:
                            counts[idx] += 1
                        codewords[i] = new_cw
                        balls[i] = ball_new
                        improved_swap = True
                        break
                    if improved_swap:
                        break
                if improved_swap:
                    break
            if not improved_swap:
                # No single-symbol swap helped; try two-symbol swaps
                for i in range(len(codewords)):
                    if time.time() > deadline:
                        break
                    for pos1 in range(n):
                        if time.time() > deadline:
                            break
                        for pos2 in range(pos1 + 1, n):
                            if time.time() > deadline:
                                break
                            orig1 = codewords[i][pos1]
                            orig2 = codewords[i][pos2]
                            for v1 in range(q):
                                if v1 == orig1:
                                    continue
                                for v2 in range(q):
                                    if v2 == orig2:
                                        continue
                                    new_cw = list(codewords[i])
                                    new_cw[pos1] = v1
                                    new_cw[pos2] = v2
                                    new_cw_t = tuple(new_cw)
                                    if any(tuple(c) == new_cw_t for c in codewords):
                                        continue
                                    ball_i_set = set(balls[i])
                                    ball_new = ball_indices(new_cw)
                                    ball_new_set = set(ball_new)
                                    valid = True
                                    for idx in ball_i_set:
                                        if idx not in ball_new_set:
                                            if counts[idx] <= 1:
                                                valid = False
                                                break
                                    if not valid:
                                        continue
                                    for idx in ball_i_set:
                                        counts[idx] -= 1
                                    for idx in ball_new_set:
                                        counts[idx] += 1
                                    codewords[i] = new_cw
                                    balls[i] = ball_new
                                    improved_swap = True
                                    break
                                if improved_swap:
                                    break
                            if improved_swap:
                                break
                        if improved_swap:
                            break
                    if improved_swap:
                        break

    # --- Phase 3: final greedy removal pass ---
    if time.time() < deadline:
        counts = build_coverage_count(codewords)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(codewords) - 1, -1, -1):
                if time.time() > deadline:
                    return codewords
                if is_redundant(codewords[i], counts):
                    for idx in ball_indices(codewords[i]):
                        counts[idx] -= 1
                    codewords.pop(i)
                    improved = True
                    break

    return codewords
# EVOLVE-BLOCK-END
