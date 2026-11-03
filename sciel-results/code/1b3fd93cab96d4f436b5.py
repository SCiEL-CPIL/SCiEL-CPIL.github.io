# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct(q, n, radius):
    """
    Greedy set-cover with randomized restarts, bitmask balls, and a two-pass reduction.
    """
    if n <= 0:
        return [[]]

    total_words = q ** n

    # For very large spaces, fall back to a simpler greedy (not expected for this task)
    if total_words > 20000:
        return _construct_fallback(q, n, radius)

    return _construct_greedy(q, n, radius)

def _construct_greedy(q, n, radius):
    total_words = q ** n

    # Precompute q-powers for fast digit extraction
    q_powers = [1] * n
    for i in range(1, n):
        q_powers[i] = q_powers[i - 1] * q

    # Precompute ball masks (bitmask of all words within Hamming distance <= radius)
    ball_masks = _precompute_balls(q, n, radius, total_words, q_powers)

    all_mask = (1 << total_words) - 1

    # Helper: index -> word
    def idx_to_word(idx):
        word = [0] * n
        temp = idx
        for k in range(n - 1, -1, -1):
            word[k] = temp % q
            temp //= q
        return word

    best_codewords = None
    best_len = float('inf')

    start_time = time.time()
    max_time = 115.0  # leave ~5s buffer

    rng = np.random.RandomState(12345)
    random.seed(42)

    restart = 0
    while True:
        # Randomized greedy: use a random priority order to break ties and explore different solutions
        # We'll use a random permutation for tie-breaking, but also add a small random jitter to scores
        perm = rng.permutation(total_words)
        # Create a list of (random_priority, index) to use for tie-breaking
        # Actually, simpler: iterate over perm, and when scores tie, the one appearing earlier in perm wins.
        # To add more randomness, we can also randomly perturb the score by a tiny amount.

        uncovered = all_mask
        selected = []

        while uncovered:
            best_idx = -1
            best_score = -1

            # Iterate in random order to break ties randomly
            for idx in perm:
                score = (ball_masks[idx] & uncovered).bit_count()
                # Add a tiny random jitter to encourage diversity
                # But keep it small so it doesn't override the main score
                if score > best_score:
                    best_score = score
                    best_idx = idx
                elif score == best_score and score > 0:
                    # Tie: use random choice
                    if random.random() < 0.5:
                        best_idx = idx

            if best_idx == -1 or best_score == 0:
                break

            selected.append(best_idx)
            uncovered &= ~ball_masks[best_idx]

        # Reduction phase: remove redundant codewords
        if selected:
            reduced = _reduce_codewords(selected, ball_masks, total_words)
            codewords = [idx_to_word(idx) for idx in reduced]

            if len(codewords) < best_len:
                best_len = len(codewords)
                best_codewords = codewords

                # Early stop if we reach the known best
                if best_len == 73:
                    break

        restart += 1

        if time.time() - start_time > max_time:
            break

        # Safety cap
        if restart > 50000:
            break

    return best_codewords

def _precompute_balls(q, n, radius, total_words, q_powers):
    """Precompute bitmask balls for each center."""
    ball_masks = [0] * total_words

    if radius == 1:
        for i in range(total_words):
            mask = 1 << i
            for p in range(n):
                step = q_powers[p]
                digit = (i // step) % q
                for v in range(q):
                    if v != digit:
                        j = i + (v - digit) * step
                        mask |= 1 << j
            ball_masks[i] = mask
    elif radius == 2:
        for i in range(total_words):
            mask = 1 << i
            # Distance 1
            for p in range(n):
                step = q_powers[p]
                digit = (i // step) % q
                for v in range(q):
                    if v != digit:
                        j = i + (v - digit) * step
                        mask |= 1 << j
            # Distance 2
            for p1 in range(n):
                for p2 in range(p1 + 1, n):
                    step1 = q_powers[p1]
                    step2 = q_powers[p2]
                    d1 = (i // step1) % q
                    d2 = (i // step2) % q
                    for v1 in range(q):
                        if v1 == d1:
                            continue
                        for v2 in range(q):
                            if v2 == d2:
                                continue
                            j = i + (v1 - d1) * step1 + (v2 - d2) * step2
                            mask |= 1 << j
            ball_masks[i] = mask
    else:
        # Generic fallback for radius > 2 (not expected, but ensure correctness)
        def idx_to_word(idx):
            word = [0] * n
            temp = idx
            for k in range(n - 1, -1, -1):
                word[k] = temp % q
                temp //= q
            return word

        def word_to_idx(word):
            idx = 0
            for k in range(n):
                idx = idx * q + word[k]
            return idx

        for i in range(total_words):
            center = idx_to_word(i)
            mask = 1 << i
            # Brute force: check all words
            for j in range(total_words):
                if i == j:
                    continue
                other = idx_to_word(j)
                dist = sum(1 for a, b in zip(center, other) if a != b)
                if dist <= radius:
                    mask |= 1 << j
            ball_masks[i] = mask

    return ball_masks

def _reduce_codewords(selected_indices, ball_masks, total_words):
    """
    Remove redundant codewords. A codeword is redundant if every word in its ball
    is also covered by at least one other codeword.
    """
    if not selected_indices:
        return []

    # Count coverage
    coverage = [0] * total_words
    for idx in selected_indices:
        mask = ball_masks[idx]
        m = mask
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            coverage[w] += 1
            m ^= lsb

    active = list(selected_indices)
    changed = True
    while changed:
        changed = False
        new_active = []
        for idx in active:
            mask = ball_masks[idx]
            m = mask
            redundant = True
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                if coverage[w] == 1:
                    redundant = False
                    break
                m ^= lsb

            if redundant:
                # Remove this codeword
                m = mask
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    coverage[w] -= 1
                    m ^= lsb
                changed = True
            else:
                new_active.append(idx)
        active = new_active

    return active

def _construct_fallback(q, n, radius):
    """Simple greedy fallback for larger spaces."""
    total_words = q ** n

    def idx_to_word(idx):
        word = [0] * n
        temp = idx
        for k in range(n - 1, -1, -1):
            word[k] = temp % q
            temp //= q
        return word

    ball_masks = _precompute_balls(q, n, radius, total_words, None)

    uncovered = (1 << total_words) - 1
    codewords = []

    while uncovered:
        best_idx = -1
        best_count = -1
        for idx in range(total_words):
            count = (ball_masks[idx] & uncovered).bit_count()
            if count > best_count:
                best_count = count
                best_idx = idx
        if best_idx == -1:
            break
        codewords.append(idx_to_word(best_idx))
        uncovered &= ~ball_masks[best_idx]

    return codewords
# EVOLVE-BLOCK-END
