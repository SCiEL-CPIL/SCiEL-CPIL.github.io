"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Direct sum of the ternary tetracode (9 words of length 4, a perfect code of covering
    # radius 1) with every word on the other n - 4 coordinates: 9 * 3**(n - 4) words.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]

    total_words = q ** n

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

    # Precompute covered indices for each codeword
    def get_covered_indices(cw):
        indices = [word_to_idx(cw)]
        if radius >= 1:
            for pos in range(n):
                orig = cw[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    lst = list(cw)
                    lst[pos] = new_val
                    indices.append(word_to_idx(lst))
        return indices

    # Precompute covered indices for all initial codewords
    covered_indices_list = [get_covered_indices(cw) for cw in codewords]

    # Build initial coverage count
    counts = [0] * total_words
    for indices in covered_indices_list:
        for idx in indices:
            counts[idx] += 1

    def is_redundant(cw_idx):
        """Check if codeword at cw_idx is redundant using current counts."""
        for w_idx in covered_indices_list[cw_idx]:
            if counts[w_idx] <= 1:
                return False
        return True

    def remove_codeword(idx):
        """Remove codeword at idx, update counts and covered_indices_list."""
        for w_idx in covered_indices_list[idx]:
            counts[w_idx] -= 1
        codewords.pop(idx)
        covered_indices_list.pop(idx)

    def add_codeword(cw, cw_idx=None):
        """Add codeword, update counts and covered_indices_list. Return new index."""
        indices = get_covered_indices(cw)
        for w_idx in indices:
            counts[w_idx] += 1
        if cw_idx is not None:
            codewords.insert(cw_idx, cw)
            covered_indices_list.insert(cw_idx, indices)
        else:
            codewords.append(cw)
            covered_indices_list.append(indices)
        return len(codewords) - 1

    def is_fully_covered():
        """Check if all words are covered (counts > 0 everywhere)."""
        for c in counts:
            if c == 0:
                return False
        return True

    # Phase 1: Multi-pass greedy removal with different orderings
    # Pass 1: Remove from end (reverse order)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(codewords) - 1, -1, -1):
            if time.time() > deadline:
                break
            if is_redundant(i):
                remove_codeword(i)
                improved = True
                break  # Restart scan after removal

    # Pass 2: Remove from beginning (forward order)
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(codewords)):
                if time.time() > deadline:
                    break
                if is_redundant(i):
                    remove_codeword(i)
                    improved = True
                    break

    # Pass 3: Random order removal
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(codewords)))
            random.shuffle(indices)
            for i in indices:
                if time.time() > deadline:
                    break
                if i < len(codewords) and is_redundant(i):
                    remove_codeword(i)
                    improved = True
                    break

    # Phase 2: Local search with swap moves (add one, remove two)
    best_code = [cw[:] for cw in codewords]
    best_len = len(codewords)

    # Precompute all possible words for random sampling
    all_words = list(itertools.product(range(q), repeat=n))

    while time.time() < deadline - 5.0:  # Leave 5s buffer
        # Try 1: Remove a random codeword
        if len(codewords) > 1:
            idx = random.randrange(len(codewords))
            if is_redundant(idx):
                remove_codeword(idx)
                if len(codewords) < best_len:
                    best_len = len(codewords)
                    best_code = [cw[:] for cw in codewords]
                continue

        # Try 2: Swap - add a random word, then try to remove two
        if len(codewords) > 2 and time.time() < deadline - 5.0:
            # Pick a random word not already in the code
            random_word = list(all_words[random.randrange(len(all_words))])
            if random_word not in codewords:
                add_codeword(random_word)
                # Try to remove two codewords
                removed_count = 0
                indices_to_try = list(range(len(codewords)))
                random.shuffle(indices_to_try)
                for i in indices_to_try:
                    if time.time() > deadline - 5.0:
                        break
                    if removed_count >= 2:
                        break
                    if i < len(codewords) and is_redundant(i):
                        remove_codeword(i)
                        removed_count += 1

                if len(codewords) < best_len:
                    best_len = len(codewords)
                    best_code = [cw[:] for cw in codewords]
                elif len(codewords) <= best_len:
                    # Keep current state if equal or better
                    best_code = [cw[:] for cw in codewords]
                    best_len = len(codewords)

        # Try 3: Add a word that covers the most uncovered words
        if len(codewords) > 1 and time.time() < deadline - 10.0:
            # Find words with lowest coverage
            # Sample some low-coverage words to find a good candidate
            uncovered_candidates = []
            for _ in range(50):
                w_idx = random.randrange(total_words)
                if counts[w_idx] <= 1:
                    uncovered_candidates.append(w_idx)

            if uncovered_candidates:
                # For each uncovered word, find a codeword that would cover it
                best_add = None
                best_gain = 0
                for w_idx in uncovered_candidates[:10]:
                    w = idx_to_word(w_idx)
                    # Find codewords at distance 1 from w
                    for pos in range(n):
                        for new_val in range(q):
                            if new_val == w[pos]:
                                continue
                            candidate = w[:]
                            candidate[pos] = new_val
                            if candidate not in codewords:
                                # Count how many words this would help cover
                                cand_indices = get_covered_indices(candidate)
                                gain = sum(1 for ci in cand_indices if counts[ci] <= 1)
                                if gain > best_gain:
                                    best_gain = gain
                                    best_add = candidate
                    # Also consider w itself
                    if w not in codewords:
                        cand_indices = get_covered_indices(w)
                        gain = sum(1 for ci in cand_indices if counts[ci] <= 1)
                        if gain > best_gain:
                            best_gain = gain
                            best_add = w

                if best_add is not None and best_gain > 0:
                    add_codeword(best_add)
                    # Try to remove two
                    removed_count = 0
                    indices_to_try = list(range(len(codewords)))
                    random.shuffle(indices_to_try)
                    for i in indices_to_try:
                        if time.time() > deadline - 5.0:
                            break
                        if removed_count >= 2:
                            break
                        if i < len(codewords) and is_redundant(i):
                            remove_codeword(i)
                            removed_count += 1

                    if len(codewords) < best_len:
                        best_len = len(codewords)
                        best_code = [cw[:] for cw in codewords]

    # Final cleanup: one more pass of greedy removal
    if time.time() < deadline:
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
            best_len = len(codewords)
            best_code = [cw[:] for cw in codewords]

    return best_code
# EVOLVE-BLOCK-END
