"""Improved ternary covering code K_3(6,1) via fast greedy + targeted local search.

Strategy:
1. Start from the direct-sum construction (81 words).
2. Use a highly optimized greedy removal with multiple random orderings.
3. Add a targeted local search that tries to replace pairs of codewords with
   a single codeword from a larger candidate pool, and also tries single swaps.
4. Use time management to finish well within the 120s limit.
"""
import itertools
import time
import random
import sys
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0

    # Build the tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(q) for b in range(q)]

    if n <= 4:
        if n == 4:
            return [list(h) for h in tetracode]
        else:
            return [list(c) for c in itertools.product(range(q), repeat=n)]

    # Build initial code: direct sum of tetracode with all tails
    code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            code.append(tuple(h) + tuple(tail))

    base_count = len(code)  # 81

    total_words = q ** n
    code_arr = np.array(code, dtype=np.int32)
    num_codewords = len(code)

    # Precompute powers of q
    q_powers = np.array([q ** i for i in range(n)], dtype=np.int32)
    # q_powers[i] = q^i, so position i (0-indexed from left) has weight q^(n-1-i)
    # Let's use: weight[pos] = q^(n-1-pos)
    weights = np.array([q ** (n - 1 - i) for i in range(n)], dtype=np.int32)

    def compute_word_indices(codewords):
        """Compute the index of each word in the codewords array."""
        # codewords is (m, n) array
        return (codewords * weights).sum(axis=1).astype(np.int32)

    # Compute indices for all initial codewords
    cw_indices = compute_word_indices(code_arr)

    # Precompute coverage for each codeword: list of word indices it covers
    # For radius=1: covers itself + n*(q-1) neighbors
    codeword_covers = []
    for ci in range(num_codewords):
        c = code_arr[ci]
        base_idx = cw_indices[ci]
        covers = [base_idx]
        for pos in range(n):
            orig = c[pos]
            w = weights[pos]
            for val in range(q):
                if val == orig:
                    continue
                covers.append(base_idx + (val - orig) * w)
        codeword_covers.append(covers)

    # Build coverage_count for a given set of active codewords
    def build_coverage_count(active_mask):
        """Build coverage count array given a boolean mask of active codewords."""
        coverage = np.zeros(total_words, dtype=np.int32)
        for ci in range(num_codewords):
            if active_mask[ci]:
                for wi in codeword_covers[ci]:
                    coverage[wi] += 1
        return coverage

    # Greedy removal with a given order
    def greedy_remove(order, initial_active=None):
        """Remove redundant codewords in the given order. Returns active mask."""
        if initial_active is None:
            active = np.ones(num_codewords, dtype=bool)
        else:
            active = initial_active.copy()

        coverage = np.zeros(total_words, dtype=np.int32)
        for ci in range(num_codewords):
            if active[ci]:
                for wi in codeword_covers[ci]:
                    coverage[wi] += 1

        for ci in order:
            if not active[ci]:
                continue
            # Check if redundant
            redundant = True
            for wi in codeword_covers[ci]:
                if coverage[wi] <= 1:
                    redundant = False
                    break
            if redundant:
                active[ci] = False
                for wi in codeword_covers[ci]:
                    coverage[wi] -= 1
        return active

    # Run multiple greedy trials
    best_active = np.ones(num_codewords, dtype=bool)
    best_count = num_codewords

    num_trials = 50
    for trial in range(num_trials):
        if time.time() > deadline - 30.0:
            break

        order = list(range(num_codewords))
        random.shuffle(order)

        active = greedy_remove(order)
        cnt = int(np.sum(active))

        if cnt < best_count:
            best_count = cnt
            best_active = active

        if best_count <= 73:
            break

    # Convert best_active to list of codewords
    best_indices = np.where(best_active)[0]
    best_code = [list(code[i]) for i in best_indices]

    # Build a larger candidate pool for local search
    # Include all 3^n words? That's 729, manageable.
    # But we only need to consider words that might be useful.
    # Let's precompute covers for all 729 words.
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    all_word_indices = compute_word_indices(all_words)

    # Precompute covers for all words
    all_covers = []
    for wi in range(total_words):
        c = all_words[wi]
        base_idx = all_word_indices[wi]
        covers = [base_idx]
        for pos in range(n):
            orig = c[pos]
            w = weights[pos]
            for val in range(q):
                if val == orig:
                    continue
                covers.append(base_idx + (val - orig) * w)
        all_covers.append(covers)

    # Local search: try to reduce the code further
    # We'll work with the current best code and try:
    # 1. Remove a codeword if redundant
    # 2. Swap: remove codeword i, add codeword j (from all_words)
    # 3. Replace two codewords with one

    def get_current_coverage(codeword_list):
        """Get coverage count for a list of codewords (as tuples)."""
        coverage = np.zeros(total_words, dtype=np.int32)
        for c in codeword_list:
            c_arr = np.array(c, dtype=np.int32)
            base_idx = int(compute_word_indices(c_arr.reshape(1, -1))[0])
            for pos in range(n):
                orig = c[pos]
                w = weights[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    coverage[base_idx + (val - orig) * w] += 1
            coverage[base_idx] += 1
        return coverage

    # Work with the best code found
    current_code = best_code
    current_n = len(current_code)

    # Build coverage for current code
    current_coverage = get_current_coverage(current_code)

    # Set of current codeword indices
    current_set = set()
    for c in current_code:
        c_arr = np.array(c, dtype=np.int32)
        idx = int(compute_word_indices(c_arr.reshape(1, -1))[0])
        current_set.add(idx)

    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        current_n = len(current_code)

        # Try single removals
        for bi in range(current_n):
            if time.time() > deadline - 5.0:
                break

            c = current_code[bi]
            c_arr = np.array(c, dtype=np.int32)
            base_idx = int(compute_word_indices(c_arr.reshape(1, -1))[0])
            covers_bi = set()
            covers_bi.add(base_idx)
            for pos in range(n):
                orig = c[pos]
                w = weights[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    covers_bi.add(base_idx + (val - orig) * w)

            # Check if redundant
            uncovered = [wi for wi in covers_bi if current_coverage[wi] <= 1]
            if len(uncovered) == 0:
                # Remove
                for wi in covers_bi:
                    current_coverage[wi] -= 1
                current_code.pop(bi)
                current_n -= 1
                improved = True
                break

            # Try swap: find a word that covers all uncovered words
            if len(uncovered) <= 4 and time.time() < deadline - 5.0:
                # For each candidate word, check if it covers all uncovered
                # We can limit candidates to words that are close to the removed codeword
                # or words that cover at least one of the uncovered words
                found_swap = False
                for wi in range(total_words):
                    if time.time() > deadline - 5.0:
                        break
                    if wi in current_set:
                        continue

                    cand_covers = all_covers[wi]
                    cand_set = set(cand_covers)
                    if all(u in cand_set for u in uncovered):
                        # Perform swap
                        # Remove bi
                        for w in covers_bi:
                            current_coverage[w] -= 1
                        # Add wi
                        for w in cand_covers:
                            current_coverage[w] += 1

                        # Update code
                        current_code[bi] = list(all_words[wi])
                        # Update set
                        old_idx = base_idx
                        current_set.discard(old_idx)
                        current_set.add(wi)
                        improved = True
                        found_swap = True
                        break

                if found_swap:
                    break

        # Try pair replacement: remove two codewords, add one
        if not improved and current_n > 2 and time.time() < deadline - 10.0:
            # This is more expensive, so do it less frequently
            for bi in range(current_n):
                if time.time() > deadline - 5.0:
                    break
                for bj in range(bi + 1, current_n):
                    if time.time() > deadline - 5.0:
                        break

                    c1 = current_code[bi]
                    c2 = current_code[bj]
                    c1_arr = np.array(c1, dtype=np.int32)
                    c2_arr = np.array(c2, dtype=np.int32)
                    idx1 = int(compute_word_indices(c1_arr.reshape(1, -1))[0])
                    idx2 = int(compute_word_indices(c2_arr.reshape(1, -1))[0])

                    covers1 = set()
                    covers1.add(idx1)
                    for pos in range(n):
                        orig = c1[pos]
                        w = weights[pos]
                        for val in range(q):
                            if val == orig:
                                continue
                            covers1.add(idx1 + (val - orig) * w)

                    covers2 = set()
                    covers2.add(idx2)
                    for pos in range(n):
                        orig = c2[pos]
                        w = weights[pos]
                        for val in range(q):
                            if val == orig:
                                continue
                            covers2.add(idx2 + (val - orig) * w)

                    # Words that would become uncovered
                    uncovered = set()
                    for w in covers1:
                        if current_coverage[w] <= 1:
                            uncovered.add(w)
                    for w in covers2:
                        if current_coverage[w] <= 1 and w not in covers1:
                            uncovered.add(w)
                        elif w in covers1 and current_coverage[w] <= 2:
                            # After removing both, if coverage was 2, it becomes 0
                            # But we already counted it in covers1 check
                            pass

                    # More carefully: compute coverage after removing both
                    # A word w is uncovered if:
                    # - w in covers1 and current_coverage[w] == 1 (only covered by c1)
                    # - w in covers2 and current_coverage[w] == 1 (only covered by c2)
                    # - w in both covers1 and covers2 and current_coverage[w] == 2 (covered only by c1 and c2)

                    uncovered = set()
                    for w in covers1:
                        if current_coverage[w] == 1:
                            uncovered.add(w)
                    for w in covers2:
                        if current_coverage[w] == 1:
                            uncovered.add(w)
                        elif current_coverage[w] == 2 and w in covers1:
                            uncovered.add(w)

                    if len(uncovered) == 0:
                        # Can remove both
                        for w in covers1:
                            current_coverage[w] -= 1
                        for w in covers2:
                            current_coverage[w] -= 1
                        # Remove from code (remove bj first since it's later)
                        current_code.pop(bj)
                        current_code.pop(bi)
                        current_set.discard(idx1)
                        current_set.discard(idx2)
                        current_n -= 2
                        improved = True
                        break

                    # Try to find a single word that covers all uncovered
                    if len(uncovered) <= 5 and time.time() < deadline - 5.0:
                        found = False
                        for wi in range(total_words):
                            if time.time() > deadline - 5.0:
                                break
                            if wi in current_set:
                                continue
                            cand_set = set(all_covers[wi])
                            if all(u in cand_set for u in uncovered):
                                # Perform replacement
                                for w in covers1:
                                    current_coverage[w] -= 1
                                for w in covers2:
                                    current_coverage[w] -= 1
                                for w in all_covers[wi]:
                                    current_coverage[w] += 1

                                current_code[bi] = list(all_words[wi])
                                current_code.pop(bj)
                                current_set.discard(idx1)
                                current_set.discard(idx2)
                                current_set.add(wi)
                                current_n -= 1
                                improved = True
                                found = True
                                break
                        if found:
                            break

                if improved:
                    break

    # Verify coverage
    if time.time() < deadline - 2.0 and len(current_code) <= 100:
        covered_set = set()
        for c in current_code:
            c_arr = np.array(c, dtype=np.int32)
            base_idx = int(compute_word_indices(c_arr.reshape(1, -1))[0])
            covered_set.add(base_idx)
            for pos in range(n):
                orig = c[pos]
                w = weights[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    covered_set.add(base_idx + (val - orig) * w)

        if len(covered_set) < total_words:
            # Fallback to the best valid code from greedy
            current_code = best_code

    return current_code
