"""Improved ternary covering code K_3(6,1) via local search (add-then-remove).

Strategy:
1. Start from the direct-sum construction (81 words).
2. Greedy removal to get an initial smaller code (~73-75).
3. Local search: repeatedly try to remove a codeword; if that fails, try adding
   a new codeword that covers the most uncovered words, then re-run greedy removal.
   This add-then-remove heuristic can escape local minima of pure greedy removal.
4. Use vectorized NumPy operations for speed where possible.
"""
import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    # For the specific problem: q=3, n=6, radius=1
    # Direct sum of ternary tetracode (9 words of length 4) with all 9 tails
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]

    # Build initial code: direct sum
    code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            code.append(tuple(h) + tuple(tail))

    base_count = len(code)  # 81 for q=3, n=6

    if n <= 4:
        if n == 4:
            return [list(h) for h in tetracode]
        else:
            result = [list(c) for c in itertools.product(range(q), repeat=n)]
            return result

    total_words = q ** n

    # Precompute ball: for each word index, list of word indices within Hamming distance radius
    # For radius=1: word itself + all words at distance 1
    # Ball size = 1 + n*(q-1) = 1 + 6*2 = 13 for q=3, n=6, r=1

    def word_to_index(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    # Precompute the ball (list of word indices) for each word index
    # ball[wi] = list of word indices within distance radius from word wi
    ball = [None] * total_words
    for idx in range(total_words):
        w = []
        tmp = idx
        for i in range(n):
            w.append(tmp % q)
            tmp //= q
        w = tuple(reversed(w))
        ball_words = [idx]
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                w_list = list(w)
                w_list[pos] = val
                ball_words.append(word_to_index(tuple(w_list)))
        ball[idx] = ball_words

    # For each codeword, the ball gives which words it covers
    # codeword_covers[ci] = list of word indices covered by code[ci]
    codeword_covers = [ball[word_to_index(c)] for c in code]

    # covered_count[wi] = number of active codewords covering word wi
    # Also need to know which codewords cover each word for redundancy check
    # Use a 2D array: cover_matrix[wi][ci] = 1 if codeword ci covers word wi
    # This is 729 x 81 booleans = ~59k entries, manageable

    # Build coverage matrix
    cover_matrix = np.zeros((total_words, len(code)), dtype=np.int8)
    for ci in range(len(code)):
        for wi in codeword_covers[ci]:
            cover_matrix[wi, ci] = 1

    # cover_count[wi] = sum over ci of cover_matrix[wi, ci]
    cover_count = cover_matrix.sum(axis=1)

    def is_redundant(ci, active):
        """Check if codeword ci is redundant given active codewords."""
        # For each word covered by ci, check if another active codeword covers it
        for wi in codeword_covers[ci]:
            # Count active codewords covering wi
            # If only ci covers it, not redundant
            count = 0
            for cj in range(len(code)):
                if active[cj] and cover_matrix[wi, cj]:
                    count += 1
                    if count > 1:
                        break
            if count <= 1:
                return False
        return True

    def greedy_remove(code_list, active, codeword_covers, cover_matrix, cover_count):
        """Greedy removal: repeatedly remove redundant codewords."""
        removed = True
        while removed:
            removed = False
            for ci in range(len(code_list)):
                if not active[ci]:
                    continue
                # Check redundancy
                redundant = True
                for wi in codeword_covers[ci]:
                    # Check if any other active codeword covers wi
                    found = False
                    for cj in range(len(code_list)):
                        if cj != ci and active[cj] and cover_matrix[wi, cj]:
                            found = True
                            break
                    if not found:
                        redundant = False
                        break
                if redundant:
                    active[ci] = False
                    # Update cover_count
                    for wi in codeword_covers[ci]:
                        cover_count[wi] -= 1
                    removed = True
        return active

    # Initial greedy removal
    active = [True] * len(code)
    active = greedy_remove(code, active, codeword_covers, cover_matrix, cover_count)

    # Get initial code size
    initial_size = sum(active)

    # Local search: add-then-remove
    # Maintain a pool of candidate codewords (all 729 words)
    best_code = [code[i] for i in range(len(code)) if active[i]]
    best_size = len(best_code)

    # For local search, we need to be able to add new codewords
    # We'll work with a list of codeword indices (into the full space of 729 words)
    # Represent codewords as word indices

    all_words = list(range(total_words))

    def get_code_indices(active_list, code_list):
        """Get list of word indices for active codewords."""
        return [word_to_index(code_list[i]) for i in range(len(code_list)) if active_list[i]]

    def coverage_check(codeword_indices):
        """Check if a set of codeword indices covers all words. Returns uncovered count."""
        # For each word, check if any codeword covers it
        # Use the ball: word wi is covered by codeword ci if wi is in ball[ci]
        # Equivalently, ci is in ball[wi]
        covered = np.zeros(total_words, dtype=bool)
        for ci_idx in codeword_indices:
            for wi in ball[ci_idx]:
                covered[wi] = True
        return int((~covered).sum())

    def try_remove(codeword_indices, idx_to_remove):
        """Try to remove codeword at position idx_to_remove. Return new list if successful."""
        remaining = codeword_indices[:idx_to_remove] + codeword_indices[idx_to_remove+1:]
        if coverage_check(remaining) == 0:
            return remaining
        return None

    def find_best_add(codeword_indices):
        """Find the word to add that covers the most currently uncovered words."""
        # Compute uncovered words
        covered = np.zeros(total_words, dtype=bool)
        for ci_idx in codeword_indices:
            for wi in ball[ci_idx]:
                covered[wi] = True
        uncovered = np.where(~covered)[0]
        if len(uncovered) == 0:
            return None, 0

        # For each candidate word, count how many uncovered words it covers
        best_candidate = None
        best_count = 0
        for cand in all_words:
            if cand in codeword_indices:
                continue
            count = 0
            for wi in ball[cand]:
                if not covered[wi]:
                    count += 1
            if count > best_count:
                best_count = count
                best_candidate = cand
        return best_candidate, best_count

    # Local search loop
    current = get_code_indices(active, code)
    current_size = len(current)
    best = list(current)
    best_sz = current_size

    no_improve_count = 0
    max_no_improve = 50

    while time.time() < deadline - 10.0:
        if no_improve_count >= max_no_improve:
            break

        # Try to remove a random codeword
        if current_size > 1:
            remove_idx = random.randint(0, current_size - 1)
            new_code = try_remove(current, remove_idx)
            if new_code is not None:
                current = new_code
                current_size = len(current)
                no_improve_count = 0
                if current_size < best_sz:
                    best = list(current)
                    best_sz = current_size
                continue

        # If can't remove, try add-then-remove
        # Add the best candidate
        cand, gain = find_best_add(current)
        if cand is None or gain == 0:
            no_improve_count += 1
            continue

        # Add it
        current.append(cand)
        current_size += 1

        # Now try to remove codewords (including possibly the one we just added or others)
        # Do multiple rounds of greedy removal
        removed_any = True
        while removed_any and time.time() < deadline - 10.0:
            removed_any = False
            # Shuffle order for diversity
            order = list(range(current_size))
            random.shuffle(order)
            for ri in order:
                if ri >= len(current):
                    break
                new_code = try_remove(current, ri)
                if new_code is not None:
                    current = new_code
                    current_size = len(current)
                    removed_any = True
                    if current_size < best_sz:
                        best = list(current)
                        best_sz = current_size
                        no_improve_count = 0
                    break  # restart the scan after a removal

        if not removed_any:
            no_improve_count += 1

    # Convert best indices back to codeword lists
    result = []
    for idx in best:
        w = []
        tmp = idx
        for i in range(n):
            w.append(tmp % q)
            tmp //= q
        result.append(list(reversed(w)))

    # Final verification
    if time.time() < deadline - 3.0:
        covered = set()
        for c in result:
            c = tuple(c)
            covered.add(word_to_index(c))
            for pos in range(n):
                orig = c[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    w_list = list(c)
                    w_list[pos] = val
                    covered.add(word_to_index(tuple(w_list)))
        if len(covered) < total_words:
            # Fallback to initial greedy result
            result = [code[i] for i in range(len(code)) if active[i]]

    return result
