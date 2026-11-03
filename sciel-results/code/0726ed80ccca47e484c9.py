"""Structural change: vectorized greedy set-cover with randomized restarts + aggressive local search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    total = q ** n  # 2187 for q=3, n=7

    # Precompute the cover mask for every word as a numpy bool array of length total
    # cover_mask[i] is True for all words within Hamming distance <= radius of word i
    # Build neighbor lists: for each word, all words within distance radius
    # For radius=1: word itself + n*(q-1) neighbors = 1 + 7*2 = 15 words

    # Precompute all words as a 2D array (total x n)
    words = np.zeros((total, n), dtype=np.int8)
    for i in range(total):
        v = i
        for j in range(n - 1, -1, -1):
            words[i, j] = v % q
            v //= q

    # Precompute cover sets: for each word, which words it covers
    # cover_set[i] = set of indices within distance radius of word i
    # For efficiency, use bit arrays: cover_bits[i] is a numpy bool array
    # But 2187 * 2187 booleans = ~4.8M booleans = ~4.8MB, fine

    # Actually let's use a more compact representation.
    # For radius=1, each word covers 1 + n*(q-1) = 15 words.
    # Let's precompute the list of covered indices for each word.
    cover_lists = []
    for i in range(total):
        covered = [i]
        for pos in range(n):
            orig = words[i, pos]
            for val in range(q):
                if val != orig:
                    # Compute index of neighbor
                    # words[i] with position pos changed to val
                    # index = sum(words[i,j] * q^(n-1-j))
                    # Let's compute directly
                    idx = 0
                    for j in range(n):
                        if j == pos:
                            idx = idx * q + val
                        else:
                            idx = idx * q + words[i, j]
                    covered.append(idx)
        cover_lists.append(covered)

    # Build a numpy array: cover_matrix[i, j] = True if j is covered by i
    cover_matrix = np.zeros((total, total), dtype=bool)
    for i in range(total):
        for j in cover_lists[i]:
            cover_matrix[i, j] = True

    # Also build: for each word j, which codewords cover it (reverse mapping)
    # covered_by[j] = list of i such that cover_matrix[i, j]
    # This is useful for checking if a word is still covered after removal
    covered_by = [[] for _ in range(total)]
    for i in range(total):
        for j in cover_lists[i]:
            covered_by[j].append(i)

    # Greedy set cover: start with all words, greedily pick the one covering most uncovered
    # But we want to MINIMIZE codewords, so we want a small covering set.
    # Greedy: repeatedly pick the codeword that covers the most currently-uncovered words.

    def greedy_cover():
        """Greedy set cover: return list of codeword indices."""
        uncovered = np.ones(total, dtype=bool)
        code = []
        # Count how many uncovered words each codeword covers
        # cover_matrix[i] & uncovered -> count
        remaining = total
        while remaining > 0:
            if time.time() > deadline:
                break
            # Find the codeword covering the most uncovered words
            counts = (cover_matrix & uncovered[np.newaxis, :]).sum(axis=1)
            best = int(np.argmax(counts))
            if counts[best] == 0:
                break
            code.append(best)
            uncovered[cover_lists[best]] = False
            remaining -= counts[best]
        return code

    def is_valid_code(code_idx):
        """Check if code_idx covers all words."""
        covered = np.zeros(total, dtype=bool)
        for ci in code_idx:
            covered[cover_lists[ci]] = True
        return covered.all()

    def local_search_reduce(code_idx, max_rounds=50):
        """Try to remove or swap codewords to reduce size while maintaining coverage."""
        code = list(code_idx)
        code_set = set(code)
        for rnd in range(max_rounds):
            if time.time() > deadline:
                break
            improved = False
            # Try removing each codeword
            for i in range(len(code)):
                if time.time() > deadline:
                    break
                # Check if removing code[i] still covers everything
                # A word is uncovered if it's only covered by code[i]
                ci = code[i]
                can_remove = True
                for j in cover_lists[ci]:
                    # Check if any other codeword in code covers j
                    if len(covered_by[j]) <= 1:
                        can_remove = False
                        break
                    # More efficient: check if any other codeword covers j
                    found = False
                    for k in code:
                        if k != ci and cover_matrix[k, j]:
                            found = True
                            break
                    if not found:
                        can_remove = False
                        break
                if can_remove:
                    code.pop(i)
                    code_set.discard(ci)
                    improved = True
                    break

            if improved:
                continue

            # Try swapping: remove one, add a better one
            for i in range(len(code)):
                if time.time() > deadline:
                    break
                ci = code[i]
                # Words that would become uncovered if we remove ci
                # (only those not covered by any other codeword)
                at_risk = []
                for j in cover_lists[ci]:
                    found = False
                    for k in code:
                        if k != ci and cover_matrix[k, j]:
                            found = True
                            break
                    if not found:
                        at_risk.append(j)

                if not at_risk:
                    # Can just remove
                    code.pop(i)
                    code_set.discard(ci)
                    improved = True
                    break

                # Find a replacement that covers all at_risk words
                # and is not already in code
                # The replacement must cover all at_risk words
                # Candidate: intersection of covered_by for all at_risk words
                candidates = None
                for j in at_risk:
                    if candidates is None:
                        candidates = set(covered_by[j])
                    else:
                        candidates &= set(covered_by[j])

                if candidates:
                    candidates -= code_set
                    # Pick the one that covers the most words overall (greedy)
                    best_cand = -1
                    best_count = 0
                    for cand in candidates:
                        cnt = sum(cover_matrix[cand])
                        if cnt > best_count:
                            best_count = cnt
                            best_cand = cand
                    if best_cand >= 0:
                        code[i] = best_cand
                        code_set.discard(ci)
                        code_set.add(best_cand)
                        improved = True
                        break

            if not improved:
                break
        return code

    # Strategy: multiple randomized greedy + local search
    best_code = None
    best_size = total  # worst case: all words

    # First: deterministic greedy
    code = greedy_cover()
    code = local_search_reduce(code)
    if is_valid_code(code):
        best_code = code
        best_size = len(code)

    # Randomized restarts: shuffle the order in which we consider candidates
    rng = random.Random(42)
    num_restarts = 200
    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Randomized greedy: add noise to the counts
        uncovered = np.ones(total, dtype=bool)
        code = []
        remaining = total
        while remaining > 0:
            if time.time() > deadline:
                break
            counts = (cover_matrix & uncovered[np.newaxis, :]).sum(axis=1).astype(float)
            # Add small random noise to break ties and explore different paths
            noise = rng.random(total) * 0.1
            counts += noise
            best = int(np.argmax(counts))
            if counts[best] < 1:  # no progress
                break
            code.append(best)
            uncovered[cover_lists[best]] = False
            remaining -= int((cover_matrix[best] & uncovered).sum()) if False else 0
            # Recalculate remaining properly
            remaining = int(uncovered.sum())

        if not is_valid_code(code):
            continue

        code = local_search_reduce(code)
        if is_valid_code(code) and len(code) < best_size:
            best_size = len(code)
            best_code = code

    # Final local search with more rounds on the best
    if best_code is not None:
        best_code = local_search_reduce(best_code, max_rounds=100)
        if not is_valid_code(best_code):
            # Fallback
            best_code = greedy_cover()
            best_code = local_search_reduce(best_code)

    if best_code is None:
        best_code = list(range(total))

    # Convert indices to word lists
    result = []
    for ci in best_code:
        result.append(words[ci].tolist())

    return result
# EVOLVE-BLOCK-END
