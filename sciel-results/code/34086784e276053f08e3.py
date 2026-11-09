# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
from collections import defaultdict

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin before 120s limit

    # For the specific task: q=3, n=8, radius=1
    # We'll implement a greedy + local search approach
    # that starts from a good initial covering and tries to remove/swap codewords.

    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters: use direct sum construction
        if n >= 4:
            tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            # For small n, just return all words (trivial covering)
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specialized for q=3, n=8, radius=1
    # Total space: 3^8 = 6561 words
    # Each codeword covers 1 + 2*8 = 17 words (itself + 16 neighbors)
    # Lower bound: ceil(6561 / 17) = 386, best known: 486

    # Step 1: Generate all 6561 words
    all_words = np.array(list(itertools.product(range(3), repeat=8)), dtype=np.int8)
    total_words = len(all_words)  # 6561

    # Precompute: for each word, its 17 neighbors (itself + 16 at distance 1)
    # Represent each word as an integer 0..6560 for efficient indexing
    # word_to_idx: map from integer representation to index
    # We'll use a lookup: word as base-3 number -> index

    # Convert all_words to integer representations
    word_ints = np.zeros(total_words, dtype=np.int32)
    for i in range(total_words):
        v = 0
        for j in range(8):
            v = v * 3 + all_words[i][j]
        word_ints[i] = v

    # Build mapping from int to index
    int_to_idx = {}
    for i in range(total_words):
        int_to_idx[word_ints[i]] = i

    # For each word, precompute its 17 neighbors (as indices)
    neighbors = np.zeros((total_words, 17), dtype=np.int32)
    for i in range(total_words):
        w = all_words[i]
        neighbors[i, 0] = i  # itself
        for pos in range(8):
            for delta in [1, 2]:
                new_w = w.copy()
                new_w[pos] = (new_w[pos] + delta) % 3
                idx = int_to_idx[int(''.join(str(x) for x in new_w), 3)]
                # Actually let's compute the int properly
                v = 0
                for j in range(8):
                    v = v * 3 + new_w[j]
                neighbors[i, 1 + pos * 2 + (delta - 1)] = int_to_idx[v]

    # Step 2: Greedy set cover to get an initial covering
    # For each word, count how many times it's covered by the current set
    # Start with empty set, repeatedly add the word that covers the most uncovered words

    covered = np.zeros(total_words, dtype=np.int8)  # 0 = uncovered, 1 = covered
    codewords = []  # list of indices into all_words
    codeword_set = set()

    # For efficiency, maintain count of uncovered words each candidate would cover
    # Initially all 6561 are uncovered
    uncovered_count = total_words

    # Greedy: pick word that covers most uncovered
    # This is O(n * total_words) per iteration, which is 8 * 6561 * ~486 ≈ 25M ops
    # That might be slow in Python. Let's use numpy.

    # Actually, let's use a smarter approach. Precompute for each word which 17 words it covers.
    # We have neighbors[i] = the 17 words covered by word i.

    # Greedy set cover with numpy
    uncovered = np.ones(total_words, dtype=np.bool_)
    codewords = []

    while uncovered.any():
        if time.time() > deadline - 5.0:
            break
        # For each word i, count how many of its 17 neighbors are uncovered
        # covered_by_i = neighbors[i] -> 17 indices
        # We need sum(uncovered[neighbors[i]] for each i)
        # This is a matrix multiply or fancy indexing

        # Use fancy indexing: uncovered[neighbors] gives shape (total_words, 17)
        # Sum over axis 1
        gains = uncovered[neighbors].sum(axis=1)  # shape (total_words,)
        best = np.argmax(gains)
        if gains[best] == 0:
            break
        codewords.append(best)
        codeword_set.add(best)
        uncovered[neighbors[best]] = False
        uncovered_count -= gains[best]

    initial_size = len(codewords)

    # Step 3: Local search - try to remove codewords
    # For each codeword, check if removing it still leaves all words covered
    # A word is "critically covered" by a codeword if it's covered by only that codeword
    # among the current set.

    def compute_coverage(cw_list):
        """Compute coverage count for each word given a list of codeword indices."""
        cov = np.zeros(total_words, dtype=np.int8)
        for cw in cw_list:
            cov[neighbors[cw]] += 1
        return cov

    def is_valid_covering(cw_list):
        """Check if every word is covered by at least one codeword."""
        if not cw_list:
            return total_words == 0
        cov = np.zeros(total_words, dtype=np.int8)
        for cw in cw_list:
            cov[neighbors[cw]] += 1
        return (cov > 0).all()

    # Local search: iteratively try to remove a codeword
    cw_list = list(codewords)
    improved = True
    while improved:
        improved = False
        if time.time() > deadline - 10.0:
            break
        cov = compute_coverage(cw_list)
        for i in range(len(cw_list)):
            if time.time() > deadline - 10.0:
                break
            # Check if removing cw_list[i] is safe
            # A word is safe if cov[word] >= 2 OR (cov[word] == 1 and word is not covered by cw_list[i])
            # Actually: after removing, cov[word] decreases by 1 if word is in neighbors[cw_list[i]]
            # So we need: for all w in neighbors[cw_list[i]], cov[w] >= 2
            # For w not in neighbors[cw_list[i]], cov[w] stays the same and is already >= 1
            cw = cw_list[i]
            # Check if all words covered by this codeword are covered by at least 2 codewords
            if (cov[neighbors[cw]] >= 2).all():
                # Safe to remove
                cw_list.pop(i)
                improved = True
                break

    # Step 4: Try swaps - replace a codeword with another that covers the same critical words
    # and potentially more
    for iteration in range(50):  # limited iterations
        if time.time() > deadline - 5.0:
            break
        cov = compute_coverage(cw_list)
        # Find words covered by exactly 1 codeword (critical)
        # For each codeword, check if it's the sole cover of any word
        swap_made = False
        for i in range(len(cw_list)):
            if time.time() > deadline - 5.0:
                break
            cw = cw_list[i]
            # Words that are critically covered by this codeword
            critical = neighbors[cw][cov[neighbors[cw]] == 1]
            if len(critical) == 0:
                # This codeword has no critical words, can try to remove (already handled above)
                continue
            # Try to find a replacement that covers all critical words
            # A replacement word r must satisfy: all critical words are in neighbors[r]
            # And ideally covers more uncovered or singly-covered words
            # For simplicity, just try random candidates
            for _ in range(20):
                if time.time() > deadline - 5.0:
                    break
                r = np.random.randint(total_words)
                if r in cw_list:
                    continue
                # Check if r covers all critical words
                r_neighbors_set = set(neighbors[r].tolist())
                if all(c in r_neighbors_set for c in critical.tolist()):
                    # Check if swapping doesn't break coverage
                    # After swap: remove cw, add r
                    # New coverage: cov - indicator(neighbors[cw]) + indicator(neighbors[r])
                    # We need new_cov >= 1 for all words
                    # Words at risk: those in neighbors[cw] but not in neighbors[r]
                    at_risk = set(neighbors[cw].tolist()) - r_neighbors_set
                    safe = True
                    for w in at_risk:
                        if cov[w] <= 1:
                            safe = False
                            break
                    if safe:
                        cw_list[i] = r
                        swap_made = True
                        break
            if swap_made:
                break
        if not swap_made:
            break

    # Step 5: Final removal pass
    improved = True
    while improved:
        improved = False
        if time.time() > deadline - 2.0:
            break
        cov = compute_coverage(cw_list)
        for i in range(len(cw_list)):
            if time.time() > deadline - 2.0:
                break
            cw = cw_list[i]
            if (cov[neighbors[cw]] >= 2).all():
                cw_list.pop(i)
                improved = True
                break

    # Convert back to list of lists
    result = [all_words[i].tolist() for i in cw_list]

    # Ensure distinctness (should be by construction)
    seen = set()
    unique_result = []
    for w in result:
        t = tuple(w)
        if t not in seen:
            seen.add(t)
            unique_result.append(w)

    return unique_result

# EVOLVE-BLOCK-END
