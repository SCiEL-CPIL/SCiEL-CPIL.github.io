# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools


def construct(q, n, radius):
    start = time.time()
    deadline = start + 118  # safety margin within 150s

    total = q ** n  # 729 for q=3, n=6

    # Precompute all words as arrays
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)

    # Build coverage matrix: cover_matrix[i, j] = True if word j is within radius of word i
    # i.e., row i = ball around word i
    # For radius=1, ball size = 1 + n*(q-1) = 1 + 6*2 = 13
    cover_matrix = np.zeros((total, total), dtype=bool)

    # Efficient computation: for each position, for each value change
    if radius >= 1:
        for pos in range(n):
            # For each word, the words at distance 1 in position pos
            # word with value v at pos: neighbors are all words with different value at pos
            # We can do this vectorially
            for val in range(q):
                mask = all_words[:, pos] == val
                for new_val in range(q):
                    if new_val == val:
                        continue
                    # Words that differ only at pos from a word with value val
                    # These are words with value new_val at pos, and same elsewhere
                    # Actually, we need: for word j, is word i within distance 1?
                    # Let me think differently.
                    # cover_matrix[i, j] = 1 if dist(word_i, word_j) <= radius
                    # For radius 1: dist = 0 or 1
                    # dist=0: i==j
                    # dist=1: differ in exactly one position
                    pass

    # Better approach: compute pairwise Hamming distances
    # For n=6, q=3, total=729: 729x729 distance matrix
    # dist[i,j] = sum over positions of (all_words[i,pos] != all_words[j,pos])

    # Vectorized: for each position, compute which pairs differ
    diff_count = np.zeros((total, total), dtype=np.int8)
    for pos in range(n):
        # all_words[:, pos] is shape (total,)
        # diff at pos: (total, 1) != (1, total)
        d = (all_words[:, pos].reshape(-1, 1) != all_words[:, pos].reshape(1, -1)).astype(np.int8)
        diff_count += d

    # Cover matrix: dist <= radius
    cover_matrix = diff_count <= radius

    # Self-coverage
    np.fill_diagonal(cover_matrix, True)

    # Greedy set cover
    uncovered = np.ones(total, dtype=bool)
    codewords = []
    covered_mask = np.zeros(total, dtype=bool)  # cumulative covered points

    while uncovered.any() and time.time() < deadline - 10:
        # For each candidate word, count how many uncovered points it covers
        # cover_matrix[i, :] is the ball around word i
        # We want: for each i, count of j where uncovered[j] and cover_matrix[i, j]
        # That is: cover_matrix[:, uncovered].sum(axis=1)
        counts = cover_matrix[:, uncovered].sum(axis=1)
        best_idx = int(np.argmax(counts))
        if counts[best_idx] == 0:
            break
        codewords.append(best_idx)
        covered_mask |= cover_matrix[best_idx]
        uncovered = ~covered_mask

    # Removal local search: try to remove redundant codewords
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(codewords)):
                if time.time() > deadline - 5:
                    break
                # Check if removing codeword i still covers everything
                # Compute union of all balls except i
                if len(codewords) == 1:
                    continue
                others = [j for j in range(len(codewords)) if j != i]
                union = np.zeros(total, dtype=bool)
                for j in others:
                    union |= cover_matrix[codewords[j]]
                if union.all():
                    codewords.pop(i)
                    improved = True
                    break

    # Convert to list of lists
    result = [list(all_words[i]) for i in codewords]
    return result
# EVOLVE-BLOCK-END
