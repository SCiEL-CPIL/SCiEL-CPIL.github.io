"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    # Deadline guard: keep a safety margin before the 120s evaluator limit.
    deadline = time.time() + 110.0

    if q != 3 or n != 6 or radius != 1:
        # Fallback for other parameters: greedy set cover.
        return _greedy_set_cover(q, n, radius, deadline)

    total = q ** n  # 729

    # Precompute neighbors for each word (index 0..728)
    # A word is an integer 0..728 representing base-3 number
    # neighbors[i] = list of indices at Hamming distance 1 from i

    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute all neighbors using vectorized approach
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx_map = {w: i for i, w in enumerate(all_words)}

    # Build neighbors list efficiently
    neighbors = [[] for _ in range(total)]
    for i in range(total):
        w = idx_to_word(i)
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    nw = list(w)
                    nw[pos] = val
                    j = word_to_idx_map[tuple(nw)]
                    neighbors[i].append(j)

    # Convert neighbors to numpy arrays for vectorized operations
    neighbors_np = np.array(neighbors, dtype=np.int32)

    # Coverage matrix: cov_matrix[i, j] = 1 if word i is within distance 1 of word j
    # Shape: (total, total) = (729, 729)
    cov_matrix = np.zeros((total, total), dtype=np.int8)
    for i in range(total):
        cov_matrix[i, i] = 1
        for nb in neighbors[i]:
            cov_matrix[i, nb] = 1

    # Phase 1: Greedy set cover to build initial code
    uncovered = np.ones(total, dtype=bool)
    code_indices = []

    while np.any(uncovered):
        if time.time() > deadline - 30:
            break
        # For each candidate codeword, count how many uncovered words it covers
        # cov_matrix[:, i] gives the words covered by codeword i
        # We want the sum of uncovered words covered by each candidate
        # scores[i] = sum(uncovered & cov_matrix[:, i])
        scores = np.sum(uncovered[:, np.newaxis] & cov_matrix, axis=0)
        best_i = int(np.argmax(scores))
        code_indices.append(best_i)
        uncovered &= ~cov_matrix[:, best_i].astype(bool)

    code = np.array([idx_to_word(i) for i in code_indices])
    code_set = set(code_indices)

    # Build coverage count
    coverage_count = np.zeros(total, dtype=np.int32)
    for ci in code_indices:
        coverage_count[cov_matrix[:, ci] == 1] += 1

    # Phase 2: Local search - try to remove redundant codewords
    improved = True
    while improved:
        if time.time() > deadline - 15:
            break
        improved = False
        for i in range(len(code_indices)):
            if time.time() > deadline - 15:
                break
            ci = code_indices[i]
            # Check if removing ci still covers all words
            # Words at risk: those where coverage_count == 1 and covered by ci
            at_risk = (cov_matrix[:, ci] == 1) & (coverage_count == 1)
            if not np.any(at_risk):
                # Safe to remove
                coverage_count[cov_matrix[:, ci] == 1] -= 1
                code_indices.pop(i)
                code_set.discard(ci)
                improved = True
                break

    # Phase 3: Local search - try to replace codewords with better ones
    # For each codeword, try swapping it with another word that covers more uncovered territory
    for iteration in range(50):
        if time.time() > deadline - 10:
            break
        made_swap = False
        for i in range(len(code_indices)):
            if time.time() > deadline - 10:
                break
            ci = code_indices[i]
            # Try removing ci and see what becomes uncovered
            temp_cov = coverage_count.copy()
            temp_cov[cov_matrix[:, ci] == 1] -= 1
            uncovered_after = temp_cov == 0

            if not np.any(uncovered_after):
                # Can remove entirely
                coverage_count[cov_matrix[:, ci] == 1] -= 1
                code_indices.pop(i)
                code_set.discard(ci)
                made_swap = True
                break

            # Find the best replacement: a word not in code_set that covers most of uncovered_after
            # But we need to be careful: the replacement should cover the uncovered words
            # and ideally also cover words that were previously covered by ci
            candidates = [j for j in range(total) if j not in code_set]
            if not candidates:
                continue

            # Score each candidate by how many uncovered_after words it covers
            cand_arr = np.array(candidates, dtype=np.int32)
            # For each candidate, count uncovered_after words it covers
            # uncovered_after[:, np.newaxis] & cov_matrix[:, cand_arr[j]]
            # This is expensive; do it in chunks
            chunk_size = 100
            best_cand = -1
            best_score = 0
            for start in range(0, len(cand_arr), chunk_size):
                end = min(start + chunk_size, len(cand_arr))
                chunk = cand_arr[start:end]
                # scores[j] = sum(uncovered_after & cov_matrix[:, chunk[j]])
                scores = np.sum(uncovered_after[:, np.newaxis] & cov_matrix[:, chunk], axis=0)
                local_best = int(np.argmax(scores))
                if scores[local_best] > best_score:
                    best_score = scores[local_best]
                    best_cand = chunk[local_best]

            if best_cand >= 0 and best_score > 0:
                # Check if replacing ci with best_cand is beneficial
                # New coverage after replacement
                new_temp_cov = temp_cov.copy()
                new_temp_cov[cov_matrix[:, best_cand] == 1] += 1
                if np.all(new_temp_cov >= 1):
                    # Valid replacement
                    coverage_count = new_temp_cov
                    code_indices[i] = best_cand
                    code_set.discard(ci)
                    code_set.add(best_cand)
                    made_swap = True
                    break

        if not made_swap:
            break

    # Phase 4: Final pass - try more aggressive local search
    # Try removing multiple codewords at once and re-covering
    for iteration in range(20):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(len(code_indices)):
            if time.time() > deadline - 5:
                break
            ci = code_indices[i]
            # Try removing ci
            temp_cov = coverage_count.copy()
            temp_cov[cov_matrix[:, ci] == 1] -= 1
            uncovered_after = temp_cov == 0

            if not np.any(uncovered_after):
                coverage_count[cov_matrix[:, ci] == 1] -= 1
                code_indices.pop(i)
                code_set.discard(ci)
                improved = True
                break

            # Try to cover uncovered words with a single new codeword
            if np.sum(uncovered_after) <= 13:  # Max coverage of one codeword
                # Find if there's a word that covers all uncovered_after
                # The intersection of neighborhoods of all uncovered words
                # A word w covers uncovered_after if for all u in uncovered_after, u is within dist 1 of w
                # This means w is in the intersection of N(u) for all u in uncovered_after
                unc_list = np.where(uncovered_after)[0]
                if len(unc_list) == 1:
                    # Just need any word that covers this one word
                    # The word itself works
                    u = int(unc_list[0])
                    if u not in code_set:
                        new_cov = temp_cov.copy()
                        new_cov[cov_matrix[:, u] == 1] += 1
                        if np.all(new_cov >= 1):
                            coverage_count = new_cov
                            code_indices[i] = u
                            code_set.discard(ci)
                            code_set.add(u)
                            improved = True
                            break
                    else:
                        # Try neighbors of u
                        for nb in neighbors[u]:
                            if nb not in code_set:
                                new_cov = temp_cov.copy()
                                new_cov[cov_matrix[:, nb] == 1] += 1
                                if np.all(new_cov >= 1):
                                    coverage_count = new_cov
                                    code_indices[i] = nb
                                    code_set.discard(ci)
                                    code_set.add(nb)
                                    improved = True
                                    break
                        if improved:
                            break
                else:
                    # Find intersection of neighborhoods
                    # A word w covers all unc_list if w is in N(u) for all u in unc_list
                    # N(u) = {u} ∪ neighbors[u]
                    candidate_set = set([int(unc_list[0])] + [int(x) for x in neighbors[unc_list[0]]])
                    for u in unc_list[1:]:
                        candidate_set &= set([int(u)] + [int(x) for x in neighbors[u]])
                        if not candidate_set:
                            break
                    if candidate_set:
                        # Find a candidate not in code_set
                        for cand in candidate_set:
                            if cand not in code_set:
                                new_cov = temp_cov.copy()
                                new_cov[cov_matrix[:, cand] == 1] += 1
                                if np.all(new_cov >= 1):
                                    coverage_count = new_cov
                                    code_indices[i] = cand
                                    code_set.discard(ci)
                                    code_set.add(cand)
                                    improved = True
                                    break
                        if improved:
                            break

        if not improved:
            break

    # Rebuild code from code_indices
    code = np.array([idx_to_word(i) for i in code_indices])

    # Return as list of lists
    return [list(cw) for cw in code]


def _greedy_set_cover(q, n, radius, deadline):
    """Greedy set cover for general parameters."""
    all_words = list(itertools.product(range(q), repeat=n))
    total = q ** n

    # Precompute neighbors within radius
    def hamming_dist(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)

    # For each word, find all words within radius
    coverage = {}
    for i, w in enumerate(all_words):
        covered = set()
        for j, v in enumerate(all_words):
            if hamming_dist(w, v) <= radius:
                covered.add(j)
        coverage[i] = covered

    uncovered = set(range(total))
    code = []

    while uncovered and time.time() < deadline - 5:
        # Find word that covers most uncovered
        best_i = -1
        best_count = 0
        for i in range(total):
            count = len(uncovered & coverage[i])
            if count > best_count:
                best_count = count
                best_i = i
            if time.time() > deadline - 5:
                break
        if best_i == -1:
            break
        code.append(all_words[best_i])
        uncovered -= coverage[best_i]

    return code
# EVOLVE-BLOCK-END
