"""Improved program for K_3(6,1) covering code with symmetry-aware search."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np
from functools import lru_cache

def construct(q, n, radius):
    deadline = time.time() + 115.0

    if q != 3 or n != 6 or radius != 1:
        return _greedy_set_cover(q, n, radius, deadline)

    total = 729

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbors
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx_map = {w: i for i, w in enumerate(all_words)}
    neighbors = [[] for _ in range(total)]
    for i in range(total):
        w = idx_to_word(i)
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    nw = list(w)
                    nw[pos] = val
                    neighbors[i].append(word_to_idx_map[tuple(nw)])

    # Build coverage matrix (symmetric)
    cov_matrix = np.zeros((total, total), dtype=np.uint8)
    for i in range(total):
        cov_matrix[i, i] = 1
        for nb in neighbors[i]:
            cov_matrix[i, nb] = 1
            cov_matrix[nb, i] = 1

    # Symmetry group: coordinate permutations (6! = 720) and symbol permutations (3!^6 but we use coordinate perms for efficiency)
    # For tie-breaking, use a canonical representative under coordinate permutations

    def get_canonical_form(word_tuple):
        """Get canonical form under coordinate permutations (smallest lex representation)."""
        best = word_tuple
        from itertools import permutations
        for perm in permutations(range(n)):
            permuted = tuple(word_tuple[perm[i]] for i in range(n))
            if permuted < best:
                best = permuted
        return best

    # Phase 1: Greedy set cover with symmetry-aware tie-breaking
    uncovered = np.ones(total, dtype=bool)
    code_indices = []
    used_canonical = set()

    while np.any(uncovered):
        if time.time() > deadline - 25:
            break
        scores = np.sum(uncovered[:, np.newaxis] & cov_matrix, axis=0)
        max_score = scores.max()
        # Find all candidates with max score
        candidates = np.where(scores == max_score)[0]

        # Symmetry-aware tie-breaking: prefer candidates whose canonical form hasn't been used
        # This encourages diverse structure
        best_cand = None
        best_key = None
        for cand in candidates:
            word = idx_to_word(int(cand))
            canon = get_canonical_form(tuple(word))
            # Prefer unused canonical forms, then smaller canonical form
            if canon not in used_canonical:
                key = (0, canon)
            else:
                key = (1, canon)
            if best_key is None or key < best_key:
                best_key = key
                best_cand = int(cand)

        code_indices.append(best_cand)
        used_canonical.add(get_canonical_form(tuple(idx_to_word(best_cand))))
        uncovered &= ~cov_matrix[:, best_cand].astype(bool)

    code_set = set(code_indices)

    # Build coverage count
    coverage_count = np.zeros(total, dtype=np.int32)
    for ci in code_indices:
        coverage_count[cov_matrix[:, ci] == 1] += 1

    # Phase 2: Remove redundant codewords
    improved = True
    while improved:
        if time.time() > deadline - 20:
            break
        improved = False
        for i in range(len(code_indices)):
            if time.time() > deadline - 20:
                break
            ci = code_indices[i]
            at_risk = (cov_matrix[:, ci] == 1) & (coverage_count == 1)
            if not np.any(at_risk):
                coverage_count[cov_matrix[:, ci] == 1] -= 1
                code_indices.pop(i)
                code_set.discard(ci)
                improved = True
                break

    # Phase 3: Local search with symmetry-aware swaps
    # Multiple rounds of improvement
    for round_num in range(10):
        if time.time() > deadline - 10:
            break
        made_swap = False
        for i in range(len(code_indices)):
            if time.time() > deadline - 10:
                break
            ci = code_indices[i]
            temp_cov = coverage_count.copy()
            temp_cov[cov_matrix[:, ci] == 1] -= 1
            uncovered_after = temp_cov == 0

            if not np.any(uncovered_after):
                coverage_count[cov_matrix[:, ci] == 1] -= 1
                code_indices.pop(i)
                code_set.discard(ci)
                made_swap = True
                break

            # Find best replacement using symmetry-aware scoring
            candidates = [j for j in range(total) if j not in code_set]
            if not candidates:
                continue

            # Score: number of uncovered_after words covered, with tie-breaking by canonical form novelty
            cand_arr = np.array(candidates, dtype=np.int32)
            chunk_size = 150
            best_cand = -1
            best_score = -1
            best_canon = None

            for start in range(0, len(cand_arr), chunk_size):
                end = min(start + chunk_size, len(cand_arr))
                chunk = cand_arr[start:end]
                scores = np.sum(uncovered_after[:, np.newaxis] & cov_matrix[:, chunk], axis=0)
                max_s = scores.max()
                # Get all candidates in this chunk with max score
                local_candidates = chunk[scores == max_s]
                for lc in local_candidates:
                    lc = int(lc)
                    # Check if replacement is valid
                    new_temp_cov = temp_cov.copy()
                    new_temp_cov[cov_matrix[:, lc] == 1] += 1
                    if np.all(new_temp_cov >= 1):
                        # Valid - now use symmetry tie-breaking
                        word = idx_to_word(lc)
                        canon = get_canonical_form(tuple(word))
                        if best_score < max_s or (best_score == max_s and (best_canon is None or canon < best_canon)):
                            best_score = max_s
                            best_cand = lc
                            best_canon = canon

            if best_cand >= 0:
                coverage_count = temp_cov.copy()
                coverage_count[cov_matrix[:, best_cand] == 1] += 1
                code_indices[i] = best_cand
                code_set.discard(ci)
                code_set.add(best_cand)
                made_swap = True
                break

        if not made_swap:
            break

    # Phase 4: Aggressive local search - try to find better global structure
    # Use simulated annealing-like approach with symmetry bias
    current_size = len(code_indices)
    best_size = current_size
    best_code = list(code_indices)

    for iteration in range(30):
        if time.time() > deadline - 5:
            break
        if len(code_indices) >= best_size:
            # Try to improve
            made_progress = False
            for i in range(len(code_indices)):
                if time.time() > deadline - 5:
                    break
                ci = code_indices[i]
                temp_cov = coverage_count.copy()
                temp_cov[cov_matrix[:, ci] == 1] -= 1
                uncovered_after = temp_cov == 0

                if not np.any(uncovered_after):
                    coverage_count[cov_matrix[:, ci] == 1] -= 1
                    code_indices.pop(i)
                    code_set.discard(ci)
                    made_progress = True
                    break

                if np.sum(uncovered_after) <= 13:
                    unc_list = np.where(uncovered_after)[0]
                    # Find intersection of neighborhoods
                    candidate_set = set([int(unc_list[0])] + [int(x) for x in neighbors[unc_list[0]]])
                    for u in unc_list[1:]:
                        candidate_set &= set([int(u)] + [int(x) for x in neighbors[u]])
                        if not candidate_set:
                            break
                    if candidate_set:
                        for cand in sorted(candidate_set, key=lambda x: get_canonical_form(tuple(idx_to_word(x)))):
                            if cand not in code_set:
                                new_cov = temp_cov.copy()
                                new_cov[cov_matrix[:, cand] == 1] += 1
                                if np.all(new_cov >= 1):
                                    coverage_count = new_cov
                                    code_indices[i] = cand
                                    code_set.discard(ci)
                                    code_set.add(cand)
                                    made_progress = True
                                    break
                        if made_progress:
                            break

            if made_progress:
                if len(code_indices) < best_size:
                    best_size = len(code_indices)
                    best_code = list(code_indices)

    # Restore best code found
    if len(best_code) < len(code_indices):
        code_indices = best_code
        code_set = set(code_indices)
        coverage_count = np.zeros(total, dtype=np.int32)
        for ci in code_indices:
            coverage_count[cov_matrix[:, ci] == 1] += 1

    # Final verification and cleanup
    code = [idx_to_word(i) for i in code_indices]
    return code


def _greedy_set_cover(q, n, radius, deadline):
    """Greedy set cover for general parameters."""
    all_words = list(itertools.product(range(q), repeat=n))
    total = q ** n

    def hamming_dist(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)

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
        code.append(list(all_words[best_i]))
        uncovered -= coverage[best_i]

    return code
# EVOLVE-BLOCK-END
