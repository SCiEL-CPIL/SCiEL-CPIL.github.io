import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    total_words = q ** n
    start_time = time.time()
    deadline = start_time + 118  # safety margin before 150s kill

    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return tuple(reversed(t))

    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute neighbors as numpy arrays for vectorized operations
    cover_count = 1 + n * (q - 1)
    neighbors_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    for i in range(total_words):
        w = idx_to_tuple(i)
        idxs = [i]
        for pos in range(n):
            original_val = w[pos]
            for new_val in range(q):
                if new_val == original_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                j = tuple_to_idx(tuple(new_w))
                idxs.append(j)
        for j, idx in enumerate(idxs):
            neighbors_arr[i, j] = idx

    # Precompute reverse mapping
    reverse_cover_arr = np.zeros((total_words, cover_count), dtype=np.int32)
    reverse_cover_counts = np.zeros(total_words, dtype=np.int32)
    for i in range(total_words):
        for j in range(cover_count):
            w = neighbors_arr[i, j]
            reverse_cover_arr[w, reverse_cover_counts[w]] = i
            reverse_cover_counts[w] += 1

    def greedy_construct_vectorized(random_seed):
        if random_seed is not None:
            random.seed(random_seed)

        uncovered = np.ones(total_words, dtype=np.bool_)
        code = []

        while uncovered.any():
            uncovered_expanded = uncovered[neighbors_arr]
            gains = uncovered_expanded.sum(axis=1)

            valid = gains > 0
            if not valid.any():
                break

            max_gain = gains[valid].max()
            best_candidates = np.where((gains == max_gain) & valid)[0]

            # Smarter tie-breaking: among max-gain candidates, prefer those
            # whose neighbors overlap most with already-covered words
            # (i.e., minimize wasted coverage on already-covered words)
            if len(best_candidates) > 1:
                # Compute how many neighbors of each candidate are already covered
                covered = ~uncovered
                overlap_scores = np.zeros(len(best_candidates), dtype=np.int32)
                for bi, bc in enumerate(best_candidates):
                    overlap_scores[bi] = covered[neighbors_arr[bc, :]].sum()
                # Prefer candidates with higher overlap (less waste)
                max_overlap = overlap_scores.max()
                top_candidates = best_candidates[overlap_scores == max_overlap]
                chosen = random.choice(top_candidates.tolist())
            else:
                chosen = best_candidates[0]

            code.append(chosen)
            uncovered[neighbors_arr[chosen, :]] = False

        return code

    def reduce_code_fast(code):
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1

        changed = True
        while changed:
            changed = False
            new_code = []
            for c in code:
                counts[neighbors_arr[c, :]] -= 1
                if np.any(counts == 0):
                    counts[neighbors_arr[c, :]] += 1
                    new_code.append(c)
                else:
                    changed = True
            code = new_code
        return code

    def local_search_improve(code, deadline):
        """Focused local search: try removing each codeword, and if coverage
        breaks, find the best replacement that minimizes newly-critical words."""
        code = list(code)
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1

        improved = True
        pass_count = 0
        while improved and time.time() < deadline:
            improved = False
            pass_count += 1
            indices = list(range(len(code)))
            random.shuffle(indices)

            for i in indices:
                if time.time() >= deadline:
                    break

                c_remove = code[i]
                counts[neighbors_arr[c_remove, :]] -= 1

                if np.any(counts == 0):
                    counts[neighbors_arr[c_remove, :]] += 1

                    critical_words = neighbors_arr[c_remove, :][counts[neighbors_arr[c_remove, :]] == 1]

                    if len(critical_words) == 0:
                        continue

                    # Find candidates that cover all critical words
                    candidate_set = None
                    for cw in critical_words:
                        rc = reverse_cover_arr[cw, :reverse_cover_counts[cw]]
                        if candidate_set is None:
                            candidate_set = set(rc.tolist())
                        else:
                            candidate_set &= set(rc.tolist())

                    if candidate_set is None or len(candidate_set) == 0:
                        continue

                    candidate_set.discard(c_remove)
                    if len(candidate_set) == 0:
                        continue

                    # Score each candidate: minimize number of words that would
                    # become critical (covered exactly once) after adding it
                    best_w = None
                    best_score = float('inf')
                    cand_list = list(candidate_set)

                    for cand in cand_list:
                        test_counts = counts.copy()
                        test_counts[neighbors_arr[cand, :]] += 1
                        # Count words in cand's neighborhood that are now critical
                        n_critical = np.sum(test_counts[neighbors_arr[cand, :]] == 1)
                        if n_critical < best_score:
                            best_score = n_critical
                            best_w = cand

                    if best_w is None:
                        best_w = random.choice(cand_list)

                    code[i] = best_w
                    counts[neighbors_arr[c_remove, :]] -= 1
                    counts[neighbors_arr[best_w, :]] += 1
                    improved = True
                else:
                    code.pop(i)
                    improved = True

            # Early termination if no improvement in two consecutive passes
            if pass_count > 2 and not improved:
                break

        return code

    def final_polish(code, deadline):
        """Final high-precision polish: try all single-codeword removals and
        2-codeword swaps to find further reductions."""
        code = list(code)
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1

        # First, try single removals in random order
        indices = list(range(len(code)))
        random.shuffle(indices)
        for i in indices:
            if time.time() >= deadline:
                break
            c_remove = code[i]
            counts[neighbors_arr[c_remove, :]] -= 1
            if not np.any(counts == 0):
                code.pop(i)
            else:
                counts[neighbors_arr[c_remove, :]] += 1

        if time.time() >= deadline:
            return code

        # Then try 1-for-1 swaps with scoring
        counts = np.zeros(total_words, dtype=np.int16)
        for c in code:
            counts[neighbors_arr[c, :]] += 1

        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = list(range(len(code)))
            random.shuffle(indices)

            for i in indices:
                if time.time() >= deadline:
                    break

                c_remove = code[i]
                counts[neighbors_arr[c_remove, :]] -= 1

                if np.any(counts == 0):
                    counts[neighbors_arr[c_remove, :]] += 1

                    critical_words = neighbors_arr[c_remove, :][counts[neighbors_arr[c_remove, :]] == 1]
                    if len(critical_words) == 0:
                        continue

                    candidate_set = None
                    for cw in critical_words:
                        rc = reverse_cover_arr[cw, :reverse_cover_counts[cw]]
                        if candidate_set is None:
                            candidate_set = set(rc.tolist())
                        else:
                            candidate_set &= set(rc.tolist())

                    if candidate_set is None or len(candidate_set) == 0:
                        continue

                    candidate_set.discard(c_remove)
                    if len(candidate_set) == 0:
                        continue

                    # Score: minimize newly-critical words
                    best_w = None
                    best_score = float('inf')
                    for cand in list(candidate_set):
                        test_counts = counts.copy()
                        test_counts[neighbors_arr[cand, :]] += 1
                        n_critical = np.sum(test_counts[neighbors_arr[cand, :]] == 1)
                        if n_critical < best_score:
                            best_score = n_critical
                            best_w = cand

                    if best_w is not None:
                        code[i] = best_w
                        counts[neighbors_arr[c_remove, :]] -= 1
                        counts[neighbors_arr[best_w, :]] += 1
                        improved = True
                else:
                    code.pop(i)
                    improved = True

        return code

    best_code = None
    best_size = float('inf')

    seed = 0
    while time.time() < deadline - 8:
        # Greedy construction with smarter tie-breaking
        code = greedy_construct_vectorized(seed)

        # Reduction pass
        code = reduce_code_fast(code)

        # Local search improvement
        if time.time() < deadline - 20:
            code = local_search_improve(code, deadline - 15)
            code = reduce_code_fast(code)

        # Final polish
        if time.time() < deadline - 5:
            code = final_polish(code, deadline - 3)
            code = reduce_code_fast(code)

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

        seed += 1

    result = []
    for idx in best_code:
        w = idx_to_tuple(idx)
        result.append(list(w))

    return result
