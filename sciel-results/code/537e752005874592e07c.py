import numpy as np
import itertools
import time
import random
import sys

def construct(q, n, radius):
    # Fallback for non-specific parameters
    if q != 3 or n != 8 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            if n == 4:
                return tetracode
            else:
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Specific optimization for K_3(8,1)
    total_points = q ** n  # 6561
    neighbors_per_word = 1 + n * (q - 1)  # 17

    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

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

    # Precompute neighbors for all 6561 words
    neighbors = np.zeros((total_points, neighbors_per_word), dtype=np.int32)
    for i in range(total_points):
        w = idx_to_word(i)
        neighbors[i, 0] = i
        count = 1
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                neighbors[i, count] = word_to_idx(w)
                count += 1
            w[pos] = orig_val

    all_words = [idx_to_word(i) for i in range(total_points)]

    # Build a set for each codeword's neighbors for fast lookup
    neighbor_sets = [set(neighbors[i].tolist()) for i in range(total_points)]

    def compute_coverage(chosen_indices):
        """Compute which points are covered and how many times."""
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in chosen_indices:
            cover_count[neighbors[ci]] += 1
        return cover_count

    def is_valid_cover(chosen_indices):
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in chosen_indices:
            cover_count[neighbors[ci]] += 1
        return np.all(cover_count > 0)

    def structured_initial_code():
        """
        Construct the lexicographic tetracode product: 81 codewords.
        Tetracode: {(a, b, a+b, a+2b) mod 3 : a,b in {0,1,2}}
        Product: first 4 coords from one tetracode word, last 4 from another.
        """
        tetracode = [(a, b, (a + b) % 3, (a + 2 * b) % 3) for a in range(3) for b in range(3)]
        indices = []
        for t1 in tetracode:
            for t2 in tetracode:
                word = list(t1) + list(t2)
                idx = word_to_idx(word)
                indices.append(idx)
        return indices

    def criticality_greedy_complete(chosen_indices):
        """
        Greedily add codewords prioritizing words that cover the most uncovered points.
        Uses criticality: points covered by fewest codewords are most critical.
        """
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in chosen_indices:
            cover_count[neighbors[ci]] += 1

        chosen = list(chosen_indices)
        available = np.ones(total_points, dtype=bool)
        for ci in chosen:
            available[ci] = False

        while not np.all(cover_count > 0):
            if time.time() - start_time > time_limit - 20.0:
                break

            uncovered = np.where(cover_count == 0)[0]
            if len(uncovered) == 0:
                break

            # For each available word, count how many uncovered points it covers
            avail_indices = np.where(available)[0]
            if len(avail_indices) == 0:
                break

            # Vectorized: for each available word, count newly covered points
            newly_covered = np.zeros(len(avail_indices), dtype=np.int32)
            for j, ai in enumerate(avail_indices):
                newly_covered[j] = np.sum(cover_count[neighbors[ai]] == 0)

            max_new = np.max(newly_covered)
            if max_new == 0:
                break

            # Among those with max_new, pick the one whose neighbors have lowest average coverage
            # (criticality: prefer covering points that are barely covered)
            best_candidates = np.where(newly_covered == max_new)[0]
            if len(best_candidates) == 1:
                pick = best_candidates[0]
            else:
                # Criticality tiebreaker: prefer words covering points with lower coverage
                min_cov_scores = []
                for bc in best_candidates:
                    ai = avail_indices[bc]
                    # Average coverage of the neighbors (lower is more critical)
                    avg_cov = np.mean(cover_count[neighbors[ai]])
                    min_cov_scores.append(avg_cov)
                min_cov_scores = np.array(min_cov_scores)
                pick = best_candidates[np.argmin(min_cov_scores)]

            best_global = avail_indices[pick]
            chosen.append(best_global)
            cover_count[neighbors[best_global]] += 1
            available[best_global] = False

        return chosen

    def strict_two_opt(chosen_indices):
        """
        Strict 2-opt local search:
        - Try removing each codeword; if all points remain covered, remove it.
        - Try swapping codeword i with codeword j: only accept if it reduces size
          (not possible in swap) or increases minimum coverage multiplicity.
        - Also try: remove codeword i, then add the single best replacement that
          covers all exposed points and maximizes new coverage.
        """
        current = list(chosen_indices)
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in current:
            cover_count[neighbors[ci]] += 1

        improved = True
        max_rounds = 50
        round_num = 0

        while improved and round_num < max_rounds and time.time() - start_time < time_limit - 10.0:
            improved = False
            round_num += 1

            # Phase 1: Try removing each codeword
            i = 0
            while i < len(current):
                if time.time() - start_time > time_limit - 10.0:
                    break

                ci = current[i]
                exposed = neighbors[ci]
                would_uncover = exposed[cover_count[exposed] == 1]

                if len(would_uncover) == 0:
                    # Safe to remove
                    cover_count[neighbors[ci]] -= 1
                    current.pop(i)
                    improved = True
                    continue

                # Try to find a single replacement that covers all exposed points
                # and adds maximum new coverage
                best_replacement = -1
                best_gain = -1

                # Find candidate words that cover all exposed points
                # Use the set of exposed points to filter
                exposed_set = set(would_uncover.tolist())

                # Sample candidates from the neighborhood of exposed points
                # For each exposed point, its neighbors are potential replacements
                candidate_set = set()
                for ep in would_uncover:
                    candidate_set.update(neighbors[ep].tolist())
                candidate_set.discard(ci)

                # Limit candidates for speed
                candidates = list(candidate_set)
                if len(candidates) > 200:
                    random.shuffle(candidates)
                    candidates = candidates[:200]

                for cand in candidates:
                    if cand in current:
                        continue
                    # Check if cand covers all exposed points
                    cand_neigh = neighbors[cand]
                    # Check coverage of exposed points
                    if not all(p in neighbor_sets[cand] for p in exposed_set):
                        continue

                    # Count new coverage
                    new_cov = np.sum(cover_count[cand_neigh] == 0)
                    if new_cov > best_gain:
                        best_gain = new_cov
                        best_replacement = cand

                if best_replacement >= 0:
                    # Replace ci with best_replacement
                    cover_count[neighbors[ci]] -= 1
                    cover_count[neighbors[best_replacement]] += 1
                    current[i] = best_replacement
                    improved = True

                i += 1

            # Phase 2: Try pairwise swaps to increase minimum coverage
            if time.time() - start_time > time_limit - 10.0:
                break

            min_cov = np.min(cover_count)
            n_current = len(current)

            for i in range(n_current):
                if time.time() - start_time > time_limit - 10.0:
                    break
                for j in range(i + 1, n_current):
                    if time.time() - start_time > time_limit - 10.0:
                        break

                    ci = current[i]
                    cj = current[j]

                    # Try swapping ci and cj: remove both, see if still covered
                    cover_count[neighbors[ci]] -= 1
                    cover_count[neighbors[cj]] -= 1

                    if np.all(cover_count > 0):
                        # Both can be removed! This is a 2-removal
                        current.pop(j)
                        current.pop(i)
                        improved = True
                        break

                    # Restore
                    cover_count[neighbors[ci]] += 1
                    cover_count[neighbors[cj]] += 1

                if improved:
                    break

            if improved:
                continue

            # Phase 3: Try replacing one codeword with two others that together
            # cover more, allowing eventual removal
            # This is expensive, so only do it if we have time
            if time.time() - start_time > time_limit - 30.0:
                break

            for i in range(len(current)):
                if time.time() - start_time > time_limit - 30.0:
                    break

                ci = current[i]
                exposed = neighbors[ci]
                would_uncover = exposed[cover_count[exposed] == 1]

                if len(would_uncover) > 3:
                    continue  # Too many points to re-cover with 2 words

                # Try to find two words that together cover all exposed points
                # and add net new coverage
                exposed_set = set(would_uncover.tolist())

                # Get candidates from neighbors of exposed points
                candidate_set = set()
                for ep in would_uncover:
                    candidate_set.update(neighbors[ep].tolist())
                candidate_set.discard(ci)
                candidates = list(candidate_set)

                if len(candidates) > 50:
                    random.shuffle(candidates)
                    candidates = candidates[:50]

                best_pair = None
                best_pair_gain = -1

                for a_idx in range(len(candidates)):
                    for b_idx in range(a_idx + 1, len(candidates)):
                        ca = candidates[a_idx]
                        cb = candidates[b_idx]
                        if ca in current or cb in current:
                            continue

                        # Check if ca and cb together cover all exposed points
                        union_cov = neighbor_sets[ca] | neighbor_sets[cb]
                        if not exposed_set.issubset(union_cov):
                            continue

                        # Count new coverage from both
                        new_cov = 0
                        for p in neighbors[ca]:
                            if cover_count[p] == 0:
                                new_cov += 1
                        for p in neighbors[cb]:
                            if cover_count[p] == 0:
                                new_cov += 1

                        if new_cov > best_pair_gain:
                            best_pair_gain = new_cov
                            best_pair = (ca, cb)

                if best_pair is not None:
                    ca, cb = best_pair
                    # Remove ci, add ca and cb
                    cover_count[neighbors[ci]] -= 1
                    cover_count[neighbors[ca]] += 1
                    cover_count[neighbors[cb]] += 1
                    current[i] = ca
                    current.append(cb)
                    improved = True
                    break

        return current

    # Main algorithm
    # 1. Start with structured tetracode product (81 words)
    initial_code = structured_initial_code()

    # 2. Complete with criticality-driven greedy
    completed_code = criticality_greedy_complete(initial_code)

    # Verify validity
    if not is_valid_cover(completed_code):
        # Emergency: add more words
        cover_count = np.zeros(total_points, dtype=np.int32)
        for ci in completed_code:
            cover_count[neighbors[ci]] += 1
        while not np.all(cover_count > 0):
            if time.time() - start_time > time_limit - 20.0:
                break
            uncovered = np.where(cover_count == 0)[0]
            if len(uncovered) == 0:
                break
            # Greedy: pick word covering most uncovered
            best_idx = -1
            best_count = -1
            for idx in range(total_points):
                cnt = np.sum(cover_count[neighbors[idx]] == 0)
                if cnt > best_count:
                    best_count = cnt
                    best_idx = idx
            if best_idx < 0:
                break
            completed_code.append(best_idx)
            cover_count[neighbors[best_idx]] += 1

    # 3. Apply strict 2-opt local search
    optimized_code = strict_two_opt(completed_code)

    # Verify final validity
    if not is_valid_cover(optimized_code):
        # Fallback to the completed code
        optimized_code = completed_code

    # Final safety check
    if not is_valid_cover(optimized_code):
        # Emergency full greedy
        cover_count = np.zeros(total_points, dtype=np.int32)
        chosen = []
        while not np.all(cover_count > 0):
            if time.time() - start_time > time_limit - 5.0:
                break
            best_idx = -1
            best_count = -1
            for idx in range(total_points):
                cnt = np.sum(cover_count[neighbors[idx]] == 0)
                if cnt > best_count:
                    best_count = cnt
                    best_idx = idx
            if best_idx < 0:
                break
            chosen.append(best_idx)
            cover_count[neighbors[best_idx]] += 1
        optimized_code = chosen

    return [all_words[i] for i in optimized_code]
