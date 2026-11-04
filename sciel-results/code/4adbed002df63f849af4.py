"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def get_direct_sum():
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        return [[0] * n]

    # For the specific case q=3, n=6, radius=1, use optimized greedy set cover
    if q == 3 and n == 6 and radius == 1:
        total = 3 ** 6
        powers = [3 ** p for p in range(n)]

        # Precompute for each word, the 13 words within Hamming distance 1
        neighbors = np.empty((total, 13), dtype=np.int32)
        for i in range(total):
            d = [(i // powers[p]) % 3 for p in range(n)]
            neighbors[i, 0] = i
            k = 1
            for p in range(n):
                for alt in range(3):
                    if alt != d[p]:
                        new_i = i + (alt - d[p]) * powers[p]
                        neighbors[i, k] = new_i
                        k += 1

        def greedy_cover():
            """Standard greedy: pick candidate covering most uncovered words."""
            uncovered = np.ones(total, dtype=np.int8)
            codewords = []

            while uncovered.sum() > 0:
                if time.time() > deadline:
                    break
                counts = np.sum(uncovered[neighbors], axis=1)
                best_candidate = np.argmax(counts)
                if counts[best_candidate] == 0:
                    break
                codewords.append(best_candidate)
                uncovered[neighbors[best_candidate]] = 0

            return codewords

        def local_search_remove(codewords):
            """Try to remove redundant codewords."""
            improved = True
            while improved:
                improved = False
                coverage_counts = np.zeros(total, dtype=np.int32)
                for idx_c in codewords:
                    coverage_counts[neighbors[idx_c]] += 1

                for i in range(len(codewords)):
                    idx_c = codewords[i]
                    if np.min(coverage_counts[neighbors[idx_c]]) >= 2:
                        codewords.pop(i)
                        improved = True
                        break

            return codewords

        def local_search_swap(codewords, max_swaps=500):
            """Try to swap codewords to reduce count or enable removals."""
            improved = True
            swap_count = 0
            while improved and swap_count < max_swaps:
                improved = False
                swap_count += 1

                coverage_counts = np.zeros(total, dtype=np.int32)
                for idx_c in codewords:
                    coverage_counts[neighbors[idx_c]] += 1

                # Find a codeword that, if removed, leaves some words with coverage 1
                # Then try to add a different codeword that covers those words
                for i in range(len(codewords)):
                    idx_c = codewords[i]
                    # If we remove idx_c, which words drop to coverage 1?
                    after_removal = coverage_counts - np.zeros(total, dtype=np.int32)
                    after_removal[neighbors[idx_c]] -= 1
                    critical = np.where((after_removal == 1) & (coverage_counts[neighbors[idx_c]] >= 1))[0]

                    if len(critical) == 0:
                        # Can just remove
                        codewords.pop(i)
                        improved = True
                        break

                    # Try to find a replacement that covers all critical words
                    # and doesn't increase total count (we swap, not add)
                    # For swap: remove idx_c, add new_idx. Net change in coverage:
                    # new_idx neighbors +1, idx_c neighbors -1
                    # We need all critical words still covered (they lose idx_c coverage)
                    # So new_idx must cover all critical words.
                    # Also, words that were only covered by idx_c (not by new_idx) would become uncovered.
                    # Actually since we're swapping (same count), we need:
                    # - All words in neighbors[idx_c] \ neighbors[new_idx] must have coverage >= 2
                    #   (they lose idx_c but keep others)
                    # - All words in neighbors[new_idx] \ neighbors[idx_c] are fine (gain coverage)
                    # - Words in intersection: lose idx_c, gain new_idx, net 0

                    # Words that would become uncovered if we remove idx_c and add new_idx:
                    # A word w is uncovered iff coverage_counts[w] - 1_{w in N(idx_c)} + 1_{w in N(new_idx)} == 0
                    # i.e., coverage_counts[w] == 1_{w in N(idx_c)} - 1_{w in N(new_idx)}
                    # If w in N(idx_c) and not in N(new_idx): need coverage_counts[w] >= 2
                    # If w not in N(idx_c) and in N(new_idx): coverage_counts[w] + 1 >= 1, always fine
                    # If w in both: coverage_counts[w] >= 1, always fine
                    # If w in neither: coverage_counts[w] >= 0, fine

                    # So we need: for all w in N(idx_c) \ N(new_idx), coverage_counts[w] >= 2

                    # Precompute: for each candidate new_idx, check if it's a valid swap
                    # This is expensive; let's limit candidates
                    # Candidates: words that cover at least one critical word
                    candidates = set()
                    for w in critical[:50]:  # Limit for speed
                        # Find which codeword-neighbor positions cover w
                        # Actually, find all new_idx such that w is in neighbors[new_idx]
                        # This is the inverse neighbor relation
                        # For efficiency, just try all 729 candidates but check quickly
                        pass

                    # Try a subset of candidates: those adjacent to critical words
                    # For each critical word w, candidates are words within distance 1 of w
                    # But we need new_idx such that all critical words are in N(new_idx)
                    # That means new_idx must be within distance 1 of ALL critical words
                    # If critical words are far apart, no such new_idx exists

                    # Quick check: if any two critical words are at distance > 2, no swap works
                    if len(critical) > 1:
                        # Check pairwise distances
                        feasible = True
                        for a in range(min(len(critical), 10)):
                            for b in range(a+1, min(len(critical), 10)):
                                wa = critical[a]
                                wb = critical[b]
                                # Hamming distance
                                da = [(wa // powers[p]) % 3 for p in range(n)]
                                db = [(wb // powers[p]) % 3 for p in range(n)]
                                dist = sum(1 for x, y in zip(da, db) if x != y)
                                if dist > 2:
                                    feasible = False
                                    break
                            if not feasible:
                                break
                        if not feasible:
                            continue

                    # Try all 729 candidates as new_idx
                    found_swap = False
                    for new_idx in range(total):
                        if new_idx in codewords:
                            continue
                        # Check: all words in N(idx_c) \ N(new_idx) have coverage >= 2
                        # N(idx_c) is neighbors[idx_c], N(new_idx) is neighbors[new_idx]
                        # A word w in N(idx_c) not in N(new_idx):
                        #   coverage_counts[w] - 1 >= 1, i.e., coverage_counts[w] >= 2
                        # Words in N(idx_c) that are also in N(new_idx): fine
                        # Words not in N(idx_c): fine

                        # Efficient check:
                        # For w in neighbors[idx_c]:
                        #   if w not in neighbors[new_idx]: need coverage_counts[w] >= 2
                        #   if w in neighbors[new_idx]: fine (net 0 change)
                        # For w not in neighbors[idx_c]: fine (only gains or unchanged)

                        # Build set of neighbors[new_idx] for quick lookup
                        # Actually, use a boolean array
                        in_new = np.zeros(total, dtype=np.int8)
                        in_new[neighbors[new_idx]] = 1

                        valid = True
                        for w in neighbors[idx_c]:
                            if in_new[w] == 0:
                                if coverage_counts[w] < 2:
                                    valid = False
                                    break

                        if valid:
                            # Perform swap
                            codewords[i] = new_idx
                            improved = True
                            found_swap = True
                            break

                    if found_swap:
                        break

            return codewords

        def greedy_with_tie_break(codewords_seed=None):
            """Greedy with different tie-breaking strategies."""
            uncovered = np.ones(total, dtype=np.int8)
            codewords = []

            if codewords_seed:
                # Start with seed codewords
                for idx_c in codewords_seed:
                    uncovered[neighbors[idx_c]] = 0
                    codewords.append(idx_c)

            while uncovered.sum() > 0:
                if time.time() > deadline:
                    break
                counts = np.sum(uncovered[neighbors], axis=1)
                # Among candidates with max count, pick one with some heuristic
                max_count = np.max(counts)
                candidates = np.where(counts == max_count)[0]
                if len(candidates) == 0:
                    break

                # Tie-break: prefer candidates that are "central" or have other properties
                # For now, just pick the first (deterministic)
                best_candidate = candidates[0]

                codewords.append(best_candidate)
                uncovered[neighbors[best_candidate]] = 0

            return codewords

        # Strategy 1: Standard greedy + local search
        best_codewords = None
        best_len = float('inf')

        # Run standard greedy
        cw = greedy_cover()
        if uncovered.sum() == 0 if 'uncovered' in dir() else True:
            # Verify
            check_cov = np.zeros(total, dtype=np.int8)
            for idx_c in cw:
                check_cov[neighbors[idx_c]] = 1
            if check_cov.sum() == total:
                cw = local_search_remove(cw)
                cw = local_search_swap(cw)
                if len(cw) < best_len:
                    best_len = len(cw)
                    best_codewords = cw[:]

        # Strategy 2: Greedy with different starting points
        # Try starting with different initial codewords
        initial_seeds = [
            [0],  # all zeros
            [total - 1],  # all twos
            [364],  # middle
        ]

        for seed in initial_seeds:
            if time.time() > deadline - 10:
                break
            cw = greedy_with_tie_break(seed)
            # Verify
            check_cov = np.zeros(total, dtype=np.int8)
            for idx_c in cw:
                check_cov[neighbors[idx_c]] = 1
            if check_cov.sum() == total:
                cw = local_search_remove(cw)
                cw = local_search_swap(cw)
                if len(cw) < best_len:
                    best_len = len(cw)
                    best_codewords = cw[:]

        # Strategy 3: Multiple full greedy restarts with perturbation
        import random
        rng = random.Random(42)

        for restart in range(10):
            if time.time() > deadline - 5:
                break

            # Perturbed greedy: at each step, with small probability pick second-best
            uncovered = np.ones(total, dtype=np.int8)
            codewords = []

            while uncovered.sum() > 0:
                if time.time() > deadline - 5:
                    break
                counts = np.sum(uncovered[neighbors], axis=1)
                max_count = np.max(counts)
                candidates = np.where(counts == max_count)[0]
                if len(candidates) == 0:
                    break

                # With 10% probability, pick a random candidate from top-5
                if rng.random() < 0.1 and len(candidates) > 1:
                    # Get top 5 by count
                    top5 = np.argsort(counts)[-5:]
                    best_candidate = rng.choice(top5.tolist())
                else:
                    best_candidate = candidates[0]

                codewords.append(best_candidate)
                uncovered[neighbors[best_candidate]] = 0

            # Verify
            check_cov = np.zeros(total, dtype=np.int8)
            for idx_c in codewords:
                check_cov[neighbors[idx_c]] = 1
            if check_cov.sum() == total:
                codewords = local_search_remove(codewords)
                codewords = local_search_swap(codewords)
                if len(codewords) < best_len:
                    best_len = len(codewords)
                    best_codewords = codewords[:]

        # Use best result if valid, else direct sum
        if best_codewords is not None and best_len <= len(get_direct_sum()):
            final_codewords = []
            for idx_c in best_codewords:
                final_codewords.append([(idx_c // powers[p]) % 3 for p in range(n)])
            return final_codewords
        else:
            return get_direct_sum()

    # General case: fallback to direct sum
    return get_direct_sum()
# EVOLVE-BLOCK-END
