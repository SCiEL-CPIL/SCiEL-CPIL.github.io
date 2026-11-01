import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0

    # For the specific task q=3, n=6, radius=1
    # Total words = 3^6 = 729
    # Each word covers itself + 1*n*(q-1) = 1*6*2 = 12 neighbors = 13 words total

    total_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))

    # Precompute coverage masks as Python ints (bitsets)
    # For q=3, n=6: each mask has 729 bits
    coverage_masks = [0] * total_words
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    for i, w in enumerate(all_words):
        mask = 1 << i
        if radius == 1:
            for pos in range(n):
                orig = w[pos]
                for nv in range(q):
                    if nv != orig:
                        lst = list(w)
                        lst[pos] = nv
                        j = word_to_idx[tuple(lst)]
                        mask |= (1 << j)
        coverage_masks[i] = mask

    all_ones = (1 << total_words) - 1

    # ---- Algorithm: Greedy + aggressive local search with 1-swap and 1-removal ----

    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    def greedy_cover(seed_random):
        """Greedy set cover with random tie-breaking."""
        uncovered = all_ones
        chosen = []
        chosen_set = set()
        indices = list(range(total_words))
        # Shuffle to break symmetry
        rng = random.Random(seed_random)
        rng.shuffle(indices)

        while uncovered:
            best_count = -1
            best_candidates = []
            for i in indices:
                if i in chosen_set:
                    continue
                common = coverage_masks[i] & uncovered
                cnt = popcount(common)
                if cnt > best_count:
                    best_count = cnt
                    best_candidates = [i]
                elif cnt == best_count:
                    best_candidates.append(i)
            if not best_candidates:
                break
            pick = rng.choice(best_candidates)
            chosen.append(pick)
            chosen_set.add(pick)
            uncovered &= ~coverage_masks[pick]

        return chosen, chosen_set

    def local_search(chosen, chosen_set, deadline):
        """Iterative removal and 1-swap local search."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            k = len(chosen)
            if k == 0:
                break

            # Build list of masks
            masks = [coverage_masks[c] for c in chosen]

            # Total coverage
            total_cov = 0
            for m in masks:
                total_cov |= m

            if total_cov != all_ones:
                # Shouldn't happen, but bail
                break

            # Try removal: for each i, check if OR of all except i covers everything
            # Use prefix/suffix OR for O(k) instead of O(k^2)
            prefix_or = [0] * (k + 1)
            for i in range(k):
                prefix_or[i + 1] = prefix_or[i] | masks[i]
            suffix_or = [0] * (k + 1)
            for i in range(k - 1, -1, -1):
                suffix_or[i] = suffix_or[i + 1] | masks[i]

            # Removal check
            removed_idx = -1
            for i in range(k):
                without_i = prefix_or[i] | suffix_or[i + 1]
                if without_i == all_ones:
                    removed_idx = i
                    break

            if removed_idx >= 0:
                c = chosen.pop(removed_idx)
                chosen_set.discard(c)
                masks.pop(removed_idx)
                improved = True
                continue

            # Try 1-swap: remove c_i, add c' not in chosen
            # For each i, missing = all_ones & ~(OR without i)
            # Need c' such that coverage_masks[c'] covers missing
            # Precompute: for each candidate c', which missing sets it can cover
            # Since k is small (~73), iterate over all c' for each i

            best_swap = None
            for i in range(k):
                without_i = prefix_or[i] | suffix_or[i + 1]
                missing = all_ones & ~without_i
                if missing == 0:
                    # Should have been caught by removal
                    continue
                # Find c' not in chosen that covers missing
                # To speed up, iterate over all words
                for cp in range(total_words):
                    if cp in chosen_set:
                        continue
                    cm = coverage_masks[cp]
                    if (missing & ~cm) == 0:
                        best_swap = (i, cp)
                        break
                if best_swap:
                    break

            if best_swap:
                i, cp = best_swap
                old = chosen[i]
                chosen[i] = cp
                chosen_set.discard(old)
                chosen_set.add(cp)
                improved = True

        return chosen, chosen_set

    best_chosen = None
    best_size = total_words + 1

    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
        seed = 42 + r * 7
        chosen, chosen_set = greedy_cover(seed)
        if time.time() > deadline - 5:
            if len(chosen) < best_size:
                best_size = len(chosen)
                best_chosen = list(chosen)
            continue
        chosen, chosen_set = local_search(chosen, chosen_set, deadline - 2)
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_chosen = list(chosen)

    if best_chosen is None:
        # Fallback
        chosen, _ = greedy_cover(0)
        best_chosen = chosen

    return [list(all_words[i]) for i in best_chosen]
