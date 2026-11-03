import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit

    # 1. Precomputation
    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}

    blocks = list(itertools.combinations(points, k))
    n_blocks = len(blocks)

    # C(k, t)
    c_k_t = 1
    for i in range(1, t + 1):
        c_k_t = c_k_t * (k - i + 1) // i

    # block_triples[i] contains the indices of the t-subsets covered by blocks[i]
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    triple_freq = np.zeros(n_triples, dtype=np.int32)

    for i, block in enumerate(blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            t_idx = triple_to_idx[triple]
            block_triples[i, idx] = t_idx
            triple_freq[t_idx] += 1
            idx += 1

    # Weights: 1 / frequency. Higher weight for rarer triples.
    weights = 1.0 / np.maximum(triple_freq, 1)

    best_solution = None
    best_size = float('inf')

    # Build a set of selected block indices for fast membership
    def build_cover_count(selected, block_triples, n_triples):
        cover_count = np.zeros(n_triples, dtype=np.int32)
        for b_idx in selected:
            cover_count[block_triples[b_idx]] += 1
        return cover_count

    def exact_polish(selected, block_triples, n_triples, c_k_t, deadline):
        """Iteratively remove redundant blocks and try single-block replacements."""
        if len(selected) <= 1:
            return selected

        current = list(selected)
        improved = True
        max_rounds = 200

        while improved and max_rounds > 0 and time.time() < deadline - 2.0:
            improved = False
            max_rounds -= 1

            cover_count = build_cover_count(current, block_triples, n_triples)

            # Pass 1: remove any block whose triples are all covered by others
            for i in range(len(current)):
                b_idx = current[i]
                b_triples = block_triples[b_idx]
                if np.min(cover_count[b_triples]) > 1:
                    cover_count[b_triples] -= 1
                    current.pop(i)
                    improved = True
                    break

            if improved:
                continue

            # Pass 2: try replacing one block with another
            # For each block i, find triples it uniquely covers
            # Then find a replacement block j that covers all those unique triples
            # and doesn't lose any other triple
            n_sel = len(current)
            for i in range(n_sel):
                b_idx = current[i]
                b_triples = block_triples[b_idx]

                # Unique triples: those with cover_count == 1
                unique_mask = cover_count == 1
                unique_in_block = unique_mask[b_triples]
                if not np.any(unique_in_block):
                    continue  # No unique triples; would have been removed in pass 1

                # We need a replacement that covers all unique_in_block triples
                # and for all other triples in b_triples, they remain covered
                # (i.e., cover_count >= 2 for non-unique, or the replacement covers them)

                # Compute which triples would be lost if we remove b_idx without replacement
                # A triple is lost if cover_count == 1 and it's in b_triples
                # So the replacement must cover all unique triples in b_triples

                # For each candidate block j (not in current), check if it covers all unique triples
                # and doesn't cause any other triple to drop below 1

                # Precompute: for each triple, how many times is it covered by current blocks
                # If we remove b_idx, cover_count drops by 1 for all b_triples
                # Then if we add j, cover_count rises by 1 for all j_triples
                # Net effect: for triples in b_triples not in j_triples, count drops by 1
                # For triples in j_triples not in b_triples, count rises by 1
                # For triples in both, no change

                # So we need: for all triples in b_triples, cover_count_after >= 1
                # cover_count_after = cover_count - 1 + (1 if in j_triples else 0)
                # So we need: cover_count - 1 + (1 if in j_triples) >= 1
                # i.e., cover_count >= 1 - (1 if in j_triples)
                # For triples NOT in j_triples: cover_count >= 1 (already true)
                # For triples IN j_triples: cover_count >= 0 (always true)
                # Wait, that's not right. Let me re-think.

                # Original: cover_count[t] >= 1 for all t (valid covering)
                # After removing b_idx: cover_count[t] - 1 for t in b_triples
                # After adding j: cover_count[t] - 1 + 1 for t in b_triples ∩ j_triples
                #                cover_count[t] - 1 for t in b_triples \ j_triples
                #                cover_count[t] + 1 for t in j_triples \ b_triples

                # We need all >= 1:
                # For t in b_triples \ j_triples: cover_count[t] - 1 >= 1 => cover_count[t] >= 2
                # For t in b_triples ∩ j_triples: cover_count[t] >= 1 (already true)
                # For t in j_triples \ b_triples: cover_count[t] + 1 >= 1 (always true)

                # So the condition is: for all t in b_triples \ j_triples, cover_count[t] >= 2
                # Equivalently: for all t with cover_count[t] == 1, t must be in j_triples

                # So j must cover all triples with cover_count == 1 that are in b_triples
                # (which is the same as the unique triples in b_triples)

                # Let's find all candidate blocks that cover all unique triples in b_idx
                # unique_triples_in_block = b_triples[unique_in_block]
                unique_triples = b_triples[unique_in_block]

                # For each candidate block j, check if it covers all unique_triples
                # We can precompute a lookup: for each triple, which blocks cover it
                # But that's expensive. Instead, we can check by seeing if the intersection
                # of j_triples with unique_triples has the same size as unique_triples

                # This is O(n_blocks * c_k_t) per i, which is too slow
                # Let's limit the number of candidates we check

                # Alternative: just try a few random candidates
                # Pick up to 50 random blocks not in current
                candidates = []
                for _ in range(min(50, n_blocks - n_sel)):
                    j = np.random.randint(0, n_blocks)
                    if j not in current:
                        candidates.append(j)

                for j in candidates:
                    j_triples = block_triples[j]
                    # Check if j covers all unique_triples
                    # Use set intersection
                    j_set = set(j_triples.tolist())
                    if all(t in j_set for t in unique_triples.tolist()):
                        # Also check that no triple drops below 1
                        # For t in b_triples \ j_triples, need cover_count[t] >= 2
                        b_set = set(b_triples.tolist())
                        # Triples in b_triples but not in j_triples
                        lost_triples = b_set - j_set
                        if lost_triples:
                            lost_arr = np.array(list(lost_triples), dtype=np.int32)
                            if np.min(cover_count[lost_arr]) < 2:
                                continue
                        # Replacement is valid
                        current[i] = j
                        improved = True
                        break

                if improved:
                    break

        return current

    # Weighted greedy with randomization
    def weighted_greedy_cover(weights, block_triples, n_triples, temperature=0.0):
        uncovered = np.ones(n_triples, dtype=bool)
        selected = []

        while np.any(uncovered):
            if time.time() > deadline:
                break

            weighted_uncovered = np.where(uncovered, weights, 0.0)
            gains = np.sum(weighted_uncovered[block_triples], axis=1)

            if temperature > 0:
                max_gain = np.max(gains)
                if max_gain > 0:
                    exp_gains = np.exp((gains - max_gain) / temperature)
                    probs = exp_gains / np.sum(exp_gains)
                    best_idx = np.random.choice(n_blocks, p=probs)
                else:
                    best_idx = np.argmax(gains)
            else:
                best_idx = np.argmax(gains)

            if gains[best_idx] == 0:
                break

            selected.append(best_idx)
            uncovered[block_triples[best_idx]] = False

        return selected

    # Run multiple greedy passes with different weight perturbations
    n_restarts = 0
    while time.time() < deadline - 5.0:
        n_restarts += 1

        # Perturb weights for diversity
        if n_restarts == 1:
            perturbed_weights = weights.copy()
            temperature = 0.0
        else:
            noise = np.random.normal(0, 0.15, size=n_triples)
            perturbed_weights = weights * (1 + noise)
            perturbed_weights = np.maximum(perturbed_weights, 0.01)
            temperature = 0.1 * n_restarts  # Increasing temperature for diversity

        # Run greedy
        selected = weighted_greedy_cover(perturbed_weights, block_triples, n_triples, temperature)

        if not selected:
            continue

        # Polish with exact method
        selected = exact_polish(selected, block_triples, n_triples, c_k_t, deadline)

        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = [list(blocks[i]) for i in selected]

        # Early exit if we hit the lower bound (69)
        if best_size <= 69:
            break

    if best_solution is None:
        # Fallback: return a valid covering
        selected = weighted_greedy_cover(weights, block_triples, n_triples)
        selected = exact_polish(selected, block_triples, n_triples, c_k_t, deadline)
        best_solution = [list(blocks[i]) for i in selected]

    return best_solution
