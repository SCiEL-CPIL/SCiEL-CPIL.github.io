"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill

    space_size = q ** n
    # Map word to index
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    def idx_to_word(idx, q, n):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute all neighbors (words within radius 1) for every word in the space
    neighbors = []
    for idx in range(space_size):
        w = idx_to_word(idx, q, n)
        nbrs = [idx]
        if radius >= 1:
            for pos in range(n):
                orig = w[pos]
                for v in range(q):
                    if v == orig:
                        continue
                    new_w = w.copy()
                    new_w[pos] = v
                    nbrs.append(word_to_idx(new_w))
        neighbors.append(nbrs)

    # Precompute conflict graph: two codewords conflict if their distance <= 2
    # (i.e., they are within radius 1 of each other's neighbors, meaning their balls overlap)
    # Actually, for a covering code, we just need every word to be within radius 1 of some codeword.
    # Two codewords can be close to each other; that's fine. The constraint is just coverage.
    # But to find a small set, we want to spread codewords out.
    # Let's use greedy set cover with a twist: prefer codewords that are far from existing ones.

    # Greedy set cover
    uncovered = np.ones(space_size, dtype=bool)
    selected_indices = []

    # For speed, precompute coverage as numpy arrays
    # cover[i] is a boolean array of length space_size
    # This is 729 * 729 = 531441 booleans, very manageable.
    cover = np.zeros((space_size, space_size), dtype=bool)
    for i in range(space_size):
        for nbr_idx in neighbors[i]:
            cover[i, nbr_idx] = True

    # Greedy set cover with diversity preference
    while uncovered.any():
        if time.time() > deadline - 15:
            break

        best_idx = -1
        best_score = -1
        # For each candidate, score = number of uncovered words it covers
        # Tie-break by distance from existing codewords (prefer farther)
        for cand_idx in range(space_size):
            count = np.count_nonzero(cover[cand_idx] & uncovered)
            if count == 0:
                continue
            # Diversity bonus: count how many existing codewords are far from this candidate
            diversity = 0
            if selected_indices:
                for sel_idx in selected_indices:
                    # Hamming distance between cand_idx and sel_idx
                    w1 = idx_to_word(cand_idx, q, n)
                    w2 = idx_to_word(sel_idx, q, n)
                    dist = sum(1 for a, b in zip(w1, w2) if a != b)
                    if dist >= 2:
                        diversity += 1
            # Score: prioritize coverage, use diversity as tie-breaker
            score = count * 1000 + diversity
            if score > best_score:
                best_score = score
                best_idx = cand_idx

        if best_idx == -1:
            break

        selected_indices.append(best_idx)
        uncovered[cover[best_idx]] = False

    # Build coverage counts
    coverage_count = np.zeros(space_size, dtype=np.int32)
    for ci in selected_indices:
        coverage_count[cover[ci]] += 1

    # Iterative removal
    removed = [False] * len(selected_indices)
    for _pass in range(100):
        if time.time() > deadline - 15:
            break
        removed_any = False
        for ci in range(len(selected_indices)):
            if removed[ci]:
                continue
            # Check if removing selected[ci] leaves any word uncovered
            can_remove = True
            for nbr_idx in range(space_size):
                if cover[selected_indices[ci], nbr_idx] and coverage_count[nbr_idx] == 1:
                    can_remove = False
                    break
            if can_remove:
                removed[ci] = True
                removed_any = True
                for nbr_idx in range(space_size):
                    if cover[selected_indices[ci], nbr_idx]:
                        coverage_count[nbr_idx] -= 1
        if not removed_any:
            break

    result_indices = [selected_indices[i] for i in range(len(selected_indices)) if not removed[i]]
    coverage_count = np.zeros(space_size, dtype=np.int32)
    for ci in result_indices:
        coverage_count[cover[ci]] += 1

    # Local search: try swapping codewords to reduce count
    if time.time() < deadline - 5:
        for iteration in range(500):
            if time.time() > deadline - 3:
                break

            improved = False
            # Try adding a random candidate and removing 2+ existing
            for trial in range(50):
                if time.time() > deadline - 3:
                    break

                cand_idx = random.randint(0, space_size - 1)
                if cand_idx in result_indices:
                    continue

                # Simulate adding cand_idx
                new_cov = coverage_count.copy()
                for nbr_idx in range(space_size):
                    if cover[cand_idx, nbr_idx]:
                        new_cov[nbr_idx] += 1

                # Try to greedily remove codewords
                temp_removed = [False] * len(result_indices)
                removable = 0
                for ri, rcw_idx in enumerate(result_indices):
                    if temp_removed[ri]:
                        continue
                    can_remove = True
                    for nbr_idx in range(space_size):
                        if cover[rcw_idx, nbr_idx] and new_cov[nbr_idx] <= 1:
                            can_remove = False
                            break
                    if can_remove:
                        temp_removed[ri] = True
                        removable += 1
                        for nbr_idx in range(space_size):
                            if cover[rcw_idx, nbr_idx]:
                                new_cov[nbr_idx] -= 1

                if removable >= 2:
                    # Apply
                    new_result_indices = [result_indices[ri] for ri in range(len(result_indices)) if not temp_removed[ri]]
                    new_result_indices.append(cand_idx)
                    result_indices = new_result_indices
                    coverage_count = new_cov
                    improved = True
                    break

            if not improved:
                break

    # Final validity check
    final_cov = np.zeros(space_size, dtype=np.int32)
    for ci in result_indices:
        final_cov[cover[ci]] += 1

    if not (final_cov > 0).all():
        # Fallback to tetracode direct sum
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    else:
        result = [idx_to_word(idx, q, n) for idx in result_indices]

    return result
# EVOLVE-BLOCK-END
