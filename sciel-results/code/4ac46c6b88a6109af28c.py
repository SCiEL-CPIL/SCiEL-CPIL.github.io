"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    total_words = q ** n
    powers = [q ** p for p in range(n)]

    # Precompute for each word, the words within Hamming distance <= radius
    # For radius=1, q=3, n=6: each word covers 1 + 6*2 = 13 words
    neighbors = np.empty((total_words, 1 + n * (q - 1)), dtype=np.int32)
    for i in range(total_words):
        digits = [(i // powers[p]) % q for p in range(n)]
        neighbors[i, 0] = i
        k = 1
        for p in range(n):
            for alt in range(q):
                if alt != digits[p]:
                    new_i = i + (alt - digits[p]) * powers[p]
                    neighbors[i, k] = new_i
                    k += 1

    # Function to convert index to word list
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    def word_to_idx(word):
        idx = 0
        for v in word:
            idx = idx * q + v
        return idx

    # Build initial code using tetracode direct sum
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    if n >= 4:
        initial_codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
    else:
        initial_codewords = [list(word) for word in itertools.product(range(q), repeat=n)]

    initial_idx_list = [word_to_idx(cw) for cw in initial_codewords]

    # Also try greedy set cover construction
    def greedy_cover():
        uncovered = np.ones(total_words, dtype=np.int8)
        codewords = []
        while uncovered.sum() > 0:
            if time.time() > deadline - 2.0:
                break
            # Find candidate that covers most uncovered words
            counts = np.sum(uncovered[neighbors], axis=1)
            best_candidate = np.argmax(counts)
            if counts[best_candidate] == 0:
                break
            codewords.append(best_candidate)
            uncovered[neighbors[best_candidate]] = 0
        return codewords

    greedy_idx_list = greedy_cover()

    # Choose the better starting point
    if len(greedy_idx_list) < len(initial_idx_list):
        best_cw_idx_list = greedy_idx_list
    else:
        best_cw_idx_list = initial_idx_list

    best_len = len(best_cw_idx_list)

    # Local search: remove redundant codewords
    def try_remove_redundant(cw_idx_list, max_passes=100):
        improved = True
        passes = 0
        while improved and passes < max_passes and time.time() < deadline - 2.0:
            improved = False
            passes += 1
            # Build coverage count
            counts = np.zeros(total_words, dtype=np.int32)
            for cw_idx in cw_idx_list:
                np.add.at(counts, neighbors[cw_idx], 1)

            # Try to find a redundant codeword
            for i in range(len(cw_idx_list) - 1, -1, -1):
                if time.time() > deadline - 2.0:
                    break
                covered = neighbors[cw_idx_list[i]]
                if np.all(counts[covered] > 1):
                    cw_idx_list.pop(i)
                    improved = True
                    break
        return cw_idx_list

    # Apply removal to initial best
    best_cw_idx_list = try_remove_redundant(best_cw_idx_list.copy())
    if len(best_cw_idx_list) < best_len:
        best_len = len(best_cw_idx_list)

    # Multiple restarts with different strategies
    for restart in range(50):
        if time.time() > deadline - 5.0:
            break

        # Start from a copy of the best
        cw_idx_list = best_cw_idx_list.copy()

        # Phase 1: Greedy removal with random order
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            counts = np.zeros(total_words, dtype=np.int32)
            for cw_idx in cw_idx_list:
                np.add.at(counts, neighbors[cw_idx], 1)

            # Try random order
            indices = list(range(len(cw_idx_list)))
            random.shuffle(indices)
            for i in indices:
                if time.time() > deadline - 3.0:
                    break
                covered = neighbors[cw_idx_list[i]]
                if np.all(counts[covered] > 1):
                    cw_idx_list.pop(i)
                    improved = True
                    break

        # Phase 2: Swap-based local search
        if time.time() < deadline - 3.0:
            swap_improved = True
            max_swap_iters = 30
            for _ in range(max_swap_iters):
                if time.time() > deadline - 3.0:
                    break
                if not swap_improved:
                    break
                swap_improved = False

                counts = np.zeros(total_words, dtype=np.int32)
                for cw_idx in cw_idx_list:
                    np.add.at(counts, neighbors[cw_idx], 1)

                # Find codewords with least unique coverage
                unique_counts = []
                for i, cw_idx in enumerate(cw_idx_list):
                    covered = neighbors[cw_idx]
                    unique = np.sum(counts[covered] == 1)
                    unique_counts.append((unique, i))

                unique_counts.sort()

                # Try swapping the least useful codeword
                for unique, i in unique_counts[:5]:
                    if time.time() > deadline - 3.0:
                        break
                    if unique > 0:
                        continue

                    # Try adding a random codeword and removing the current one
                    new_cw_idx = random.randint(0, total_words - 1)
                    if new_cw_idx in cw_idx_list:
                        continue

                    test_cw_idx_list = cw_idx_list.copy()
                    test_cw_idx_list.pop(i)
                    test_cw_idx_list.append(new_cw_idx)

                    # Check validity
                    test_counts = np.zeros(total_words, dtype=np.int32)
                    for cw_idx in test_cw_idx_list:
                        np.add.at(test_counts, neighbors[cw_idx], 1)
                    if np.any(test_counts == 0):
                        continue

                    # Try to remove more
                    test_cw_idx_list = try_remove_redundant(test_cw_idx_list, max_passes=20)

                    if len(test_cw_idx_list) < len(cw_idx_list):
                        cw_idx_list = test_cw_idx_list
                        swap_improved = True
                        break

        # Update best
        if len(cw_idx_list) < best_len:
            best_len = len(cw_idx_list)
            best_cw_idx_list = cw_idx_list.copy()

        # If we're at or near the known best, try perturbation
        if best_len <= 75 and time.time() < deadline - 5.0:
            for pert in range(5):
                if time.time() > deadline - 5.0:
                    break
                perturbed = best_cw_idx_list.copy()
                if len(perturbed) > 10:
                    remove_idx = random.randint(0, len(perturbed) - 1)
                    perturbed.pop(remove_idx)

                    # Try to fix coverage
                    for _ in range(20):
                        if time.time() > deadline - 5.0:
                            break
                        counts = np.zeros(total_words, dtype=np.int32)
                        for cw_idx in perturbed:
                            np.add.at(counts, neighbors[cw_idx], 1)
                        if np.all(counts > 0):
                            break
                        uncovered_indices = np.where(counts == 0)[0]
                        if len(uncovered_indices) == 0:
                            break
                        unc_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
                        # Find candidates that cover unc_idx
                        candidates = np.where(neighbors[:, :] == unc_idx)
                        candidate_list = candidates[0].tolist()
                        if candidate_list:
                            new_cw_idx = random.choice(candidate_list)
                            if new_cw_idx not in perturbed:
                                perturbed.append(new_cw_idx)

                    perturbed = try_remove_redundant(perturbed, max_passes=20)

                    # Check validity
                    final_counts = np.zeros(total_words, dtype=np.int32)
                    for cw_idx in perturbed:
                        np.add.at(final_counts, neighbors[cw_idx], 1)
                    if np.all(final_counts > 0) and len(perturbed) < best_len:
                        best_len = len(perturbed)
                        best_cw_idx_list = perturbed.copy()

    # Final cleanup
    best_cw_idx_list = try_remove_redundant(best_cw_idx_list.copy(), max_passes=50)

    return [idx_to_word(idx) for idx in best_cw_idx_list]
# EVOLVE-BLOCK-END
