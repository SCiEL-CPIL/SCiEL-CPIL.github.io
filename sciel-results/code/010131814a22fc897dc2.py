"""Combined program for ternary covering code K_3(7,1)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    # Generate all words in the space
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    total_words = len(all_words)

    # Precompute coverage masks for all candidates
    coverage = np.zeros((total_words, total_words), dtype=bool)

    # Chunked distance computation to manage memory
    chunk_size = 500
    for i_start in range(0, total_words, chunk_size):
        i_end = min(i_start + chunk_size, total_words)
        diff = all_words[i_start:i_end, None, :] != all_words[None, :, :]
        dists = np.sum(diff, axis=2)
        coverage[i_start:i_end, :] = dists <= radius

    # Greedy Set Cover with improved strategy
    remaining = np.ones(total_words, dtype=bool)
    remaining_count = total_words
    selected_indices = []

    # Use int8 for faster dot product
    remaining_int8 = remaining.astype(np.int8)

    while remaining_count > 0:
        if time.time() > deadline - 15.0:
            break

        # Calculate coverage count for all candidates against remaining words
        counts = coverage.dot(remaining_int8)

        # Find candidate with max count
        best_idx = np.argmax(counts)
        best_count = counts[best_idx]

        if best_count == 0:
            break

        selected_indices.append(best_idx)

        # Update remaining
        remaining &= ~coverage[best_idx]
        remaining_int8 = remaining.astype(np.int8)
        remaining_count = np.count_nonzero(remaining)

    # Convert indices to codewords
    selected_codewords = [all_words[i].tolist() for i in selected_indices]

    # Post-processing: Remove redundant codewords
    if time.time() < deadline - 10.0:
        K = len(selected_indices)
        if K > 0:
            sel_cov = coverage[selected_indices, :]

            local_cover_counts = np.sum(sel_cov, axis=0)
            removed = [False] * K

            # Iterative removal of redundant codewords
            for _ in range(K):
                if time.time() > deadline - 3.0:
                    break

                to_remove = []
                for i in range(K):
                    if removed[i]:
                        continue
                    # Check if i covers any word with count 1 (essential)
                    if np.any(sel_cov[i] & (local_cover_counts == 1)):
                        continue
                    to_remove.append(i)

                if not to_remove:
                    break

                for i in to_remove:
                    removed[i] = True
                    local_cover_counts -= sel_cov[i]

            final_indices = [i for i in range(K) if not removed[i]]
            selected_codewords = [all_words[selected_indices[i]].tolist() for i in final_indices]

            # Simulated annealing refinement
            if time.time() < deadline - 5.0 and len(selected_codewords) > 2:
                active_idx = [selected_indices[i] for i in final_indices]
                K2 = len(active_idx)

                # Build a set for O(1) membership
                active_set = set(active_idx)

                # Precompute coverage for active codewords
                active_cov = np.array([coverage[idx] for idx in active_idx])
                cover_counts = np.sum(active_cov, axis=0)

                # Simulated annealing parameters
                temp_start = 1.0
                temp_end = 0.01
                cooling = 0.999
                max_iter = 50000
                temp = temp_start

                best_solution = list(active_idx)
                best_size = K2
                current_solution = list(active_idx)
                current_size = K2

                for it in range(max_iter):
                    if time.time() > deadline - 3.0:
                        break
                    if temp < temp_end:
                        break

                    # Randomly pick a codeword to remove
                    if current_size <= 1:
                        break

                    remove_pos = np.random.randint(current_size)
                    remove_idx = current_solution[remove_pos]

                    # Words that would become uncovered
                    words_to_check = active_cov[remove_pos] & (cover_counts == 1)
                    n_uncovered = np.count_nonzero(words_to_check)

                    if n_uncovered == 0:
                        # Safe removal
                        # Remove from solution
                        current_solution.pop(remove_pos)
                        cover_counts -= active_cov[remove_pos]
                        active_cov = np.delete(active_cov, remove_pos, axis=0)
                        current_size -= 1

                        if current_size < best_size:
                            best_size = current_size
                            best_solution = list(current_solution)

                        # Try to add a new codeword that covers the most remaining uncovered words
                        # But first, let's see if we can just keep going
                        continue

                    # Try to find a replacement codeword
                    # Find a codeword that covers many of the words_to_check
                    # and is not already in the solution
                    # We want to minimize the net uncovered words

                    # Compute how many of words_to_check each candidate covers
                    # This is expensive, so sample a few candidates
                    n_candidates = min(50, total_words)
                    candidate_indices = np.random.choice(total_words, n_candidates, replace=False)

                    best_candidate = -1
                    best_net_uncovered = n_uncovered  # if we just remove without replacement

                    for cand_idx in candidate_indices:
                        if cand_idx in active_set:
                            continue
                        # How many of words_to_check does cand_idx cover?
                        cand_cov = coverage[cand_idx]
                        n_covered = np.count_nonzero(words_to_check & cand_cov)
                        net_uncovered = n_uncovered - n_covered

                        if net_uncovered < best_net_uncovered:
                            best_net_uncovered = net_uncovered
                            best_candidate = cand_idx

                    if best_candidate >= 0 and best_net_uncovered < n_uncovered:
                        # Accept the swap
                        # Remove the old codeword
                        old_cov = active_cov[remove_pos]
                        cover_counts -= old_cov
                        active_cov = np.delete(active_cov, remove_pos, axis=0)
                        current_solution.pop(remove_pos)
                        current_size -= 1
                        active_set.discard(remove_idx)

                        # Add the new codeword
                        new_cov = coverage[best_candidate]
                        cover_counts += new_cov
                        active_cov = np.vstack([active_cov, new_cov.reshape(1, -1)])
                        current_solution.append(best_candidate)
                        current_size += 1
                        active_set.add(best_candidate)

                        # Check if this is better
                        if current_size < best_size:
                            best_size = current_size
                            best_solution = list(current_solution)

                    else:
                        # No good replacement found, skip
                        pass

                    temp *= cooling

                # Use the best solution found
                if best_size < current_size:
                    active_idx = best_solution
                else:
                    active_idx = current_solution

                selected_codewords = [all_words[i].tolist() for i in active_idx]

    return selected_codewords
# EVOLVE-BLOCK-END
