"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def construct(q, n, radius):
    deadline = time.time() + 110.0
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    base_size = 9 * (3 ** (n - 4))
    base = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    if len(base) <= 156:
        return base

    total_words = q ** n
    if total_words > 500000:
        return base

    # Precompute all words as indices
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.uint8)

    # Convert base to array
    base_arr = np.array(base, dtype=np.uint8)  # (base_size, n)

    # Compute covering sets as boolean array (base_size, total_words)
    if base_size * total_words > 200_000_000:
        return base

    # Compute Hamming distances in chunks
    chunk_size = max(1, 200_000_000 // (total_words * n))
    covering_sets = []
    for i in range(0, base_size, chunk_size):
        chunk = base_arr[i:i + chunk_size]
        dist = np.sum(chunk[:, np.newaxis, :] != all_words[np.newaxis, :, :], axis=2)
        covered = dist <= radius
        covering_sets.append(covered)

    covering_sets = np.concatenate(covering_sets, axis=0)  # (base_size, total_words)
    # Convert to int8 for fast arithmetic
    cov = covering_sets.astype(np.int8)  # (base_size, total_words)

    # Phase 1: Greedy set cover to get an initial solution
    uncovered = np.ones(total_words, dtype=np.int8)
    selected = []
    selected_mask = np.zeros(base_size, dtype=np.uint8)

    rng = random.Random(42)

    # Greedy: always pick the codeword that covers the most uncovered words
    while uncovered.sum() > 0 and time.time() < deadline - 80:
        gains = (cov * uncovered).sum(axis=1)
        best = int(np.argmax(gains))
        if gains[best] == 0:
            break
        selected.append(best)
        selected_mask[best] = 1
        uncovered[cov[best] == 1] = 0

    # Phase 2: Binary inclusion vector simulated annealing
    # x: binary vector of length base_size; x[i]=1 means codeword i is selected
    # Energy: M * (number of uncovered words) + (number of selected codewords)
    # Use a coverage count vector: coverage[w] = number of selected codewords covering word w
    # A word is uncovered iff coverage[w] == 0

    M = 1000  # Heavy penalty for uncovered words

    x = selected_mask.copy()  # binary inclusion vector
    # Compute coverage counts
    coverage = cov[x == 1].sum(axis=0).astype(np.int32)  # (total_words,)
    current_uncovered_count = int((coverage == 0).sum())
    current_selected_count = int(x.sum())
    current_energy = M * current_uncovered_count + current_selected_count

    best_x = x.copy()
    best_energy = current_energy
    best_uncovered_count = current_uncovered_count
    best_selected_count = current_selected_count

    # Simulated annealing parameters
    T_start = 3.0
    T_end = 0.05
    cooling_rate = 0.998
    max_iter = 200000

    T = T_start
    iter_count = 0
    rng_np = np.random.RandomState(42)

    while T > T_end and iter_count < max_iter and time.time() < deadline - 5:
        iter_count += 1

        # Choose move type: flip a bit (add or remove a codeword)
        # With some probability, do a 2-flip swap (remove one, add another)
        if rng.random() < 0.3 and current_selected_count > 1:
            # Swap: remove one selected, add one unselected
            selected_indices = np.where(x == 1)[0]
            unselected_indices = np.where(x == 0)[0]
            if len(selected_indices) == 0 or len(unselected_indices) == 0:
                T *= cooling_rate
                continue
            rem = selected_indices[rng_np.randint(0, len(selected_indices) - 1)]
            add = unselected_indices[rng_np.randint(0, len(unselected_indices) - 1)]

            # Compute delta
            # Removing rem: coverage decreases by cov[rem]
            # Adding add: coverage increases by cov[add]
            new_coverage = coverage - cov[rem] + cov[add]
            new_uncovered = int((new_coverage == 0).sum())
            new_selected = current_selected_count  # swap keeps count same
            new_energy = M * new_uncovered + new_selected
            delta_E = new_energy - current_energy

            if delta_E < 0 or rng.random() < np.exp(-delta_E / max(T, 1e-6)):
                x[rem] = 0
                x[add] = 1
                coverage = new_coverage
                current_uncovered_count = new_uncovered
                current_selected_count = new_selected
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_x = x.copy()
                    best_uncovered_count = current_uncovered_count
                    best_selected_count = current_selected_count
        else:
            # Single flip: toggle one bit
            # Bias towards removing selected bits (to reduce count) when covered
            # and adding unselected bits when there are uncovered words
            if current_uncovered_count > 0 and rng.random() < 0.5:
                # Try to add an unselected codeword that covers uncovered words
                unselected_indices = np.where(x == 0)[0]
                if len(unselected_indices) == 0:
                    T *= cooling_rate
                    continue
                # Pick one that covers uncovered words
                # Compute gain for each unselected
                unc_mask = (coverage == 0).astype(np.int8)
                gains = (cov[unselected_indices] * unc_mask).sum(axis=1)
                # Pick randomly among top gainers or the best
                if gains.max() > 0:
                    # Pick from top 10% or best
                    top_thresh = max(1, int(gains.max() * 0.5))
                    candidates = unselected_indices[gains >= top_thresh]
                    add = candidates[rng_np.randint(0, len(candidates) - 1)]
                else:
                    add = unselected_indices[rng_np.randint(0, len(unselected_indices) - 1)]

                new_coverage = coverage + cov[add]
                new_uncovered = int((new_coverage == 0).sum())
                new_selected = current_selected_count + 1
                new_energy = M * new_uncovered + new_selected
                delta_E = new_energy - current_energy

                if delta_E < 0 or rng.random() < np.exp(-delta_E / max(T, 1e-6)):
                    x[add] = 1
                    coverage = new_coverage
                    current_uncovered_count = new_uncovered
                    current_selected_count = new_selected
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_x = x.copy()
                        best_uncovered_count = current_uncovered_count
                        best_selected_count = current_selected_count
            else:
                # Try to remove a selected codeword
                selected_indices = np.where(x == 1)[0]
                if len(selected_indices) == 0:
                    T *= cooling_rate
                    continue
                rem = selected_indices[rng_np.randint(0, len(selected_indices) - 1)]

                new_coverage = coverage - cov[rem]
                new_uncovered = int((new_coverage == 0).sum())
                new_selected = current_selected_count - 1
                new_energy = M * new_uncovered + new_selected
                delta_E = new_energy - current_energy

                if delta_E < 0 or rng.random() < np.exp(-delta_E / max(T, 1e-6)):
                    x[rem] = 0
                    coverage = new_coverage
                    current_uncovered_count = new_uncovered
                    current_selected_count = new_selected
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_x = x.copy()
                        best_uncovered_count = current_uncovered_count
                        best_selected_count = current_selected_count

        # Cool down
        T *= cooling_rate

    # Return the best solution found
    result_indices = np.where(best_x == 1)[0]
    result = [base[i] for i in result_indices]
    return result
# EVOLVE-BLOCK-END
