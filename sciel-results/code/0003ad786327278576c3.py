"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np
import random


def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius
    'radius'. Uses a bitmask-based greedy set-cover approach, then local search
    (remove/swap moves) for optimization.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    if q == 3 and n == 7 and radius == 1:
        # Precompute all words
        all_words = list(itertools.product(range(3), repeat=7))
        word_to_idx = {w: i for i, w in enumerate(all_words)}
        total_words = len(all_words)

        # Precompute coverage as bitmasks (Python ints for fast operations)
        # coverage_mask[i] = bitmask of all words covered by codeword i
        coverage_mask = [0] * total_words
        for i, w in enumerate(all_words):
            mask = 1 << i
            for pos in range(n):
                for val in range(q):
                    if val != w[pos]:
                        new_list = list(w)
                        new_list[pos] = val
                        new_word = tuple(new_list)
                        mask |= 1 << word_to_idx[new_word]
            coverage_mask[i] = mask

        # Convert to numpy array of uint64 for batch operations if possible
        # But Python ints are arbitrary precision, so we'll use them directly
        # Actually, let's use a list of Python ints for bitmask operations

        # Start with tetracode product construction for initial solution
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial_codewords = []
        for h in tetracode:
            for tail in itertools.product(range(3), repeat=3):
                initial_codewords.append(tuple(h + list(tail)))

        # Convert to indices
        codeword_indices = [word_to_idx[cw] for cw in initial_codewords]

        # Compute coverage using bitmask OR
        covered_mask = 0
        for cw_idx in codeword_indices:
            covered_mask |= coverage_mask[cw_idx]

        # Greedily add codewords to cover remaining words
        while time.time() < deadline - 60:
            # Find uncovered words
            uncovered = 0
            for i in range(total_words):
                if not (covered_mask >> i) & 1:
                    uncovered |= 1 << i

            if uncovered == 0:
                break

            # Find the codeword that covers the most uncovered words
            best_cw = -1
            best_count = 0
            for cw_idx in range(total_words):
                # Count how many uncovered words this codeword covers
                new_covered = coverage_mask[cw_idx] & uncovered
                count = bin(new_covered).count('1')
                if count > best_count:
                    best_count = count
                    best_cw = cw_idx

            if best_count <= 0:
                break

            codeword_indices.append(best_cw)
            covered_mask |= coverage_mask[best_cw]

        # Local search: try to remove redundant codewords
        best_indices = list(codeword_indices)
        best_size = len(best_indices)
        current_indices = list(codeword_indices)
        current_size = len(current_indices)

        # Rebuild coverage mask for current solution
        cur_covered = 0
        for cw_idx in current_indices:
            cur_covered |= coverage_mask[cw_idx]

        # Track coverage counts for faster redundancy checks
        # cov_count[word_idx] = number of codewords covering word_idx
        cov_count = np.zeros(total_words, dtype=np.int32)
        for cw_idx in current_indices:
            # Add coverage of this codeword
            mask = coverage_mask[cw_idx]
            for i in range(total_words):
                if (mask >> i) & 1:
                    cov_count[i] += 1

        # Phase 1: Aggressive removal of redundant codewords
        progress = True
        while progress and time.time() < deadline - 30:
            progress = False
            i = 0
            while i < len(current_indices) and time.time() < deadline - 30:
                cw_to_remove = current_indices[i]
                # Check if removing this codeword keeps all words covered
                # A word becomes uncovered if its coverage count drops to 0
                mask = coverage_mask[cw_to_remove]
                would_uncover = 0
                for j in range(total_words):
                    if (mask >> j) & 1:
                        if cov_count[j] == 1:
                            would_uncover += 1

                if would_uncover == 0:
                    # Safe removal
                    for j in range(total_words):
                        if (mask >> j) & 1:
                            cov_count[j] -= 1
                    current_indices.pop(i)
                    current_size -= 1
                    progress = True
                    if current_size < best_size:
                        best_size = current_size
                        best_indices = list(current_indices)
                else:
                    i += 1

        # Phase 2: Swap-based local search
        # Try replacing one codeword with another to improve coverage efficiency
        swap_deadline = deadline - 10
        temperature = 1.0
        cooling_rate = 0.999

        while time.time() < swap_deadline:
            if current_size <= 1:
                break

            # Try to remove a random codeword
            pos = random.randint(0, current_size - 1)
            cw_to_remove = current_indices[pos]

            mask_remove = coverage_mask[cw_to_remove]
            would_uncover = 0
            for j in range(total_words):
                if (mask_remove >> j) & 1:
                    if cov_count[j] == 1:
                        would_uncover += 1

            if would_uncover == 0:
                # Safe removal
                for j in range(total_words):
                    if (mask_remove >> j) & 1:
                        cov_count[j] -= 1
                current_indices.pop(pos)
                current_size -= 1

                if current_size < best_size:
                    best_size = current_size
                    best_indices = list(current_indices)
            else:
                # Try to find a replacement codeword that covers the uncovered words
                # Find which words would become uncovered
                uncovered_words = []
                for j in range(total_words):
                    if (mask_remove >> j) & 1:
                        if cov_count[j] == 1:
                            uncovered_words.append(j)

                if uncovered_words:
                    # Find a codeword that covers all uncovered_words and is not already in the list
                    best_add = -1
                    best_overlap = -1
                    for candidate in range(total_words):
                        if candidate in current_indices:
                            continue
                        mask_add = coverage_mask[candidate]
                        # Check if it covers all uncovered words
                        covers_all = True
                        for j in uncovered_words:
                            if not (mask_add >> j) & 1:
                                covers_all = False
                                break
                        if covers_all:
                            # Count how many currently covered words it also covers (redundancy)
                            overlap = 0
                            for j in range(total_words):
                                if (mask_add >> j) & 1 and cov_count[j] > 0:
                                    overlap += 1
                            if overlap > best_overlap:
                                best_overlap = overlap
                                best_add = candidate

                    if best_add >= 0:
                        # Perform the swap
                        # Remove old codeword
                        for j in range(total_words):
                            if (mask_remove >> j) & 1:
                                cov_count[j] -= 1
                        # Add new codeword
                        mask_add = coverage_mask[best_add]
                        for j in range(total_words):
                            if (mask_add >> j) & 1:
                                cov_count[j] += 1
                        current_indices[pos] = best_add

                        if current_size < best_size:
                            best_size = current_size
                            best_indices = list(current_indices)

            temperature *= cooling_rate
            if temperature < 0.01:
                temperature = 0.01

        # Use the best solution found
        final_indices = best_indices

        # Final redundancy removal pass (deterministic)
        final_cov_count = np.zeros(total_words, dtype=np.int32)
        for cw_idx in final_indices:
            mask = coverage_mask[cw_idx]
            for i in range(total_words):
                if (mask >> i) & 1:
                    final_cov_count[i] += 1

        i = 0
        while i < len(final_indices) and time.time() < deadline - 2:
            cw_to_remove = final_indices[i]
            mask = coverage_mask[cw_to_remove]
            would_uncover = 0
            for j in range(total_words):
                if (mask >> j) & 1:
                    if final_cov_count[j] == 1:
                        would_uncover += 1

            if would_uncover == 0:
                for j in range(total_words):
                    if (mask >> j) & 1:
                        final_cov_count[j] -= 1
                final_indices.pop(i)
            else:
                i += 1

        return [list(all_words[idx]) for idx in final_indices]

    # General fallback: greedy set cover
    all_words = list(itertools.product(range(q), repeat=n))

    def get_coverage_general(codeword):
        covered = set()
        covered.add(codeword)
        for i in range(n):
            for val in range(q):
                if val != codeword[i]:
                    new_word = list(codeword)
                    new_word[i] = val
                    covered.add(tuple(new_word))
        return covered

    codewords = []
    covered = set()

    while time.time() < deadline - 10:
        best_cw = None
        best_count = -1
        best_new_covered = set()

        for candidate in all_words:
            new_covered = get_coverage_general(candidate) - covered
            count = len(new_covered)
            if count > best_count:
                best_count = count
                best_cw = candidate
                best_new_covered = new_covered

        if best_count <= 0:
            break

        codewords.append(best_cw)
        covered.update(best_new_covered)

        if len(covered) == len(all_words):
            break

    # Redundancy removal
    i = 0
    while i < len(codewords) and time.time() < deadline - 5:
        cw_to_remove = codewords[i]
        coverage_of_cw = get_coverage_general(cw_to_remove)

        can_remove = True
        for w in coverage_of_cw:
            is_covered_by_other = False
            for j, other_cw in enumerate(codewords):
                if i == j:
                    continue
                if w in get_coverage_general(other_cw):
                    is_covered_by_other = True
                    break
            if not is_covered_by_other:
                can_remove = False
                break

        if can_remove:
            codewords.pop(i)
        else:
            i += 1

    return [list(cw) for cw in codewords]
# EVOLVE-BLOCK-END
