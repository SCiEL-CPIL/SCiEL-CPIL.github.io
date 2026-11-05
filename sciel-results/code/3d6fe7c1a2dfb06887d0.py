import itertools
import time
import random
import numpy as np

# Global Precomputation
V = 19
K = 7
T = 4

# Map 4-subsets to indices
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

# Generate all 7-subsets and their data
all_blocks = list(itertools.combinations(range(V), K))
N_BLOCKS = len(all_blocks)

all_masks = [0] * N_BLOCKS
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]
subset_frequencies = [0] * N_SUBSETS
block_bit_indices = [[] for _ in range(N_BLOCKS)]

for i, block in enumerate(all_blocks):
    mask = 0
    bits = []
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        blocks_containing_subset[idx].append(i)
        subset_frequencies[idx] += 1
        bits.append(idx)
    all_masks[i] = mask
    block_bit_indices[i] = bits

# Precompute weights for weighted greedy: prioritize rare subsets
weights = np.array([1.0 / f for f in subset_frequencies], dtype=np.float64)

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_subsets = list(itertools.combinations(range(v), k))
        random.shuffle(all_k_subsets)
        for b in all_k_subsets:
            if not uncovered:
                break
            covered = set(itertools.combinations(b, t))
            if uncovered & covered:
                blocks.append(list(b))
                uncovered -= covered
        return blocks

    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # 1. Weighted Greedy Initialization
    counts = np.zeros(N_SUBSETS, dtype=np.int32)
    current_blocks = []
    current_block_set = set()
    uncovered_mask = (1 << N_SUBSETS) - 1

    # Precompute block weights contribution for faster gain
    # gain[i] = sum(weights[bit] for bit in block_bit_indices[i] if bit is uncovered)
    # We'll recompute this efficiently

    def compute_gains(uncovered_mask):
        """Compute weighted gain for each block given uncovered_mask"""
        gains = np.zeros(N_BLOCKS, dtype=np.float64)
        for i in range(N_BLOCKS):
            bits = block_bit_indices[i]
            g = 0.0
            for bit in bits:
                if (uncovered_mask >> bit) & 1:
                    g += weights[bit]
            gains[i] = g
        return gains

    # Greedy loop with weighted selection
    while uncovered_mask:
        if time.time() > deadline - 60:
            break

        # Compute gains for all blocks
        gains = compute_gains(uncovered_mask)

        # Find block with maximum gain (tie-break: prefer blocks already not in set)
        best_idx = -1
        best_gain = 0.0
        for i in range(N_BLOCKS):
            if i in current_block_set:
                continue
            g = gains[i]
            if g > best_gain:
                best_gain = g
                best_idx = i

        if best_idx == -1 or best_gain == 0:
            break

        # Add the block
        current_blocks.append(best_idx)
        current_block_set.add(best_idx)
        for bit in block_bit_indices[best_idx]:
            counts[bit] += 1
        uncovered_mask &= ~all_masks[best_idx]

    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()

    # 2. Local Search
    no_improve_count = 0
    max_no_improve = 30

    def is_redundant(idx):
        """Check if block idx is redundant given current counts"""
        for bit in block_bit_indices[idx]:
            if counts[bit] == 1:
                return False
        return True

    def remove_block(idx):
        """Remove block idx from current solution"""
        current_block_set.remove(idx)
        pos = current_blocks.index(idx)
        current_blocks[pos] = current_blocks[-1]
        current_blocks.pop()
        for bit in block_bit_indices[idx]:
            counts[bit] -= 1

    def add_block(idx):
        """Add block idx to current solution"""
        current_blocks.append(idx)
        current_block_set.add(idx)
        for bit in block_bit_indices[idx]:
            counts[bit] += 1

    def try_swap(remove_idx, add_idx):
        """Try swapping remove_idx with add_idx. Returns True if successful."""
        if add_idx in current_block_set:
            return False
        # Check if after removing remove_idx, all its unique subsets are covered by add_idx
        m = all_masks[remove_idx]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            if counts[bit_idx] == 1:
                # This subset is only covered by remove_idx
                # Check if add_idx covers it
                if not (all_masks[add_idx] >> bit_idx) & 1:
                    return False
            mm ^= lsb
        # Swap
        remove_block(remove_idx)
        add_block(add_idx)
        return True

    while time.time() < deadline - 5:
        # Phase 1: Prune redundant blocks
        removed_any = False
        indices = list(range(len(current_blocks)))
        random.shuffle(indices)

        for i in indices:
            if time.time() > deadline - 5:
                break
            idx = current_blocks[i]
            if is_redundant(idx):
                remove_block(idx)
                removed_any = True
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                break

        if removed_any:
            no_improve_count = 0
            continue

        # Phase 2: Try 1-for-1 swaps
        if not current_blocks:
            break

        found_swap = False
        # Try multiple random blocks for swap
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            i = random.randint(0, len(current_blocks) - 1)
            idx = current_blocks[i]

            # Find unique subsets covered by idx
            m = all_masks[idx]
            unique_bits = []
            mm = m
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    unique_bits.append(bit_idx)
                mm ^= lsb

            if not unique_bits:
                # Redundant block, remove it
                remove_block(idx)
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                no_improve_count = 0
                found_swap = True
                break
            else:
                # Find candidates that cover all unique_bits
                if len(unique_bits) <= 5:
                    # Intersection of candidate lists
                    candidates = set(blocks_containing_subset[unique_bits[0]])
                    for b in unique_bits[1:]:
                        if not candidates:
                            break
                        candidates &= set(blocks_containing_subset[b])

                    # Filter out blocks already in cover
                    valid_candidates = [c for c in candidates if c not in current_block_set]

                    if valid_candidates:
                        # Choose candidate with highest weighted gain for the unique bits
                        best_candidate = None
                        best_candidate_gain = -1.0
                        for c in valid_candidates:
                            gain = sum(weights[b] for b in unique_bits if (all_masks[c] >> b) & 1)
                            # Also consider overall coverage quality
                            gain += 0.1 * sum(weights[b] for b in block_bit_indices[c] if counts[b] == 0)
                            if gain > best_candidate_gain:
                                best_candidate_gain = gain
                                best_candidate = c

                        if best_candidate is not None:
                            if try_swap(idx, best_candidate):
                                found_swap = True
                                no_improve_count = 0
                                break

        if not found_swap:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturbation: remove blocks with highest redundancy
                n_remove = random.randint(3, 7)
                # Score each block by its redundancy (how many subsets it uniquely covers)
                block_scores = []
                for idx in current_blocks:
                    unique_count = 0
                    m = all_masks[idx]
                    mm = m
                    while mm:
                        lsb = mm & -mm
                        bit_idx = lsb.bit_length() - 1
                        if counts[bit_idx] == 1:
                            unique_count += 1
                        mm ^= lsb
                    block_scores.append((unique_count, idx))

                # Sort by unique_count (remove most redundant first)
                block_scores.sort(key=lambda x: x[0])
                to_remove = [idx for _, idx in block_scores[:min(n_remove, len(current_blocks))]]

                for idx in to_remove:
                    remove_block(idx)

                # Re-greedy to recover coverage
                if time.time() < deadline - 10:
                    # Compute uncovered
                    uncov_mask = 0
                    for s in range(N_SUBSETS):
                        if counts[s] == 0:
                            uncov_mask |= 1 << s

                    while uncov_mask:
                        if time.time() > deadline - 5:
                            break
                        # Weighted greedy for uncovered
                        best_idx = -1
                        best_gain = 0.0
                        for i in range(N_BLOCKS):
                            if i in current_block_set:
                                continue
                            g = 0.0
                            for bit in block_bit_indices[i]:
                                if (uncov_mask >> bit) & 1:
                                    g += weights[bit]
                            if g > best_gain:
                                best_gain = g
                                best_idx = i
                        if best_idx == -1 or best_gain == 0:
                            break
                        add_block(best_idx)
                        uncov_mask &= ~all_masks[best_idx]

                no_improve_count = 0
                max_no_improve = min(150, max_no_improve * 2)

    # 3. Final polish: multiple passes of redundant removal
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(current_blocks) - 1, -1, -1):
            if time.time() > deadline:
                break
            idx = current_blocks[i]
            if is_redundant(idx):
                remove_block(idx)
                changed = True
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()

    # Return the best solution found
    return [list(all_blocks[i]) for i in best_blocks]
