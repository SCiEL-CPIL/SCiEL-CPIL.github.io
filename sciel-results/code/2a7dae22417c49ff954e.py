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

# Generate all 7-subsets and their masks
all_blocks = []
all_masks = []
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]

for block in itertools.combinations(range(V), K):
    mask = 0
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        blocks_containing_subset[idx].append(len(all_blocks))
    all_blocks.append(block)
    all_masks.append(mask)

N_BLOCKS = len(all_blocks)

# Precompute a numpy uint64 view of the masks for vectorized gain computation
all_masks_arr = np.array(all_masks, dtype=np.uint64)

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters (not expected in this task)
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
    deadline = start_time + 118  # Safety margin before the 150s kill

    # 1. Greedy Initialization (vectorized)
    counts = np.zeros(N_SUBSETS, dtype=np.int32)
    current_blocks = []
    current_block_set = set()

    def add_block(idx):
        current_blocks.append(idx)
        current_block_set.add(idx)
        counts[all_blocks_bits(idx)] += 1

    # Precompute per-block list of bit indices for fast count updates
    _bits_cache = {}
    def all_blocks_bits(idx):
        b = _bits_cache.get(idx)
        if b is None:
            b = [i for i in range(N_SUBSETS) if (all_masks[idx] >> i) & 1]
            _bits_cache[idx] = b
        return b

    def remove_block(idx):
        current_block_set.remove(idx)
        pos = current_blocks.index(idx)
        current_blocks[pos] = current_blocks[-1]
        current_blocks.pop()
        counts[all_blocks_bits(idx)] -= 1

    def is_redundant(idx):
        for b in all_blocks_bits(idx):
            if counts[b] == 1:
                return False
        return True

    def uncovered_mask():
        # numpy: bits where count == 0
        zero_mask = (counts == 0)
        # build integer mask
        m = 0
        # iterate set bits of zero_mask
        arr = zero_mask.astype(np.uint8)
        # accumulate
        m = 0
        for i in range(N_SUBSETS):
            if arr[i]:
                m |= (1 << i)
        return m

    # Greedy loop using vectorized gain
    uncovered_mask = (1 << N_SUBSETS) - 1
    while uncovered_mask:
        if time.time() > deadline - 60:
            break
        gains = np.bitwise_and(all_masks_arr, np.uint64(uncovered_mask)).bit_count(axis=1)
        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break
        add_block(best_idx)
        uncovered_mask &= ~all_masks[best_idx]

    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()

    # 2. Local Search with Perturbation (ILS style)
    no_improve_count = 0
    max_no_improve = 50

    while time.time() < deadline - 3:
        # Phase 1: Prune redundant blocks
        removed_any = False
        indices = list(range(len(current_blocks)))
        random.shuffle(indices)

        for i in indices:
            if time.time() > deadline - 3:
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
            max_no_improve = max(50, max_no_improve // 2)
            continue

        # Phase 2: Try 1-for-1 swap
        if not current_blocks:
            break

        found_swap = False
        i = random.randint(0, len(current_blocks) - 1)
        idx = current_blocks[i]

        # Find unique subsets covered by idx
        unique_bits = [b for b in all_blocks_bits(idx) if counts[b] == 1]

        if not unique_bits:
            remove_block(idx)
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = current_blocks.copy()
            no_improve_count = 0
            continue
        else:
            if len(unique_bits) <= 4:
                unique_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
                candidates = set(blocks_containing_subset[unique_bits[0]])
                for b in unique_bits[1:]:
                    if not candidates:
                        break
                    candidates &= set(blocks_containing_subset[b])
                valid_candidates = [c for c in candidates if c not in current_block_set]
                if valid_candidates:
                    new_idx = random.choice(valid_candidates)
                    remove_block(idx)
                    add_block(new_idx)
                    found_swap = True
                    no_improve_count = 0

        if not found_swap:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturbation: either remove random blocks + regreedy, or add a random block
                mode = random.random()
                if mode < 0.6:
                    n_remove = random.randint(3, 6)
                    to_remove = random.sample(current_blocks, min(n_remove, len(current_blocks)))
                    for idx2 in to_remove:
                        remove_block(idx2)
                    # Re-greedy repair (vectorized)
                    if time.time() < deadline - 8:
                        uncov_mask = uncovered_mask()
                        while uncov_mask:
                            if time.time() > deadline - 3:
                                break
                            gains = np.bitwise_and(all_masks_arr, np.uint64(uncov_mask)).bit_count(axis=1)
                            bi = int(np.argmax(gains))
                            if gains[bi] == 0:
                                break
                            add_block(bi)
                            uncov_mask &= ~all_masks[bi]
                else:
                    # add a random block not already present, then prune
                    c = random.randint(0, N_BLOCKS - 1)
                    if c not in current_block_set:
                        add_block(c)
                        # immediate prune pass
                        indices2 = list(range(len(current_blocks)))
                        random.shuffle(indices2)
                        for ii in indices2:
                            if time.time() > deadline - 3:
                                break
                            if is_redundant(current_blocks[ii]):
                                remove_block(current_blocks[ii])
                                if len(current_blocks) < best_len:
                                    best_len = len(current_blocks)
                                    best_blocks = current_blocks.copy()
                                break
                no_improve_count = 0
                max_no_improve = min(200, max_no_improve * 2)

    # Final cleanup: ensure we return the best found
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(current_blocks) - 1, -1, -1):
            idx = current_blocks[i]
            if is_redundant(idx):
                remove_block(idx)
                changed = True
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()

    return [list(all_blocks[i]) for i in best_blocks]
